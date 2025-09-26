# ga_deap_ray_trainer.py
# DEAP + Ray remote map GA with per-score resource control, deterministic seeding, and full logging.

from __future__ import annotations

import copy, hashlib, json, math, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ray
from matplotlib import pyplot as plt
from deap import base, creator, tools

from mdp_network import MDPNetwork
from mdp_network.mdp_tables import q_table_to_policy, create_random_policy
from mdp_network.solvers import optimal_value_iteration, compute_occupancy_measure
from genetic_algorithms.mdp_ops import (
    _list_all_triples, _prune_low_prob_transitions, _mutation_add_edge,
    _mutation_prob_pairwise, _mutation_reward_smallstep, _crossover_action_block
)
# Need outputs registry to map metrics -> score functions (no extra helpers elsewhere)
from genetic_algorithms.score_fns import SCORE_FNS, SCORE_FN_OUTPUTS  # fn name -> callable / outputs list


# ============== small, local utilities (kept minimal) ==============
def derive_seed(master_seed: int, *tags: Any) -> int:
    # Deterministic 64-bit seed from (master_seed, tags...)
    h = hashlib.sha256(); h.update(str(int(master_seed)).encode("utf-8"))
    for t in tags: h.update(b"::"); h.update(str(t).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little", signed=False)

def as_uint32(x: int) -> int:
    # Map int to [0, 2**32-1] for NumPy legacy seeding compatibility
    return int(x) & 0xFFFFFFFF

def safe_name(s: str) -> str:
    t = s.replace("/", "__").replace("\\", "__").replace(" ", "_")
    return "".join(c for c in t if (c.isalnum() or c in "._-+%=:@#[]{}()"))

def jsonl_append(path: str, rec: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def summ_stats(objs: List[List[float]]) -> Dict[str, float]:
    if not objs: return {}
    arr = np.asarray(objs, dtype=float); M = arr.shape[1]
    return {f"{s}_{m}": float(getattr(np, s)(arr[:, m])) for m in range(M) for s in ("min","mean","max")}

def update_metric_curves(metrics: List[Dict[str, Optional[float]]], gen: int,
                         hist: Dict[str, Dict[str, List[float]]], all_keys: set) -> None:
    # Per-metric independent x-axis: append only when metric appears at this gen.
    present = set()
    for md in metrics:
        for k, v in md.items():
            if v is not None and math.isfinite(float(v)): present.add(k)
    all_keys.update(present)
    for k in sorted(present):
        vals = [float(m[k]) for m in metrics if k in m and m[k] is not None and math.isfinite(float(m[k]))]
        if not vals: continue
        b = hist.setdefault(k, {"gen": [], "min": [], "mean": [], "max": []})
        b["gen"].append(int(gen)); b["min"].append(float(np.min(vals)))
        b["mean"].append(float(np.mean(vals))); b["max"].append(float(np.max(vals)))

def export_metric_curves(out_dir: Path, hist: Dict[str, Dict[str, List[float]]], wandb_writer=None) -> None:
    curves = out_dir / "metrics_curves"; plots = out_dir / "metrics_plots"
    curves.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)
    for key, series in hist.items():
        gens = series.get("gen", [])
        with open(curves / f"{safe_name(key)}.csv", "w", encoding="utf-8") as f:
            f.write("gen,min,mean,max\n")
            for g, a, b, c in zip(gens, series["min"], series["mean"], series["max"]):
                f.write(f"{g},{a},{b},{c}\n")
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
        axes[0].plot(gens, series["min"]);  axes[0].set_title(f"{key} - min")
        axes[1].plot(gens, series["mean"]); axes[1].set_title(f"{key} - mean")
        axes[2].plot(gens, series["max"]);  axes[2].set_title(f"{key} - max")
        for ax in axes: ax.set_xlabel("gen")
        png = plots / f"{safe_name(key)}.png"; fig.savefig(png, dpi=150); plt.close(fig)
        if wandb_writer is not None:
            wandb_writer.log_image.remote(key=f"ga_metric_plots/{key}", path=str(png), caption=f"{key} (min/mean/max)")
    if wandb_writer is not None:
        wandb_writer.log_artifact_dir.remote("ga-metric-plots", "plots", str(plots), {"kind":"metric_curves_pngs"})

def log_best_by_keys_for_batch(gen_to_log: int, metrics: List[Dict[str, Optional[float]]],
                               keys: List[str], wandb_writer=None) -> None:
    if wandb_writer is None or not keys: return
    for key in keys:
        best_idx, best_val = None, None
        for i, md in enumerate(metrics):
            v = md.get(key, None)
            if v is None or not math.isfinite(float(v)): continue
            if (best_val is None) or (float(v) > best_val): best_val, best_idx = float(v), i
        if best_idx is None: continue
        payload = {"ga/gen": int(gen_to_log)}
        for mk, mv in metrics[best_idx].items():
            if mv is not None and math.isfinite(float(mv)):
                payload[f"ga_metrics_max_{key}/{mk}"] = float(mv)
        wandb_writer.log.remote(payload)


# ============== Ray scoring tasks ==============

@dataclass
class ScoreItem:
    name: str
    params: Dict[str, Any]
    resources: Dict[str, Any]  # {"cpus": float|int, "gpus": float|int, "resources": {...}}
    retries: int = 0
    timeout_s: Optional[float] = None

def normalize_score_spec(spec: Any) -> List[ScoreItem]:
    if not isinstance(spec, list): raise TypeError("score_spec must be a list of dicts.")
    out: List[ScoreItem] = []
    for it in spec:
        if not isinstance(it, dict) or "name" not in it: raise TypeError("Each score item needs 'name'.")
        name = str(it["name"])
        if name not in SCORE_FNS: raise KeyError(f"Unknown score '{name}'.")
        params = dict(it.get("params", {}))
        res = dict(it.get("resources", {})); res.setdefault("cpus", 1); res.setdefault("gpus", 0)
        out.append(ScoreItem(name=name, params=params, resources=res,
                             retries=int(it.get("retries", 0)), timeout_s=it.get("timeout_s", None)))
    return out

@ray.remote
def score_remote(score_name: str, score_params: Dict[str, Any],
                 mdp_portable: Dict[str, Any], shared: Dict[str, Any], seed: int
                 ) -> Dict[str, Optional[float]]:
    # Deterministic task seeding
    random.seed(int(seed)); np.random.seed(as_uint32(int(seed))); _ = np.random.default_rng(int(seed))
    mdp = MDPNetwork.from_portable(mdp_portable)
    out = SCORE_FNS[score_name](mdp, shared, **score_params)
    return {k: (None if v is None else float(v)) for k, v in out.items()}

@ray.remote
def evaluate_one_remote(mdp_portable_ref, score_items: List[Dict[str, Any]], objective_keys: List[str],
                        shared_ref, master_seed: int, gen: int, ind_idx: int
                        ) -> Tuple[List[float], Dict[str, Optional[float]]]:
    items = normalize_score_spec(score_items)
    launched: List[ray.ObjectRef] = []
    for i, it in enumerate(items):
        opts = {}
        if "cpus" in it.resources: opts["num_cpus"] = float(it.resources["cpus"])
        if "gpus" in it.resources: opts["num_gpus"] = float(it.resources["gpus"])
        if "resources" in it.resources and isinstance(it.resources["resources"], dict) and it.resources["resources"]:
            opts["resources"] = it.resources["resources"]
        seed = derive_seed(master_seed, "score", gen, ind_idx, it.name, i)
        launched.append(score_remote.options(**opts).remote(it.name, it.params, mdp_portable_ref, shared_ref, int(seed)))
    # Merge metrics
    metrics: Dict[str, Optional[float]] = {}
    for ref in launched:
        res = ray.get(ref)
        for k, v in res.items():
            if k in metrics: raise ValueError(f"Duplicate metric key: {k}")
            metrics[k] = v
    # Extract objectives in declared order
    obj_vals: List[float] = []
    for k in objective_keys:
        v = metrics.get(k, None)
        if v is None or not math.isfinite(float(v)): raise ValueError(f"Objective '{k}' missing/non-finite.")
        obj_vals.append(float(v))
    return obj_vals, metrics


# ============== GA Trainer (compact) ==============

class Trainer:
    """DEAP + Ray GA with per-gen fitness/eval schedule. Simple, compact, deterministic."""

    def __init__(self, base_mdp: MDPNetwork, population_size: int, generations: int, master_seed: int,
                 algo_type: str, mu: int, lambd: int, cxpb: float, survivor: str,
                 parents: str, parent_k: int, score_spec: List[Dict[str, Any]],
                 objective_keys: Optional[List[str]], max_metric_keys: Optional[Sequence[str]],
                 max_in_flight: int, ray_init: Optional[Dict[str, Any]],
                 ops: Optional[Dict[str, Any]], distance: Optional[Dict[str, Any]],
                 solver: Optional[Dict[str, Any]], output_dir: Optional[str], wandb_writer,
                 # NEW: schedules (both support blocks 'gens' and periodic 'every')
                 fitness_schedule: Optional[List[Dict[str, Any]]] = None,
                 eval_schedule: Optional[List[Dict[str, Any]]] = None,
                 eval_at_begin: Optional[Sequence[str] | str] = None,
                 eval_at_end: Optional[Sequence[str] | str] = None):
        self.base_mdp = base_mdp
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.master_seed = int(master_seed)
        self.algo_type = str(algo_type)
        self.mu = int(mu) if mu is not None else int(population_size)
        self.lambd = int(lambd) if lambd is not None else int(population_size)
        self.cxpb = float(cxpb)
        self.survivor = str(survivor)
        self.parents = str(parents)
        self.parent_k = int(parent_k)
        self.items = normalize_score_spec(score_spec)
        self.mmk_list = list(max_metric_keys) if isinstance(max_metric_keys, (list, tuple)) else ([] if max_metric_keys is None else [str(max_metric_keys)])
        self.max_in_flight = int(max_in_flight)
        self.ray_init = dict(ray_init or {})
        self.ops = dict(ops or {})
        self.distance = dict(distance or {})
        self.solver = dict(solver or {})
        self.output_dir = output_dir
        self.wandb_writer = wandb_writer

        if not ray.is_initialized(): ray.init(**self.ray_init)
        random.seed(int(self.master_seed))

        # DEAP types (fitness dim decided below)
        self.obj_keys: List[str] = list(objective_keys) if objective_keys else []

        # Schedules: keep structure minimal; precompute cycles for blocks
        def _norm(s: Optional[List[Dict[str, Any]]]) -> Tuple[List[Tuple[int, List[str]]], int, List[Tuple[int,int,List[str]]]]:
            # return (blocks:[(gens,keys)], cycle, periodic:[(every,offset,keys)])
            if not s: return [], 0, []
            blocks: List[Tuple[int, List[str]]] = []
            periodic: List[Tuple[int, int, List[str]]] = []
            for r in s:
                # allow 'objectives' or 'metrics' or 'keys'
                keys = r.get("keys");  keys = keys if keys is not None else r.get("objectives", r.get("metrics", []))
                keys = ["*"] if keys == "*" else [str(k) for k in (keys or [])]
                if "gens" in r:
                    blocks.append((int(r["gens"]), keys))
                elif "every" in r:
                    periodic.append((int(r["every"]), int(r.get("offset", 0)), keys))
                else:
                    raise ValueError("Schedule rule needs 'gens' or 'every'.")
            cyc = sum(g for g,_ in blocks)
            return blocks, cyc, periodic

        self.fit_blocks, self.fit_cycle, self.fit_periodic = _norm(fitness_schedule)
        self.eval_blocks, self.eval_cycle, self.eval_periodic = _norm(eval_schedule)
        self.eval_at_begin = eval_at_begin
        self.eval_at_end = eval_at_end

        # Fitness vector dim = union of all metric keys referenced by fitness schedule (unless given explicitly)
        if not self.obj_keys:
            u = set()
            for g, ks in self.fit_blocks:
                for k in ks:
                    if k != "*": u.add(k)
            for e, off, ks in self.fit_periodic:
                for k in ks:
                    if k != "*": u.add(k)
            self.obj_keys = sorted(u)
        if "FitnessMulti" not in creator.__dict__:
            creator.create("FitnessMulti", base.Fitness, weights=tuple([1.0]*len(self.obj_keys)))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", dict, fitness=creator.FitnessMulti)
        self.toolbox = base.Toolbox(); self.toolbox.register("clone", copy.deepcopy)

        # Domain caches and precompute
        self.whitelist = _list_all_triples(self.base_mdp)
        self.base_portable = self.base_mdp.to_portable()
        gamma = float(self.solver.get("vi_gamma", 0.99))
        theta = float(self.solver.get("vi_theta", 1e-6))
        max_iters = int(self.solver.get("vi_max_iterations", 1000))
        policy_temp = float(self.solver.get("policy_temperature", 1.0))
        policy_mix = tuple(self.solver.get("policy_mix", (0.0,1.0,0.0)))
        tie_tol = float(self.solver.get("policy_tie_tol", 1e-6))
        t0 = time.perf_counter()
        _, Q = optimal_value_iteration(self.base_mdp, gamma=gamma, theta=theta, max_iterations=max_iters)
        base_policy = q_table_to_policy(Q, list(self.base_mdp.states), self.base_mdp.num_actions,
                                        mixing=policy_mix, temperature=policy_temp, tie_tol=tie_tol)
        base_occ = compute_occupancy_measure(self.base_mdp, base_policy, gamma=gamma, theta=theta, max_iterations=max_iters)
        rand_policy = create_random_policy(self.base_mdp)
        self.precomputed = {
            "base_policy": base_policy.to_portable(),
            "base_occupancy": base_occ.to_portable(),
            "base_mdp": self.base_portable,
            "base_q": Q.to_portable(),
            "rand_policy": rand_policy.to_portable(),
        }
        self.shared_ref = ray.put({"solver": self.solver, "precomputed": self.precomputed})
        t1 = time.perf_counter()
        if self.wandb_writer is not None:
            self.wandb_writer.define_metric.remote("ga_metrics/*", step_metric="ga/gen")
            self.wandb_writer.define_metric.remote("ga_object/*",  step_metric="ga/gen")
            self.wandb_writer.define_metric.remote("ga_meta/*",    step_metric="ga/gen")
            self.wandb_writer.define_metric.remote("ga/pop/*",     step_metric="ga/gen")
            self.wandb_writer.define_metric.remote("ga/time/*",    step_metric="ga/gen")
            self.wandb_writer.define_metric.remote("ga/init/*",    step_metric="ga/gen")
            self.wandb_writer.define_metric.remote("ga/final/*",   step_metric="ga/gen")
            self.wandb_writer.define_metric.remote("ga_metric_plots/*", step_metric="ga/gen")
            for k in self.mmk_list:
                self.wandb_writer.define_metric.remote(f"ga_metrics_max_{k}/*", step_metric="ga/gen")
            self.wandb_writer.log.remote({"ga/time/precompute_sec": float(t1 - t0), "ga/gen": -1})

        # Output & curve state
        self.metrics_dir: Optional[Path] = None
        self.jsonl_path: Optional[str] = None
        if self.output_dir:
            self.metrics_dir = Path(self.output_dir) / "ga"
            (self.metrics_dir / "metrics_curves").mkdir(parents=True, exist_ok=True)
            (self.metrics_dir / "metrics_plots").mkdir(parents=True, exist_ok=True)
            (self.metrics_dir / "mdps").mkdir(parents=True, exist_ok=True)
            self.jsonl_path = str(self.metrics_dir / "metrics.jsonl")
        self.uid_counter = 0
        self.all_metric_keys: set = set()
        self.metrics_history: Dict[str, Dict[str, List[float]]] = {}
        self.fitness_schedule = fitness_schedule or []
        self.eval_schedule = eval_schedule or []

    # ---- tiny helpers inside class (no nesting outside) ----

    def _active_keys(self, gen: int, for_eval: bool) -> List[str]:
        # Union of block + periodic; ignore '*' (only for monitor-all)
        blocks, cyc, periodic = (self.eval_blocks, self.eval_cycle, self.eval_periodic) if for_eval \
                                else (self.fit_blocks, self.fit_cycle, self.fit_periodic)
        keys = set()
        if blocks and cyc > 0:
            pos = (max(1, gen) - 1) % cyc
            acc = 0
            for g, ks in blocks:
                acc += g
                if pos < acc:
                    for k in ks:
                        if k != "*": keys.add(k)
                    break
        for every, offset, ks in periodic:
            if every > 0 and (gen - offset) % every == 0:
                for k in ks:
                    if k != "*": keys.add(k)
        return sorted(keys) if keys else list(self.obj_keys)

    def _needed_score_items(self, metric_keys: Sequence[str]) -> List[Dict[str, Any]]:
        # Minimal set of score functions that produce the requested metrics
        need = set()
        for fn_name, outs in SCORE_FN_OUTPUTS.items():
            if any(m in outs for m in metric_keys): need.add(fn_name)
        out: List[Dict[str, Any]] = []
        for s in self.items:
            if s.name in need:
                out.append(dict(name=s.name, params=s.params, resources=s.resources,
                                retries=s.retries, timeout_s=s.timeout_s))
        return out

    # ---- domain ops ----

    def domain_crossover(self, pa_p: Dict[str, Any], pb_p: Dict[str, Any], seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(int(seed))
        prob_floor = float(self.ops.get("prob_floor", 1e-6))
        pa = MDPNetwork.from_portable(pa_p); pb = MDPNetwork.from_portable(pb_p)
        child = _crossover_action_block(pa, pb, rng, whitelist=set(tuple(x) for x in self.whitelist), prob_floor=prob_floor)
        return child.to_portable()

    def domain_mutate(self, p_in: Dict[str, Any], seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(int(seed))
        prob_floor = float(self.ops.get("prob_floor", 1e-6))
        ind = MDPNetwork.from_portable(p_in).clone()
        dist = {
            "max_hops": self.distance.get("dist_max_hops", self.distance.get("max_hops", None)),
            "node_cap": self.distance.get("dist_node_cap", self.distance.get("node_cap", None)),
            "weight_eps": float(self.distance.get("dist_weight_eps", self.distance.get("weight_eps", 1e-9))),
            "unreachable": float(self.distance.get("dist_unreachable", self.distance.get("unreachable", 1e6))),
        }
        for _ in range(int(self.ops.get("add_edge_attempts_per_child", 2))):
            _mutation_add_edge(ind, rng, MDPNetwork.from_portable(self.base_portable), self.ops, dist,
                               whitelist=set(tuple(x) for x in self.whitelist))
        _mutation_prob_pairwise(ind, rng, self.ops, whitelist=set(tuple(x) for x in self.whitelist))
        if int(self.ops.get("reward_tweak_edges_per_child", 50)) > 0:
            _mutation_reward_smallstep(ind, rng, self.ops)
        if self.ops.get("prune_prob_threshold", None) is not None:
            _prune_low_prob_transitions(ind, float(self.ops["prune_prob_threshold"]),
                                        whitelist=set(tuple(x) for x in self.whitelist), prob_floor=prob_floor)
        return ind.to_portable()

    # ---- evaluation & logging ----
    def _eval_on_metrics(self, inds: List[Any], gen: int, metric_keys: Sequence[str]) -> Tuple[
        List[List[float]], List[Dict[str, Optional[float]]]]:
        """Evaluate only the requested metrics for monitoring (independent from fitness).
        - Expand '*' to all outputs producible by current score_fns.
        - Launch minimal set of score tasks and append curves at this gen.
        """

        # 1) Resolve requested metric names (expand '*')
        if metric_keys == "*" or (isinstance(metric_keys, (list, tuple)) and "*" in metric_keys):
            req_keys: List[str] = sorted({mk for s in self.items for mk in SCORE_FN_OUTPUTS.get(s.name, [])})
        else:
            req_keys = [str(k) for k in metric_keys]

        # 2) Minimal score set that can produce these metrics
        score_items = self._needed_score_items(req_keys)
        if not score_items:
            return [], []

        # 3) Print eval plan
        print(
            f"[Plan][EVAL] gen={gen} | keys=[{', '.join(req_keys)}] | scores=[{', '.join(s['name'] for s in score_items)}]")

        # 4) Launch evaluations; objective_keys empty -> no strict extraction
        pending: List[ray.ObjectRef] = []
        results: List[Tuple[List[float], Dict[str, Optional[float]]]] = []
        for idx, ind in enumerate(inds):
            mdp_ref = ray.put(ind["portable"])
            ref = evaluate_one_remote.remote(
                mdp_ref, score_items, [], self.shared_ref, int(self.master_seed), int(gen), int(idx)
            )
            pending.append(ref)
            if len(pending) >= self.max_in_flight:
                ready, pending = ray.wait(pending, num_returns=1);
                results.append(ray.get(ready[0]))
        while pending:
            ready, pending = ray.wait(pending, num_returns=1);
            results.append(ray.get(ready[0]))

        # 5) Keep only requested metrics
        filtered: List[Dict[str, Optional[float]]] = []
        for (_obj_vals, md) in results:
            filt = {k: md.get(k, None) for k in req_keys if k in md}
            filtered.append(filt)

        # 6) Append per-metric curves at this gen
        update_metric_curves(filtered, gen, self.metrics_history, self.all_metric_keys)
        return [], filtered

    # ---- evaluation & logging ----
    def eval_population(self, pop_inds: List[Any], gen: int) -> Tuple[
        List[List[float]], List[Dict[str, Optional[float]]]]:
        # NOTE: remove plan print here to avoid duplicates; we print only from run().
        pending: List[ray.ObjectRef] = []
        results: List[Tuple[List[float], Dict[str, Optional[float]]]] = []

        for idx, ind in enumerate(pop_inds):
            mdp_ref = ray.put(ind["portable"])  # put once per individual
            ref = evaluate_one_remote.remote(
                mdp_ref,
                [dict(name=s.name, params=s.params, resources=s.resources,
                      retries=s.retries, timeout_s=s.timeout_s) for s in self.items],
                self.obj_keys,
                self.shared_ref,  # shared heavy data by reference
                int(self.master_seed), int(gen), int(idx)
            )
            pending.append(ref)

            if len(pending) >= self.max_in_flight:
                ready, pending = ray.wait(pending, num_returns=1)
                results.append(ray.get(ready[0]))

        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            results.append(ray.get(ready[0]))

        objs: List[List[float]] = []
        metrics: List[Dict[str, Optional[float]]] = []
        for (o, md) in results:
            objs.append([float(x) for x in o])
            metrics.append({k: (None if v is None else float(v)) for k, v in md.items()})

        for ind, o in zip(pop_inds, objs):
            ind.fitness.values = tuple(o)

        return objs, metrics

    def log_batch(self, gen: int, metrics: List[Dict[str, Optional[float]]], is_child: bool) -> None:
        if self.jsonl_path:
            for idx, md in enumerate(metrics):
                payload = {
                    "gen": int(gen), "uid": int(self.uid_counter + idx), "ind_in_gen": int(idx),
                    "is_child": bool(is_child), "objective_keys": list(self.obj_keys), "metrics": md
                }
                jsonl_append(self.jsonl_path, payload)
            self.uid_counter += len(metrics)
        update_metric_curves(metrics, gen, self.metrics_history, self.all_metric_keys)

    def wb_log_survivors(self, gen: int, survivors: List[Any], metrics: List[Dict[str, Optional[float]]]) -> None:
        if self.wandb_writer is None: return
        for ind_in_gen, md in enumerate(metrics):
            payload = {"ga/gen": int(gen), "ga_meta/ind_in_gen": int(ind_in_gen), "ga_meta/is_child": 0}
            for mk, mv in md.items():
                if mv is not None and math.isfinite(float(mv)): payload[f"ga_metrics/{mk}"] = float(mv)
            for ok in self.obj_keys:
                v = md.get(ok, None)
                if v is not None and math.isfinite(float(v)): payload[f"ga_object/{ok}"] = float(v)
            self.wandb_writer.log.remote(payload)
        if self.mmk_list: log_best_by_keys_for_batch(gen, metrics, list(self.mmk_list), self.wandb_writer)

    # ---- selection helpers ----

    def select_survivors(self, pool: List[Any], size: int) -> List[Any]:
        if self.survivor == "nsga2": return tools.selNSGA2(pool, size)
        if self.survivor == "spea2": return tools.selSPEA2(pool, size)
        if self.survivor == "nsga3": return tools.selNSGA2(pool, size)
        raise ValueError(f"Unknown survivor selector: {self.survivor}")

    def select_parents_pair(self, population: List[Any]) -> Tuple[Any, Any]:
        if self.parents == "tournament_dcd":
            return tools.selTournamentDCD(population, 1)[0], tools.selTournamentDCD(population, 1)[0]
        if self.parents == "tournament":
            return tools.selTournament(population, 1, tournsize=int(self.parent_k))[0], \
                   tools.selTournament(population, 1, tournsize=int(self.parent_k))[0]
        if self.parents == "roulette":
            return tools.selRoulette(population, 1)[0], tools.selRoulette(population, 1)[0]
        raise ValueError(f"Unknown parent selector: {self.parents}")

    def run(self) -> Tuple[List[MDPNetwork], List[List[float]], List[MDPNetwork], List[List[float]]]:
        # Fixed objective layout (DEAP needs constant fitness dimension)
        obj_idx = {k: i for i, k in enumerate(self.obj_keys)}

        # Parse fitness schedule into blocks and cycle length
        fit_blocks: List[Tuple[int, List[str]]] = [];
        fit_cycle = 0
        for r in (self.fitness_schedule or []):
            if "gens" in r:
                ks = r.get("keys") or r.get("objectives") or r.get("metrics") or []
                ks = list(self.obj_keys) if ks == "*" else [str(k) for k in ks]
                g = int(r["gens"])
                if g > 0: fit_blocks.append((g, ks)); fit_cycle += g

        # Parse eval schedule into blocks and cycle length
        eval_blocks: List[Tuple[int, List[str]]] = [];
        eval_cycle = 0
        for r in (self.eval_schedule or []):
            if "gens" in r:
                ks = r.get("keys") or r.get("objectives") or r.get("metrics") or []
                ks = list(self.obj_keys) if ks == "*" else [str(k) for k in ks]
                g = int(r["gens"])
                if g > 0: eval_blocks.append((g, ks)); eval_cycle += g

        # Inner helper to pick active block keys (no new global helpers)
        def _active_from_blocks(blocks: List[Tuple[int, List[str]]], cycle: int, gen_idx: int) -> List[str]:
            if not blocks or cycle <= 0: return list(self.obj_keys)
            pos = (cycle - 1) if gen_idx == 0 else ((gen_idx - 1) % cycle)
            acc = 0
            for span, ks in blocks:
                acc += span
                if pos < acc: return list(ks)
            return list(self.obj_keys)

        # Build initial population
        pop: List[Any] = [creator.Individual({"portable": self.base_portable})]
        for i in range(self.population_size - 1):
            p_child = self.domain_mutate(self.base_portable, seed=derive_seed(self.master_seed, "init", i))
            pop.append(creator.Individual({"portable": p_child}))

        # -------------------- gen=0 (init) --------------------
        active0 = _active_from_blocks(fit_blocks, fit_cycle, gen_idx=0)

        # Resolve begin-eval keys: eval_at_begin > eval_schedule(last block) > active0
        if self.eval_at_begin is not None:
            begin_keys = "*" if self.eval_at_begin == "*" else (
                list(self.eval_at_begin) if isinstance(self.eval_at_begin, (list, tuple)) else [
                    str(self.eval_at_begin)])
        elif eval_blocks:
            begin_keys = _active_from_blocks(eval_blocks, eval_cycle, gen_idx=0)
        else:
            begin_keys = list(active0)

        ek_str = "ALL" if begin_keys == "*" else ", ".join(begin_keys)
        print(f"[Plan] gen=0 | fitness objs=[{', '.join(active0)}] | eval keys=[{ek_str}]")

        # Full eval for fitness vector (keep original behavior), then zero-out inactive dims
        objs, pop_metrics = self.eval_population(pop, gen=0)
        for ind, o_full in zip(pop, objs):
            vec = [0.0] * len(self.obj_keys)
            for k in active0:
                if k in obj_idx:
                    vec[obj_idx[k]] = float(o_full[obj_idx[k]])
            ind.fitness.values = tuple(vec)

        # Begin overlay eval strictly follows begin_keys and REPLACES what we log at gen=0
        if begin_keys:
            _, begin_metrics = self._eval_on_metrics(pop, gen=0, metric_keys=begin_keys)
            pop_metrics = begin_metrics  # log curves/jsonl using the explicit begin-eval set
        if self.wandb_writer is not None:
            self.wb_log_survivors(0, pop, pop_metrics)

        # Log gen=0 (uses begin overlay if present)
        self.log_batch(0, pop_metrics, is_child=False)
        pop = tools.selNSGA2(pop, len(pop))

        # Optional init summary (unchanged min/mean/max format)
        init_stats = summ_stats(objs)
        if objs:
            M = len(objs[0]);
            parts = []
            for m in range(M):
                parts.append(
                    f"obj{m}: min={init_stats.get(f'min_{m}', float('nan')):.4f} "
                    f"mean={init_stats.get(f'mean_{m}', float('nan')):.4f} "
                    f"max={init_stats.get(f'max_{m}', float('nan')):.4f}"
                )
            print(f"[Init] pop={len(pop)} | " + " | ".join(parts))
        else:
            print(f"[Init] pop={len(pop)} | NA")

        if self.wandb_writer is not None:
            payload = {"ga/init/pop_size": int(len(pop)), "ga/gen": 0}
            M = len(objs[0]) if objs else 0
            for m in range(M):
                payload[f"ga/init/obj{m}_min"] = init_stats.get(f"min_{m}", float("nan"))
                payload[f"ga/init/obj{m}_mean"] = init_stats.get(f"mean_{m}", float("nan"))
                payload[f"ga/init/obj{m}_max"] = init_stats.get(f"max_{m}", float("nan"))
            self.wandb_writer.log.remote(payload)

        # -------------------- evolution --------------------
        for gen in range(self.generations):
            g1 = gen + 1
            if self.parents == "tournament_dcd":
                pop = tools.selNSGA2(pop, len(pop))

            # (A) active fitness objectives this gen
            active_fit = _active_from_blocks(fit_blocks, fit_cycle, gen_idx=g1)
            print(f"[Plan] gen={g1} | active objs=[{', '.join(active_fit)}] | eval keys=[{', '.join(active_fit)}]")

            # Minimal score set for active fitness
            need_fns = set()
            for fn_name, outs in SCORE_FN_OUTPUTS.items():
                if any(k in outs for k in active_fit): need_fns.add(fn_name)
            score_items = [
                dict(name=s.name, params=s.params, resources=s.resources, retries=s.retries, timeout_s=s.timeout_s)
                for s in self.items if s.name in need_fns]

            # Re-evaluate parents on active fitness only; other dims -> 0
            pending, par_results = [], []
            for idx, ind in enumerate(pop):
                mdp_ref = ray.put(ind["portable"])
                ref = evaluate_one_remote.remote(mdp_ref, score_items, list(active_fit),
                                                 self.shared_ref, int(self.master_seed), int(g1), int(idx))
                pending.append(ref)
                if len(pending) >= self.max_in_flight:
                    ready, pending = ray.wait(pending, num_returns=1);
                    par_results.append(ray.get(ready[0]))
            while pending:
                ready, pending = ray.wait(pending, num_returns=1);
                par_results.append(ray.get(ready[0]))
            par_objs_active = [list(o) for (o, _) in par_results]
            for ind, o_act in zip(pop, par_objs_active):
                full_vec = [0.0] * len(self.obj_keys)
                for k, v in zip(active_fit, o_act):
                    if k in obj_idx: full_vec[obj_idx[k]] = float(v)
                ind.fitness.values = tuple(full_vec)

            # Children
            children: List[Any] = []
            for k in range(self.lambd):
                pa, pb = self.select_parents_pair(pop)
                seed_child = derive_seed(self.master_seed, "child", gen, k)
                if random.random() < self.cxpb:
                    c_portable = self.domain_crossover(pa["portable"], pb["portable"],
                                                       seed=derive_seed(seed_child, "cx"))
                else:
                    src = pa if (random.random() < 0.5) else pb
                    c_portable = dict(src["portable"])
                c_portable = self.domain_mutate(c_portable, seed=derive_seed(seed_child, "mut"))
                children.append(creator.Individual({"portable": c_portable}))

            # Evaluate children on active fitness only
            pending, child_results = [], []
            for idx, ind in enumerate(children):
                mdp_ref = ray.put(ind["portable"])
                ref = evaluate_one_remote.remote(mdp_ref, score_items, list(active_fit),
                                                 self.shared_ref, int(self.master_seed), int(g1), int(idx))
                pending.append(ref)
                if len(pending) >= self.max_in_flight:
                    ready, pending = ray.wait(pending, num_returns=1);
                    child_results.append(ray.get(ready[0]))
            while pending:
                ready, pending = ray.wait(pending, num_returns=1);
                child_results.append(ray.get(ready[0]))
            child_objs_active = [list(o) for (o, _) in child_results]
            child_metrics = [md for (_, md) in child_results]
            for ind, o_act in zip(children, child_objs_active):
                full_vec = [0.0] * len(self.obj_keys)
                for k, v in zip(active_fit, o_act):
                    if k in obj_idx: full_vec[obj_idx[k]] = float(v)
                ind.fitness.values = tuple(full_vec)

            # Log only computed child metrics this gen
            self.log_batch(g1, child_metrics, is_child=True)

            # Selection
            union = pop + children
            survivors = self.select_survivors(union, self.population_size)

            # (B) survivors eval keys follow eval_schedule (fallback to active fitness)
            eval_keys = _active_from_blocks(eval_blocks, eval_cycle, gen_idx=g1) if eval_blocks else list(active_fit)
            print(f"[Plan] gen={g1} (survivors) | eval keys=[{', '.join(eval_keys)}]")

            # Minimal score set for eval
            need_fns_eval = set()
            for fn_name, outs in SCORE_FN_OUTPUTS.items():
                if any(k in outs for k in eval_keys): need_fns_eval.add(fn_name)
            score_items_eval = [
                dict(name=s.name, params=s.params, resources=s.resources, retries=s.retries, timeout_s=s.timeout_s)
                for s in self.items if s.name in need_fns_eval]

            # Evaluate survivors for monitoring only (do not overwrite fitness)
            pending, surv_results = [], []
            for idx, ind in enumerate(survivors):
                mdp_ref = ray.put(ind["portable"])
                ref = evaluate_one_remote.remote(mdp_ref, score_items_eval, list(eval_keys),
                                                 self.shared_ref, int(self.master_seed), int(g1), int(idx))
                pending.append(ref)
                if len(pending) >= self.max_in_flight:
                    ready, pending = ray.wait(pending, num_returns=1);
                    surv_results.append(ray.get(ready[0]))
            while pending:
                ready, pending = ray.wait(pending, num_returns=1);
                surv_results.append(ray.get(ready[0]))

            surv_objs = [list(o) for (o, _) in surv_results]
            surv_metrics = [md for (_, md) in surv_results]
            self.wb_log_survivors(g1, survivors, surv_metrics)
            update_metric_curves(surv_metrics, gen=g1, hist=self.metrics_history, all_keys=self.all_metric_keys)

            # Gen summary on eval_keys
            if surv_objs and eval_keys:
                arr = np.asarray(surv_objs, dtype=float);
                parts = []
                for j, key in enumerate(eval_keys):
                    col = arr[:, j]
                    parts.append(
                        f"{key}: min={float(np.min(col)):.4f} mean={float(np.mean(col)):.4f} max={float(np.max(col)):.4f}")
                print(f"[Gen {g1}/{self.generations}] pop={len(survivors)} | " + " | ".join(parts))
            else:
                print(f"[Gen {g1}/{self.generations}] pop={len(survivors)} | NA")

            if self.wandb_writer is not None:
                self.wandb_writer.log.remote({"ga/pop/size": int(len(survivors)),
                                              "ga/time/total_gen_sec": float(time.perf_counter()),
                                              "ga/gen": int(g1)})

            pop = survivors

        # -------------------- eval_at_end overlay (optional) --------------------
        if self.eval_at_end is not None:
            end_keys = "*" if self.eval_at_end == "*" else (
                list(self.eval_at_end) if isinstance(self.eval_at_end, (list, tuple)) else [str(self.eval_at_end)])
            ek_end_str = "ALL" if end_keys == "*" else ", ".join(end_keys)
            print(f"[Plan] gen={self.generations} (end) | eval keys=[{ek_end_str}]")
            _, end_metrics = self._eval_on_metrics(pop, gen=self.generations, metric_keys=end_keys)
            if self.wandb_writer is not None:
                self.wb_log_survivors(self.generations, pop, end_metrics)
            # Record an extra monitoring row at the last gen
            self.log_batch(self.generations, end_metrics, is_child=False)

        # -------- final Pareto (unchanged) --------
        final_sorted = tools.selNSGA2(pop, len(pop))
        fronts = tools.sortNondominated(final_sorted, k=len(final_sorted), first_front_only=False)
        F1 = fronts[0] if fronts else final_sorted
        pareto_mdps = [MDPNetwork.from_portable(ind["portable"]).clone() for ind in F1]
        pareto_objs = [list(ind.fitness.values) for ind in F1]

        if self.metrics_dir is not None:
            out = self.metrics_dir / "mdps"
            for i, (m, objv) in enumerate(zip(pareto_mdps, pareto_objs)):
                tag = "_".join(f"{v:.4f}" for v in objv)
                path = out / f"pareto_{i}_objs_{tag}.json"
                m.export_to_json(str(path))
                print(f"[GA] Saved PF[{i}] -> {path.name}")

        if self.wandb_writer is not None:
            payload = {"ga/final/F1_size": int(len(F1)), "ga/gen": int(self.generations)}
            fstats = summ_stats(
                [list(ind.fitness.values) for ind in F1] if F1 else [list(ind.fitness.values) for ind in pop])
            M = len(pareto_objs[0]) if pareto_objs else 0
            for m in range(M):
                payload[f"ga/final/obj{m}_min"] = fstats.get(f"min_{m}", float("nan"))
                payload[f"ga/final/obj{m}_mean"] = fstats.get(f"mean_{m}", float("nan"))
                payload[f"ga/final/obj{m}_max"] = fstats.get(f"max_{m}", float("nan"))
            self.wandb_writer.log.remote(payload)

        if self.metrics_dir is not None:
            export_metric_curves(self.metrics_dir, self.metrics_history, self.wandb_writer)

        final_mdps = [MDPNetwork.from_portable(ind["portable"]).clone() for ind in pop]
        final_objs = [list(ind.fitness.values) for ind in pop]
        return pareto_mdps, pareto_objs, final_mdps, final_objs


# ============== public wrapper ==============
def run_ga_deap_ray(*, base_mdp: MDPNetwork, population_size: int, generations: int, master_seed: int,
                    algo_type: str = "mu_plus_lambda", mu: Optional[int] = None, lambd: Optional[int] = None,
                    cxpb: float = 1.0, survivor: str = "nsga2", parents: str = "tournament_dcd", parent_k: int = 2,
                    score_spec: List[Dict[str, Any]] = None, objective_keys: Optional[List[str]] = None,
                    max_metric_keys: Optional[Sequence[str]] = None, max_in_flight: int = 128,
                    ray_init: Optional[Dict[str, Any]] = None, ops: Optional[Dict[str, Any]] = None,
                    distance: Optional[Dict[str, Any]] = None, solver: Optional[Dict[str, Any]] = None,
                    output_dir: Optional[str] = None, wandb_writer=None,
                    fitness_schedule: Optional[List[Dict[str, Any]]] = None,
                    eval_schedule: Optional[List[Dict[str, Any]]] = None,
                    eval_at_begin: Optional[Sequence[str] | str] = None,
                    eval_at_end: Optional[Sequence[str] | str] = None
                    ) -> Tuple[List[MDPNetwork], List[List[float]], List[MDPNetwork], List[List[float]]]:
    t = Trainer(base_mdp, population_size, generations, master_seed, algo_type, mu, lambd, cxpb,
                survivor, parents, parent_k, score_spec or [], objective_keys, max_metric_keys,
                max_in_flight, ray_init, ops, distance, solver, output_dir, wandb_writer,
                fitness_schedule=fitness_schedule, eval_schedule=eval_schedule,
                eval_at_begin=eval_at_begin, eval_at_end=eval_at_end)
    return t.run()
