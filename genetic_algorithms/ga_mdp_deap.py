# ga_deap_ray_trainer.py
# DEAP + Ray remote map GA with per-score resource control, deterministic seeding, and full logging.

from __future__ import annotations

import copy
import hashlib, json, math, os, random, time, logging
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
from genetic_algorithms.score_fns import SCORE_FNS  # name -> fn(mdp, shared, **params) -> Dict[str, float|None]


# =========================
# Small utilities (global)
# =========================

def derive_seed(master_seed: int, *tags: Any) -> int:
    """Deterministic 64-bit seed from (master_seed, tags...)."""
    h = hashlib.sha256()
    h.update(str(int(master_seed)).encode("utf-8"))
    for t in tags:
        h.update(b"::"); h.update(str(t).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little", signed=False)

def as_uint32(x: int) -> int:
    # map arbitrary int to [0, 2**32 - 1] for numpy RandomState compatibility
    return int(x) & 0xFFFFFFFF

def safe_name(s: str) -> str:
    t = s.replace("/", "__").replace("\\", "__").replace(" ", "_")
    return "".join(c for c in t if (c.isalnum() or c in "._-+%=:@#[]{}()"))

def jsonl_append(path: str, rec: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def finite_vals(vals: Sequence[Optional[float]]) -> List[float]:
    out: List[float] = []
    for v in vals:
        if v is None: continue
        fv = float(v)
        if math.isfinite(fv): out.append(fv)
    return out

def summ_stats(objs: List[List[float]]) -> Dict[str, float]:
    if not objs: return {}
    arr = np.asarray(objs, dtype=float); M = arr.shape[1]
    return {f"{s}_{m}": float(getattr(np, s)(arr[:, m])) for m in range(M) for s in ("min","mean","max")}

def update_metric_curves(metrics: List[Dict[str, Optional[float]]],
                         hist: Dict[str, Dict[str, List[float]]],
                         all_keys: set) -> None:
    for md in metrics: all_keys.update(md.keys())
    for k in sorted(all_keys):
        f = finite_vals([m.get(k, None) for m in metrics])
        bucket = hist.setdefault(k, {"min":[], "mean":[], "max":[]})
        if f:
            bucket["min"].append(float(np.min(f)))
            bucket["mean"].append(float(np.mean(f)))
            bucket["max"].append(float(np.max(f)))
        else:
            nan = float("nan"); bucket["min"].append(nan); bucket["mean"].append(nan); bucket["max"].append(nan)

def export_metric_curves(out_dir: Path, hist: Dict[str, Dict[str, List[float]]], wandb_writer=None) -> None:
    curves = out_dir / "metrics_curves"; plots = out_dir / "metrics_plots"
    curves.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)
    for key, series in hist.items():
        with open(curves / f"{safe_name(key)}.csv", "w", encoding="utf-8") as f:
            f.write("gen,min,mean,max\n")
            for g,(a,b,c) in enumerate(zip(series["min"],series["mean"],series["max"])):
                f.write(f"{g},{a},{b},{c}\n")
        gens = list(range(len(series["min"])))
        fig, axes = plt.subplots(3,1, figsize=(8,9), constrained_layout=True)
        axes[0].plot(gens, series["min"]);  axes[0].set_title(f"{key} - min")
        axes[1].plot(gens, series["mean"]); axes[1].set_title(f"{key} - mean")
        axes[2].plot(gens, series["max"]);  axes[2].set_title(f"{key} - max")
        for ax in axes: ax.set_xlabel("gen")
        png = plots / f"{safe_name(key)}.png"; fig.savefig(png, dpi=150); plt.close(fig)
        if wandb_writer is not None:
            wandb_writer.log_image.remote(key=f"ga_metric_plots/{key}", path=str(png), caption=f"{key} (min/mean/max)")
    if wandb_writer is not None:
        wandb_writer.log_artifact_dir.remote(name="ga-metric-plots", a_type="plots",
                                             dir_path=str(plots), metadata={"kind":"metric_curves_pngs"})

def log_best_by_keys_for_batch(gen_to_log: int, metrics: List[Dict[str, Optional[float]]],
                               keys: List[str], wandb_writer=None) -> None:
    if wandb_writer is None or not keys: return
    for key in keys:
        best_idx, best_val = None, None
        for i, md in enumerate(metrics):
            v = md.get(key, None);
            if v is None: continue
            fv = float(v)
            if not math.isfinite(fv): continue
            if (best_val is None) or (fv > best_val): best_val, best_idx = fv, i
        if best_idx is None: raise ValueError(f"No finite values for key '{key}' at gen={gen_to_log}.")
        payload = {"ga/gen": int(gen_to_log)}
        for mk, mv in metrics[best_idx].items():
            if mv is None: continue
            fv = float(mv)
            if math.isfinite(fv): payload[f"ga_metrics_max_{key}/{mk}"] = fv
        wandb_writer.log.remote(payload)

def ray_map(func, iterable: List[Any], max_in_flight: int = 128) -> List[Any]:
    """Submit func(x) via Ray tasks; preserve submission order; apply backpressure."""
    if not ray.is_initialized(): ray.init(ignore_reinit_error=True)
    @ray.remote
    def _wrap(arg): return func(arg)
    refs, out = [], []
    for x in iterable:
        if len(refs) >= max_in_flight:
            ready, refs = ray.wait(refs, num_returns=1)
            out.append(ray.get(ready[0]))
        refs.append(_wrap.remote(x))
    while refs:
        ready, refs = ray.wait(refs, num_returns=1)
        out.append(ray.get(ready[0]))
    return out


# =========================
# Score spec + remote tasks
# =========================

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
                             retries=int(it.get("retries", 0)),
                             timeout_s=it.get("timeout_s", None)))
    return out


@ray.remote
def score_remote(score_name: str,
                 score_params: Dict[str, Any],
                 mdp_portable: Dict[str, Any],      # may be an ObjectRef; Ray auto-derefs
                 shared: Dict[str, Any],             # may be an ObjectRef; Ray auto-derefs
                 seed: int) -> Dict[str, Optional[float]]:
    # Task-scoped deterministic seeding (covers both random and NumPy)
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(as_uint32(seed_i))
    _ = np.random.default_rng(seed_i)

    mdp = MDPNetwork.from_portable(mdp_portable)
    fn = SCORE_FNS[score_name]
    res = fn(mdp, shared, **score_params)

    # Strict float coercion
    return {k: (None if v is None else float(v)) for k, v in res.items()}


@ray.remote
def evaluate_one_remote(mdp_portable_ref,                 # pass ObjectRef
                        score_items: List[Dict[str, Any]],
                        objective_keys: List[str],
                        shared_ref,                       # pass ObjectRef
                        master_seed: int, gen: int, ind_idx: int
                        ) -> Tuple[List[float], Dict[str, Optional[float]]]:
    items = normalize_score_spec(score_items)

    # Fan-out scores; forward refs so mdp/shared are not re-copied
    launched: List[Tuple[str, ray.ObjectRef]] = []
    for i, it in enumerate(items):
        seed = derive_seed(master_seed, "score", gen, ind_idx, it.name, i)
        opts = {}
        if "cpus" in it.resources: opts["num_cpus"] = float(it.resources["cpus"])
        if "gpus" in it.resources: opts["num_gpus"] = float(it.resources["gpus"])
        if "resources" in it.resources and isinstance(it.resources["resources"], dict) and it.resources["resources"]:
            opts["resources"] = it.resources["resources"]

        ref = score_remote.options(**opts).remote(
            it.name, it.params, mdp_portable_ref, shared_ref, int(seed)
        )
        launched.append((it.name, ref))

    # Gather and merge metrics
    metrics: Dict[str, Optional[float]] = {}
    for name, ref in launched:
        res = ray.get(ref)
        for k, v in res.items():
            if k in metrics:
                raise ValueError(f"Duplicate metric key: {k}")
            metrics[k] = v

    # Strict objective extraction
    obj_vals: List[float] = []
    for k in objective_keys:
        v = metrics.get(k, None)
        if v is None or not math.isfinite(float(v)):
            raise ValueError(f"Objective '{k}' missing/non-finite.")
        obj_vals.append(float(v))

    return obj_vals, metrics


# =========================
# Trainer class
# =========================

class Trainer:
    """GA trainer using DEAP + Ray remote-map; concise, no nested functions."""

    def __init__(self, base_mdp: MDPNetwork, population_size: int, generations: int, master_seed: int,
                 algo_type: str, mu: int, lambd: int, cxpb: float, survivor: str,
                 parents: str, parent_k: int, score_spec: List[Dict[str, Any]],
                 objective_keys: List[str], max_metric_keys: Optional[Sequence[str]],
                 max_in_flight: int, ray_init: Optional[Dict[str, Any]],
                 ops: Optional[Dict[str, Any]], distance: Optional[Dict[str, Any]],
                 solver: Optional[Dict[str, Any]], output_dir: Optional[str], wandb_writer):
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
        self.obj_keys = list(objective_keys)
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

        self.logger = logging.getLogger("ga")
        if not self.logger.handlers:
            h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")); self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

        if "FitnessMulti" not in creator.__dict__:
            creator.create("FitnessMulti", base.Fitness, weights=tuple([1.0]*len(self.obj_keys)))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", dict, fitness=creator.FitnessMulti)
        self.toolbox = base.Toolbox()
        self.toolbox.register("clone", copy.deepcopy)

        self.whitelist = _list_all_triples(self.base_mdp)
        self.base_portable = self.base_mdp.to_portable()

        # Precompute shared objects
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

        # Output paths
        self.metrics_dir: Optional[Path] = None
        self.jsonl_path: Optional[str] = None
        if self.output_dir:
            self.metrics_dir = Path(self.output_dir) / "ga"
            (self.metrics_dir / "metrics_curves").mkdir(parents=True, exist_ok=True)
            (self.metrics_dir / "metrics_plots").mkdir(parents=True, exist_ok=True)
            (self.metrics_dir / "mdps").mkdir(parents=True, exist_ok=True)
            self.jsonl_path = str(self.metrics_dir / "metrics.jsonl")

        # State for curves/log
        self.uid_counter = 0
        self.all_metric_keys: set = set()
        self.metrics_history: Dict[str, Dict[str, List[float]]] = {}

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
    def eval_population(self, pop_inds: List[Any], gen: int) -> Tuple[
        List[List[float]], List[Dict[str, Optional[float]]]]:
        # Submit evaluate_one_remote with backpressure; pass ObjectRefs to avoid copies
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

            # Backpressure on driver
            if len(pending) >= self.max_in_flight:
                ready, pending = ray.wait(pending, num_returns=1)
                results.append(ray.get(ready[0]))

        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            results.append(ray.get(ready[0]))

        # Unpack + write fitness
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
                obj_vals = [float(md[k]) for k in self.obj_keys]
                jsonl_append(self.jsonl_path, {
                    "gen": int(gen), "uid": int(self.uid_counter + idx), "ind_in_gen": int(idx),
                    "is_child": bool(is_child), "objective_keys": list(self.obj_keys),
                    "objectives": obj_vals, "metrics": md
                })
            self.uid_counter += len(metrics)
        update_metric_curves(metrics, self.metrics_history, self.all_metric_keys)

    def wb_log_survivors(self, gen: int, survivors: List[Any], metrics: List[Dict[str, Optional[float]]]) -> None:
        if self.wandb_writer is None: return
        for ind_in_gen, md in enumerate(metrics):
            payload = {"ga/gen": int(gen), "ga_meta/ind_in_gen": int(ind_in_gen), "ga_meta/is_child": 0}
            for mk, mv in md.items():
                if mv is None: continue
                fv = float(mv)
                if math.isfinite(fv): payload[f"ga_metrics/{mk}"] = fv
            for ok in self.obj_keys:
                v = md.get(ok, None)
                if v is None: continue
                fv = float(v)
                if math.isfinite(fv): payload[f"ga_object/{ok}"] = fv
            self.wandb_writer.log.remote(payload)
        if self.mmk_list:
            log_best_by_keys_for_batch(gen, metrics, list(self.mmk_list), self.wandb_writer)

    # ---- selection helpers ----
    def select_survivors(self, pool: List[Any], size: int) -> List[Any]:
        if self.survivor == "nsga2": return tools.selNSGA2(pool, size)
        if self.survivor == "spea2": return tools.selSPEA2(pool, size)
        if self.survivor == "nsga3": return tools.selNSGA2(pool, size)  # keep simple
        raise ValueError(f"Unknown survivor selector: {self.survivor}")

    def select_parents_pair(self, population: List[Any]) -> Tuple[Any, Any]:
        if self.parents == "tournament_dcd":
            return tools.selTournamentDCD(population, 1)[0], tools.selTournamentDCD(population, 1)[0]
        if self.parents == "tournament":
            k = int(self.parent_k)
            return tools.selTournament(population, 1, tournsize=k)[0], tools.selTournament(population, 1, tournsize=k)[0]
        if self.parents == "roulette":
            return tools.selRoulette(population, 1)[0], tools.selRoulette(population, 1)[0]
        raise ValueError(f"Unknown parent selector: {self.parents}")

    # ---- main run ----
    def run(self) -> Tuple[List[MDPNetwork], List[List[float]], List[MDPNetwork], List[List[float]]]:
        # init population = base + mutated copies
        pop: List[Any] = [creator.Individual({"portable": self.base_portable})]
        for i in range(self.population_size - 1):
            p_child = self.domain_mutate(self.base_portable, seed=derive_seed(self.master_seed, "init", i))
            pop.append(creator.Individual({"portable": p_child}))

        # initial eval (gen=0)
        objs, pop_metrics = self.eval_population(pop, gen=0)
        self.log_batch(0, pop_metrics, is_child=False)
        pop = tools.selNSGA2(pop, len(pop))
        init_stats = summ_stats(objs)
        self.logger.info("[Init] pop=%d | %s", len(pop),
                         " | ".join(f"obj{m}: min={init_stats.get(f'min_{m}', float('nan')):.4f} "
                                    f"mean={init_stats.get(f'mean_{m}', float('nan')):.4f} "
                                    f"max={init_stats.get(f'max_{m}', float('nan')):.4f}"
                                    for m in range(len(objs[0]) if objs else 0)) or "NA")
        if self.wandb_writer is not None:
            payload = {"ga/init/pop_size": int(len(pop)), "ga/gen": 0}
            M = len(objs[0]) if objs else 0
            for m in range(M):
                payload[f"ga/init/obj{m}_min"] = init_stats.get(f"min_{m}", float("nan"))
                payload[f"ga/init/obj{m}_mean"] = init_stats.get(f"mean_{m}", float("nan"))
                payload[f"ga/init/obj{m}_max"] = init_stats.get(f"max_{m}", float("nan"))
            self.wandb_writer.log.remote(payload)

        # evolution
        for gen in range(self.generations):
            gstart = time.perf_counter()
            if self.parents == "tournament_dcd":
                pop = tools.selNSGA2(pop, len(pop))
            children: List[Any] = []
            for k in range(self.lambd):
                pa, pb = self.select_parents_pair(pop)
                seed_child = derive_seed(self.master_seed, "child", gen, k)
                if random.random() < self.cxpb:
                    c_portable = self.domain_crossover(pa["portable"], pb["portable"], seed=derive_seed(seed_child, "cx"))
                else:
                    src = pa if (random.random() < 0.5) else pb
                    c_portable = dict(src["portable"])
                c_portable = self.domain_mutate(c_portable, seed=derive_seed(seed_child, "mut"))
                children.append(creator.Individual({"portable": c_portable}))

            child_objs, child_metrics = self.eval_population(children, gen=gen+1)
            self.log_batch(gen+1, child_metrics, is_child=True)

            union = pop + children
            survivors = self.select_survivors(union, self.population_size)

            # survivors' metrics (re-eval for full metric dicts)
            pending, surv_results = [], []
            for idx, ind in enumerate(survivors):
                mdp_ref = ray.put(ind["portable"])
                ref = evaluate_one_remote.remote(
                    mdp_ref,
                    [dict(name=s.name, params=s.params, resources=s.resources,
                          retries=s.retries, timeout_s=s.timeout_s) for s in self.items],
                    self.obj_keys,
                    self.shared_ref,
                    int(self.master_seed), int(gen + 1), int(idx)
                )
                pending.append(ref)
                if len(pending) >= self.max_in_flight:
                    ready, pending = ray.wait(pending, num_returns=1)
                    surv_results.append(ray.get(ready[0]))

            while pending:
                ready, pending = ray.wait(pending, num_returns=1)
                surv_results.append(ray.get(ready[0]))

            surv_objs = [list(o) for (o, _) in surv_results]
            surv_metrics = [md for (_, md) in surv_results]

            self.wb_log_survivors(gen+1, survivors, surv_metrics)
            update_metric_curves(surv_metrics, self.metrics_history, self.all_metric_keys)

            gen_stats = summ_stats(surv_objs)
            self.logger.info("[Gen %d/%d] pop=%d | %s", gen+1, self.generations, len(survivors),
                             " | ".join(
                                f"obj{m}: min={gen_stats.get(f'min_{m}', float('nan')):.4f} "
                                f"mean={gen_stats.get(f'mean_{m}', float('nan')):.4f} "
                                f"max={gen_stats.get(f'max_{m}', float('nan')):.4f}"
                                for m in range(len(surv_objs[0]) if surv_objs else 0)
                             ) or "NA")
            if self.wandb_writer is not None:
                payload = {"ga/pop/size": int(len(survivors)),
                           "ga/time/total_gen_sec": float(time.perf_counter() - gstart),
                           "ga/gen": int(gen+1)}
                M = len(surv_objs[0]) if surv_objs else 0
                for m in range(M):
                    payload[f"ga/pop/obj{m}_min"] = gen_stats.get(f"min_{m}", float("nan"))
                    payload[f"ga/pop/obj{m}_mean"] = gen_stats.get(f"mean_{m}", float("nan"))
                    payload[f"ga/pop/obj{m}_max"] = gen_stats.get(f"max_{m}", float("nan"))
                self.wandb_writer.log.remote(payload)

            pop = survivors

        # final Pareto (F1)
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
                self.logger.info("[GA] Saved PF[%d] -> %s", i, path.name)

        if self.wandb_writer is not None:
            payload = {"ga/final/F1_size": int(len(F1)), "ga/gen": int(self.generations)}
            fstats = summ_stats([list(ind.fitness.values) for ind in F1] if F1 else [list(ind.fitness.values) for ind in pop])
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


# =========================
# Public wrapper
# =========================

def run_ga_deap_ray(*, base_mdp: MDPNetwork, population_size: int, generations: int, master_seed: int,
                    algo_type: str = "mu_plus_lambda", mu: Optional[int] = None, lambd: Optional[int] = None,
                    cxpb: float = 1.0, survivor: str = "nsga2", parents: str = "tournament_dcd", parent_k: int = 2,
                    score_spec: List[Dict[str, Any]], objective_keys: List[str],
                    max_metric_keys: Optional[Sequence[str]] = None, max_in_flight: int = 128,
                    ray_init: Optional[Dict[str, Any]] = None, ops: Optional[Dict[str, Any]] = None,
                    distance: Optional[Dict[str, Any]] = None, solver: Optional[Dict[str, Any]] = None,
                    output_dir: Optional[str] = None, wandb_writer=None
                    ) -> Tuple[List[MDPNetwork], List[List[float]], List[MDPNetwork], List[List[float]]]:
    t = Trainer(base_mdp, population_size, generations, master_seed, algo_type, mu, lambd, cxpb,
                survivor, parents, parent_k, score_spec, objective_keys, max_metric_keys,
                max_in_flight, ray_init, ops, distance, solver, output_dir, wandb_writer)
    return t.run()
