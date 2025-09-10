# genetic_algorithms/ga_mdp_search.py
# Ray-based NSGA-II over MDPNetwork
# - No GAConfig dataclass; top-level GA params + grouped dicts.
# - Stable seeding (deterministic w.r.t. master seed & tags), independent of concurrency.
# - One GAWorker actor type (mutate + score); driver orchestrates selection/offspring/eval.
# - Score interface: fn(mdp, shared, **params) -> Sequence[float]
# - W&B and saving are controlled by the caller.

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import ray
from matplotlib import pyplot as plt
from ray.actor import ActorHandle

from experiment_utils.utils import ensure_dir
from genetic_algorithms.mdp_ops import (
    EdgeTriple,
    _list_all_triples,
    _prune_low_prob_transitions,
    _mutation_add_edge,
    _mutation_prob_pairwise,
    _mutation_reward_smallstep,
    _crossover_action_block,
)
from genetic_algorithms.score_fns import _normalize_score_spec, SCORE_FNS
from mdp_network import MDPNetwork
from mdp_network.mdp_tables import (
    q_table_to_policy,
)
from mdp_network.solvers import optimal_value_iteration, compute_occupancy_measure


def _derive_seed(master_seed: int, *tags: Any) -> int:
    """Deterministic 64-bit seed derived from (master_seed, tags*)."""
    h = hashlib.sha256()
    h.update(str(int(master_seed)).encode("utf-8"))
    for t in tags:
        h.update(b"::")
        h.update(str(t).encode("utf-8"))
    # take first 8 bytes -> [0, 2**64-1]
    return int.from_bytes(h.digest()[:8], "little", signed=False)


# =========================================================
# NSGA-II helpers
# =========================================================


def _dominates_max(a: Sequence[float], b: Sequence[float]) -> bool:
    ge, gt = True, False
    for ai, bi in zip(a, b):
        if ai < bi:
            ge = False
            break
        if ai > bi:
            gt = True
    return ge and gt


def _fast_non_dominated_sort(objs: List[List[float]]) -> List[List[int]]:
    N = len(objs)
    S = [set() for _ in range(N)]
    n = [0] * N
    fronts: List[List[int]] = [[]]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if _dominates_max(objs[p], objs[q]):
                S[p].add(q)
            elif _dominates_max(objs[q], objs[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    Q.append(q)
        i += 1
        fronts.append(Q)
    fronts.pop()
    return fronts


def _compute_crowding_distance(
    objs: List[List[float]], idxs: List[int]
) -> Dict[int, float]:
    M = len(objs[0]) if objs else 0
    Nf = len(idxs)
    if Nf == 0:
        return {}
    distance = {i: 0.0 for i in idxs}
    if Nf <= 2:
        for i in idxs:
            distance[i] = float("inf")
        return distance
    for m in range(M):
        vals = [objs[i][m] for i in idxs]
        order = [x for _, x in sorted(zip(vals, idxs))]
        vmin, vmax = vals[np.argmin(vals)], vals[np.argmax(vals)]
        if vmax == vmin:
            continue
        distance[order[0]] = float("inf")
        distance[order[-1]] = float("inf")
        for k in range(1, Nf - 1):
            i_prev, i_next = order[k - 1], order[k + 1]
            i_mid = order[k]
            gap = (objs[i_next][m] - objs[i_prev][m]) / (vmax - vmin)
            distance[i_mid] += gap
    return distance


# =========================================================
# Ray worker
# =========================================================


@ray.remote
class GAWorker:
    def __init__(
        self,
        base_portable: Dict[str, Any],
        whitelist: List[Tuple[int, int, int]],
        ops: Dict[str, Any],
        distance_cfg: Dict[str, Any],
        solver: Dict[str, Any],
        precomputed: Optional[Dict[str, Any]] = None,
    ):
        self.base_ref = MDPNetwork.from_portable(base_portable)
        self.whitelist: Set[EdgeTriple] = set(tuple(x) for x in whitelist)
        self.ops = dict(ops or {})
        self.distance = {
            "max_hops": distance_cfg.get("dist_max_hops", distance_cfg.get("max_hops", None)),
            "node_cap": distance_cfg.get("dist_node_cap", distance_cfg.get("node_cap", None)),
            "weight_eps": float(distance_cfg.get("dist_weight_eps", distance_cfg.get("weight_eps", 1e-9))),
            "unreachable": float(distance_cfg.get("dist_unreachable", distance_cfg.get("unreachable", 1e6))),
        }
        self.solver = dict(solver or {})
        self.precomputed = dict(precomputed or {})

    def mutate(
        self,
        seed: int,
        pa_portable: Optional[Dict[str, Any]] = None,
        pb_portable: Optional[Dict[str, Any]] = None,
        do_crossover: bool = False,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(int(seed))
        prob_floor = float(self.ops.get("prob_floor", 1e-6))

        if pa_portable is None:
            ind = MDPNetwork.from_portable(self.base_ref.to_portable())
        else:
            pa = MDPNetwork.from_portable(pa_portable)
            if pb_portable is None:
                ind = pa.clone()
            else:
                pb = MDPNetwork.from_portable(pb_portable)
                ind = (
                    _crossover_action_block(
                        pa,
                        pb,
                        rng,
                        whitelist=self.whitelist,
                        prob_floor=prob_floor,
                    )
                    if do_crossover
                    else (pa if rng.random() < 0.5 else pb).clone()
                )

        for _ in range(int(self.ops.get("add_edge_attempts_per_child", 2))):
            _mutation_add_edge(
                ind,
                rng,
                self.base_ref,
                self.ops,
                self.distance,
                whitelist=self.whitelist,
            )
        _mutation_prob_pairwise(ind, rng, self.ops, whitelist=self.whitelist)
        if int(self.ops.get("reward_tweak_edges_per_child", 50)) > 0:
            _mutation_reward_smallstep(ind, rng, self.ops)

        if self.ops.get("prune_prob_threshold", None) is not None:
            _prune_low_prob_transitions(
                ind,
                float(self.ops["prune_prob_threshold"]),
                whitelist=self.whitelist,
                prob_floor=prob_floor,
            )
        return ind.to_portable()

    def score_batch(
        self,
        portables: List[Dict[str, Any]],
        score_spec: List[Tuple[str, Dict[str, Any]]],
    ) -> List[Dict[str, Optional[float]]]:
        """
        Evaluate a batch; return one flat metrics dict per portable.
        - Keys are "simple names" returned by score functions.
        - If duplicate keys appear across score functions -> raise ValueError.
        """
        try:
            fns_spec = _normalize_score_spec(score_spec)
            shared = {"solver": self.solver, "precomputed": self.precomputed}
            results: List[Dict[str, Optional[float]]] = []
            for p in portables:
                mdp = MDPNetwork.from_portable(p)
                merged: Dict[str, Optional[float]] = {}
                for name, params in fns_spec:
                    fn = SCORE_FNS[name]
                    m = fn(mdp, shared, **params)
                    if not isinstance(m, dict):
                        raise TypeError(
                            f"Score function '{name}' must return Dict[str, float|None]."
                        )
                    # duplicate key detection
                    dup = set(merged.keys()).intersection(m.keys())
                    if dup:
                        raise ValueError(
                            f"Duplicate metric keys detected across score functions: {sorted(dup)}"
                        )
                    # type normalization
                    for k, v in m.items():
                        if v is None:
                            merged[k] = None
                        else:
                            fv = float(v)
                            merged[k] = fv
                results.append(merged)
            return results
        except BaseException as e:
            import traceback

            tb = traceback.format_exc()
            print(f"[GAWorker.score_batch] FATAL: {e}\n{tb}")
            raise RuntimeError(f"GAWorker.score_batch crashed with: {e}") from e


# =========================================================
# Public driver
# =========================================================


def _summ_stats(objs: List[List[float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not objs:
        return out
    arr = np.asarray(objs, dtype=float)
    M = arr.shape[1]
    for m in range(M):
        out[f"min_{m}"] = float(np.min(arr[:, m]))
        out[f"mean_{m}"] = float(np.mean(arr[:, m]))
        out[f"max_{m}"] = float(np.max(arr[:, m]))
    return out


def _safe_name(s: str) -> str:
    """Make metric key safe for filenames."""
    t = s.replace("/", "__").replace("\\", "__").replace(" ", "_")
    return "".join(c for c in t if (c.isalnum() or c in "._-+%=:@#[]{}()"))


def _jsonl_append(path: str, rec: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _finite_vals(vals: List[Optional[float]]) -> List[float]:
    out: List[float] = []
    for v in vals:
        if v is None:
            continue
        fv = float(v)
        if math.isfinite(fv):
            out.append(fv)
    return out


def ga_make_worker(
    base_portable: Dict[str, Any],
    whitelist: List[Tuple[int, int, int]],
    ops: Dict[str, Any],
    distance_cfg: Dict[str, Any],
    solver: Dict[str, Any],
    precomputed: Optional[Dict[str, Any]],
):
    """Create a short-lived GAWorker actor."""
    return GAWorker.options(num_cpus=1).remote(
        base_portable=base_portable,
        whitelist=whitelist,
        ops=ops,
        distance_cfg=distance_cfg,
        solver=solver,
        precomputed=precomputed,
    )


def ga_score_portables_with_metrics(
    *,
    gen: int,
    is_child: bool,
    portables: List[Dict[str, Any]],
    score_spec: List[Tuple[str, Dict[str, Any]]],
    objective_keys: List[str],
    base_portable: Dict[str, Any],
    whitelist: List[Tuple[int, int, int]],
    ops: Dict[str, Any],
    distance_cfg: Dict[str, Any],
    solver: Dict[str, Any],
    precomputed: Optional[Dict[str, Any]],
    wandb_writer: Optional[ActorHandle],
    jsonl_path: Optional[str],
    uid_counter: int,
) -> Tuple[List[List[float]], List[Dict[str, Optional[float]]], int]:
    """
    Fan-out evaluate -> (objs_list, metrics_list, new_uid_counter).
    Also logs to W&B and JSONL.
    W&B policy:
      - All finite metrics (non-None) -> ga_metrics/<key>
      - Objective keys                -> ga_object/<key>
      - Meta info (ids/flags/indices) -> ga_meta/*
    """
    if not portables:
        return [], [], uid_counter

    futs = []
    for p in portables:
        worker = ga_make_worker(
            base_portable, whitelist, ops, distance_cfg, solver, precomputed
        )
        futs.append(worker.score_batch.remote([p], score_spec))
    parts = ray.get(futs)  # List[List[Dict[str,float|None]]], each batch size=1
    metrics_list: List[Dict[str, Optional[float]]] = [one[0] for one in parts]

    objs_list: List[List[float]] = []
    for ind_in_gen, md in enumerate(metrics_list):
        # strict objective extraction
        obj_vals: List[float] = []
        for k in objective_keys:
            if k not in md or md[k] is None or not math.isfinite(float(md[k])):
                raise ValueError(
                    f"Objective key '{k}' missing/non-finite at gen={gen}, ind={ind_in_gen}."
                )
            obj_vals.append(float(md[k]))
        objs_list.append(obj_vals)

        # per-evaluation id
        uid = uid_counter
        uid_counter += 1

        # W&B payload (groups renamed; booleans cast to int under ga_meta/*)
        if wandb_writer is not None:
            payload: Dict[str, Any] = {
                "ga/gen": int(gen),
                "ga_meta/uid": int(uid),
                "ga_meta/ind_in_gen": int(ind_in_gen),
                "ga_meta/is_child": int(bool(is_child)),  # avoid media warning
            }
            # full metrics (finite only) -> ga_metrics/*
            for mk, mv in md.items():
                if mv is None:
                    continue
                fv = float(mv)
                if math.isfinite(fv):
                    payload[f"ga_metrics/{mk}"] = fv
            # objectives (subset) -> ga_object/*
            for ok in objective_keys:
                v = md.get(ok, None)
                if v is None:
                    continue
                fv = float(v)
                if math.isfinite(fv):
                    payload[f"ga_object/{ok}"] = fv
            try:
                wandb_writer.log.remote(payload)
            except Exception:
                pass

        # JSONL append
        if jsonl_path:
            _jsonl_append(
                jsonl_path,
                {
                    "gen": int(gen),
                    "uid": int(uid),
                    "ind_in_gen": int(ind_in_gen),
                    "is_child": bool(is_child),
                    "objective_keys": list(objective_keys),
                    "objectives": obj_vals,
                    "metrics": md,
                },
            )

    return objs_list, metrics_list, uid_counter


def ga_update_metric_curves(
    metrics_for_pop: List[Dict[str, Optional[float]]],
    metrics_history: Dict[str, Dict[str, List[float]]],
    all_metric_keys: Set[str],
) -> None:
    """Update per-generation min/mean/max series in-place."""
    for md in metrics_for_pop:
        all_metric_keys.update(md.keys())
    for k in sorted(all_metric_keys):
        vals = [m.get(k, None) for m in metrics_for_pop]
        fvals = _finite_vals(vals)
        if k not in metrics_history:
            metrics_history[k] = {"min": [], "mean": [], "max": []}
        if fvals:
            metrics_history[k]["min"].append(float(np.min(fvals)))
            metrics_history[k]["mean"].append(float(np.mean(fvals)))
            metrics_history[k]["max"].append(float(np.max(fvals)))
        else:
            nan = float("nan")
            metrics_history[k]["min"].append(nan)
            metrics_history[k]["mean"].append(nan)
            metrics_history[k]["max"].append(nan)


def ga_export_metric_curves(
    metrics_dir: Path,
    metrics_history: Dict[str, Dict[str, List[float]]],
    wandb_writer: Optional[ActorHandle] = None,
) -> None:
    """Write CSVs and PNGs (3 subplots per metric: min/mean/max), also upload PNGs to W&B."""
    curves_dir = metrics_dir / "metrics_curves"
    plots_dir = metrics_dir / "metrics_plots"
    ensure_dir(curves_dir)
    ensure_dir(plots_dir)

    # CSV
    for key, series in metrics_history.items():
        csv_path = curves_dir / f"{_safe_name(key)}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("gen,min,mean,max\n")
            for g in range(len(series["min"])):
                f.write(
                    f"{g},{series['min'][g]},{series['mean'][g]},{series['max'][g]}\n"
                )

    # PNG (+ upload to W&B)
    for key, series in metrics_history.items():
        gens = list(range(len(series["min"])))
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
        axes[0].plot(gens, series["min"]);  axes[0].set_title(f"{key} - min");  axes[0].set_xlabel("gen");  axes[0].set_ylabel("min")
        axes[1].plot(gens, series["mean"]); axes[1].set_title(f"{key} - mean"); axes[1].set_xlabel("gen");  axes[1].set_ylabel("mean")
        axes[2].plot(gens, series["max"]);  axes[2].set_title(f"{key} - max");  axes[2].set_xlabel("gen");  axes[2].set_ylabel("max")
        png_path = plots_dir / f"{_safe_name(key)}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

        if wandb_writer is not None:
            # one image per metric under a consistent namespace
            try:
                wandb_writer.log_image.remote(
                    key=f"ga_metric_plots/{key}",
                    path=str(png_path),
                    caption=f"{key} (min/mean/max vs gen)",
                )
            except Exception:
                pass

    # (optional) also log the whole plots directory as an artifact for easy download
    if wandb_writer is not None:
        try:
            wandb_writer.log_artifact_dir.remote(
                name="ga-metric-plots",
                a_type="plots",
                dir_path=str(plots_dir),
                metadata={"kind": "metric_curves_pngs"},
            )
        except Exception:
            pass


def run_ga(
    *,
    base_mdp: MDPNetwork,
    population_size: int,
    generations: int,
    seed: int,
    tournament_k: int = 2,
    elitism: int = 8,
    crossover_rate: float = 1.0,
    output_dir: Optional[str] = None,
    wandb_writer: Optional[ActorHandle] = None,
    ops: Optional[Dict[str, Any]] = None,
    distance: Optional[Dict[str, Any]] = None,
    solver: Optional[Dict[str, Any]] = None,
    score: List[Tuple[str, Dict[str, Any]]] = None,  # strict format
    objective_keys: List[str] = None,  # required: keys used as GA objectives
) -> Tuple[List[MDPNetwork], List[List[float]], List[MDPNetwork], List[List[float]]]:
    """
    GA driver (NSGA-II, maximize objectives).
    - score: list of ('name', {params}); each function returns a flat metrics dict with simple keys.
    - objective_keys: the exact metric keys to optimize; must exist & be finite per individual.
    W&B:
      * Full metrics -> ga_metrics/<key>   (only values that are not None and finite)
      * Objectives   -> ga_object/<key>    (subset of metrics, optimized by GA)
    All bound to step 'ga/gen'.
    - Logging:
        * Append per-evaluation records to <output_dir>/ga/metrics.jsonl.
    - End of run:
        * For each metric key, save CSV(gen,min,mean,max) and a PNG with 3 subplots (min/mean/max).
    """
    if (
        objective_keys is None
        or not isinstance(objective_keys, list)
        or not objective_keys
    ):
        raise ValueError("objective_keys must be a non-empty list of metric keys.")
    score = score or [("obj_multi_perf", {})]

    ops = dict(ops or {})
    distance = dict(distance or {})
    solver = dict(solver or {})

    logger = logging.getLogger("ga")
    if not logger.handlers:
        _h = logging.StreamHandler(sys.stdout)
        _h.setFormatter(
            logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(_h)
    logger.setLevel(logging.INFO)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    if wandb_writer is not None:
        try:
            # bind curves to x-axis "ga/gen"
            wandb_writer.define_metric.remote("ga_metrics/*", step_metric="ga/gen")
            wandb_writer.define_metric.remote("ga_object/*", step_metric="ga/gen")
            # optional meta streams (ids, flags, etc.)
            wandb_writer.define_metric.remote("ga_meta/*", step_metric="ga/gen")
            # keep existing families if you still log them
            wandb_writer.define_metric.remote("ga/pop/*", step_metric="ga/gen")
            wandb_writer.define_metric.remote("ga/time/*", step_metric="ga/gen")
            wandb_writer.define_metric.remote("ga/init/*", step_metric="ga/gen")
            wandb_writer.define_metric.remote("ga/final/*", step_metric="ga/gen")
            # images (media) do not need step binding, but this is harmless
            wandb_writer.define_metric.remote("ga_metric_plots/*", step_metric="ga/gen")
        except Exception:
            pass

    # ===== Precompute baseline stuff (independent of W&B) =====
    gamma = float(solver.get("vi_gamma", 0.99))
    theta = float(solver.get("vi_theta", 1e-6))
    max_iters = int(solver.get("vi_max_iterations", 1000))
    policy_temp = float(solver.get("policy_temperature", 1.0))
    policy_mix = tuple(solver.get("policy_mix", (0.0, 1.0, 0.0)))
    tie_tol = float(solver.get("policy_tie_tol", 1e-6))

    t0 = time.perf_counter()
    _, Q = optimal_value_iteration(
        base_mdp, gamma=gamma, theta=theta, max_iterations=max_iters
    )
    base_policy = q_table_to_policy(
        Q,
        states=list(base_mdp.states),
        num_actions=base_mdp.num_actions,
        mixing=policy_mix,
        temperature=policy_temp,
        tie_tol=tie_tol,
    )
    base_occupancy = compute_occupancy_measure(
        base_mdp, base_policy, gamma=gamma, theta=theta, max_iterations=max_iters
    )
    precomputed = {
        "base_policy": base_policy.to_portable(),
        "base_occupancy": base_occupancy.to_portable(),
        "base_mdp": base_mdp.to_portable(),
    }
    t1 = time.perf_counter()

    if wandb_writer is not None:
        try:
            wandb_writer.log.remote(
                {"ga/time/precompute_sec": float(t1 - t0), "ga/gen": -1}
            )
        except Exception:
            pass

    whitelist = _list_all_triples(base_mdp)
    base_portable = base_mdp.to_portable()
    rng_drv = np.random.default_rng(_derive_seed(seed, "driver"))

    # ===== Output dirs & JSONL path =====
    metrics_dir = None
    if output_dir:
        metrics_dir = Path(output_dir) / "ga"
        ensure_dir(metrics_dir)
        ensure_dir(metrics_dir / "metrics_curves")
        ensure_dir(metrics_dir / "metrics_plots")
        ensure_dir(metrics_dir / "mdps")
        jsonl_path = str(metrics_dir / "metrics.jsonl")
    else:
        jsonl_path = None

    # ===== Init population =====
    pop: List[MDPNetwork] = [base_mdp.clone()]
    need = population_size - 1
    if need > 0:
        futs = []
        for i in range(need):
            worker = ga_make_worker(
                base_portable, whitelist, ops, distance, solver, precomputed
            )
            futs.append(worker.mutate.remote(seed=_derive_seed(seed, "init", i)))
        children_portables = ray.get(futs)
        pop.extend([MDPNetwork.from_portable(p) for p in children_portables])

    # ===== Initial evaluation (gen=0) =====
    init_portables = [m.to_portable() for m in pop]
    uid_counter = 0
    objs, pop_metrics, uid_counter = ga_score_portables_with_metrics(
        gen=0,
        is_child=False,
        portables=init_portables,
        score_spec=score,
        objective_keys=objective_keys,
        base_portable=base_portable,
        whitelist=whitelist,
        ops=ops,
        distance_cfg=distance,
        solver=solver,
        precomputed=precomputed,
        wandb_writer=wandb_writer,
        jsonl_path=jsonl_path,
        uid_counter=uid_counter,
    )

    # ===== Metric curves state =====
    all_metric_keys: Set[str] = (
        set().union(*[set(md.keys()) for md in pop_metrics]) if pop_metrics else set()
    )
    metrics_history: Dict[str, Dict[str, List[float]]] = {}
    ga_update_metric_curves(pop_metrics, metrics_history, all_metric_keys)

    # ===== Logs for gen=0 =====
    init_stats = _summ_stats(objs)
    logger.info(
        "[Init] pop=%d | %s",
        len(pop),
        " | ".join(
            f"obj{m}: min={init_stats.get(f'min_{m}', float('nan')):.4f} "
            f"mean={init_stats.get(f'mean_{m}', float('nan')):.4f} "
            f"max={init_stats.get(f'max_{m}', float('nan')):.4f}"
            for m in range(len(objs[0]) if objs else 0)
        )
        or "NA",
    )
    if wandb_writer is not None:
        payload = {"ga/init/pop_size": int(len(pop)), "ga/gen": 0}
        M = len(objs[0]) if objs else 0
        for m in range(M):
            payload[f"ga/init/obj{m}_min"] = init_stats.get(f"min_{m}", float("nan"))
            payload[f"ga/init/obj{m}_mean"] = init_stats.get(f"mean_{m}", float("nan"))
            payload[f"ga/init/obj{m}_max"] = init_stats.get(f"max_{m}", float("nan"))
        try:
            wandb_writer.log.remote(payload)
        except Exception:
            pass

    # ===== Ranks & crowding =====
    fronts = _fast_non_dominated_sort(objs)
    ranks = [0] * len(pop)
    for r, F in enumerate(fronts):
        for i in F:
            ranks[i] = r
    crowding: Dict[int, float] = {}
    for F in fronts:
        crowding.update(_compute_crowding_distance(objs, F))

    # ===== Evolutions =====
    for gen in range(generations):
        gstart = time.perf_counter()
        elite_k = max(0, min(int(elitism), population_size, len(pop)))
        elite_parent_idxs = set()
        if elite_k > 0:
            order_prev = sorted(
                range(len(pop)), key=lambda i: (ranks[i], -crowding.get(i, 0.0))
            )
            elite_parent_idxs = set(order_prev[:elite_k])

        # tournament selection
        parents_pairs: List[Tuple[MDPNetwork, MDPNetwork]] = []
        for k in range(population_size):
            idxs = rng_drv.choice(len(pop), size=int(tournament_k), replace=False)
            best = int(idxs[0])
            for j in idxs[1:]:
                j = int(j)
                if ranks[j] < ranks[best] or (
                    ranks[j] == ranks[best]
                    and crowding.get(j, 0.0) > crowding.get(best, 0.0)
                ):
                    best = j
            idxs2 = rng_drv.choice(len(pop), size=int(tournament_k), replace=False)
            best2 = int(idxs2[0])
            for j in idxs2[1:]:
                j = int(j)
                if ranks[j] < ranks[best2] or (
                    ranks[j] == ranks[best2]
                    and crowding.get(j, 0.0) > crowding.get(best2, 0.0)
                ):
                    best2 = j
            parents_pairs.append((pop[best], pop[best2]))

        # offspring
        futs = []
        for k, (pa, pb) in enumerate(parents_pairs):
            do_x = rng_drv.random() < float(crossover_rate)
            worker = ga_make_worker(
                base_portable, whitelist, ops, distance, solver, precomputed
            )
            futs.append(
                worker.mutate.remote(
                    seed=_derive_seed(seed, "child", gen, k),
                    pa_portable=pa.to_portable(),
                    pb_portable=pb.to_portable(),
                    do_crossover=bool(do_x),
                )
            )
        child_portables = ray.get(futs)
        children = [MDPNetwork.from_portable(p) for p in child_portables]

        # evaluate children at gen+1
        child_objs, child_metrics, uid_counter = ga_score_portables_with_metrics(
            gen=gen + 1,
            is_child=True,
            portables=[c.to_portable() for c in children],
            score_spec=score,
            objective_keys=objective_keys,
            base_portable=base_portable,
            whitelist=whitelist,
            ops=ops,
            distance_cfg=distance,
            solver=solver,
            precomputed=precomputed,
            wandb_writer=wandb_writer,
            jsonl_path=jsonl_path,
            uid_counter=uid_counter,
        )

        # environmental selection with locked elites
        union_pop = pop + children
        union_objs = objs + child_objs
        union_metrics = pop_metrics + child_metrics

        union_fronts = _fast_non_dominated_sort(union_objs)
        locked = set(int(i) for i in elite_parent_idxs)

        new_pop: List[MDPNetwork] = [union_pop[i] for i in locked]
        new_objs: List[List[float]] = [union_objs[i] for i in locked]
        new_metrics: List[Dict[str, Optional[float]]] = [
            union_metrics[i] for i in locked
        ]

        for F in union_fronts:
            F_remaining = [i for i in F if i not in locked]
            if len(new_pop) + len(F_remaining) <= population_size:
                new_pop.extend([union_pop[i] for i in F_remaining])
                new_objs.extend([union_objs[i] for i in F_remaining])
                new_metrics.extend([union_metrics[i] for i in F_remaining])
            else:
                dist = _compute_crowding_distance(union_objs, F_remaining)
                sorted_F = sorted(
                    F_remaining, key=lambda i: dist.get(i, 0.0), reverse=True
                )
                remain = population_size - len(new_pop)
                chosen = sorted_F[:remain]
                new_pop.extend([union_pop[i] for i in chosen])
                new_objs.extend([union_objs[i] for i in chosen])
                new_metrics.extend([union_metrics[i] for i in chosen])
                break

        pop, objs, pop_metrics = new_pop, new_objs, new_metrics

        # refresh ranks & crowding
        fronts = _fast_non_dominated_sort(objs)
        ranks = [0] * len(pop)
        for r, F in enumerate(fronts):
            for i in F:
                ranks[i] = r
        crowding = {}
        for F in fronts:
            crowding.update(_compute_crowding_distance(objs, F))

        # update per-generation metric curves
        ga_update_metric_curves(pop_metrics, metrics_history, all_metric_keys)

        # generation summary logs
        gen_stats = _summ_stats(objs)
        logger.info(
            "[Gen %d/%d] pop=%d | %s | F1=%d",
            gen + 1,
            generations,
            len(pop),
            " | ".join(
                f"obj{m}: min={gen_stats.get(f'min_{m}', float('nan')):.4f} "
                f"mean={gen_stats.get(f'mean_{m}', float('nan')):.4f} "
                f"max={gen_stats.get(f'max_{m}', float('nan')):.4f}"
                for m in range(len(objs[0]) if objs else 0)
            )
            or "NA",
            len(fronts[0]) if fronts else 0,
        )
        if wandb_writer is not None:
            payload = {
                "ga/pop/size": int(len(pop)),
                "ga/pop/F1_size": int(len(fronts[0]) if fronts else 0),
                "ga/time/total_gen_sec": float(time.perf_counter() - gstart),
                "ga/gen": int(gen + 1),
            }
            M = len(objs[0]) if objs else 0
            for m in range(M):
                payload[f"ga/pop/obj{m}_min"] = gen_stats.get(f"min_{m}", float("nan"))
                payload[f"ga/pop/obj{m}_mean"] = gen_stats.get(
                    f"mean_{m}", float("nan")
                )
                payload[f"ga/pop/obj{m}_max"] = gen_stats.get(f"max_{m}", float("nan"))
            try:
                wandb_writer.log.remote(payload)
            except Exception:
                pass

    # ===== Final Pareto =====
    final_fronts = _fast_non_dominated_sort(objs)
    F1 = final_fronts[0] if final_fronts else list(range(len(pop)))
    pareto_mdps = [pop[i].clone() for i in F1]
    pareto_objs = [objs[i][:] for i in F1]

    # save Pareto MDPs
    if output_dir:
        mdp_out_dir = Path(output_dir) / "ga" / "mdps"
        ensure_dir(mdp_out_dir)
        for i, (m, objv) in enumerate(zip(pareto_mdps, pareto_objs)):
            tag = "_".join(f"{v:.4f}" for v in objv)
            p = mdp_out_dir / f"pareto_{i}_objs_{tag}.json"
            m.export_to_json(str(p))
            logger.info("[GA] Saved PF[%d] -> %s", i, p.name)

    # final W&B summary
    if wandb_writer is not None:
        payload = {"ga/final/F1_size": int(len(F1)), "ga/gen": int(generations)}
        M = len(objs[0]) if objs else 0
        fstats = _summ_stats([objs[i] for i in F1] if F1 else objs)
        for m in range(M):
            payload[f"ga/final/obj{m}_min"] = fstats.get(f"min_{m}", float("nan"))
            payload[f"ga/final/obj{m}_mean"] = fstats.get(f"mean_{m}", float("nan"))
            payload[f"ga/final/obj{m}_max"] = fstats.get(f"max_{m}", float("nan"))
        try:
            wandb_writer.log.remote(payload)
        except Exception:
            pass

    # ===== Export per-metric curves: CSV + PNG =====
    if metrics_dir is not None:
        ga_export_metric_curves(metrics_dir, metrics_history, wandb_writer=wandb_writer)

    return pareto_mdps, pareto_objs, pop, objs
