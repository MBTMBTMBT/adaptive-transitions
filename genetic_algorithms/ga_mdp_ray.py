# genetic_algorithms/ga_mdp_search.py
# Ray-based NSGA-II over MDPNetwork
# - No GAConfig dataclass; top-level GA params + grouped dicts.
# - Stable seeding (deterministic w.r.t. master seed & tags), independent of concurrency.
# - One GAWorker actor type (mutate + score); driver orchestrates selection/offspring/eval.
# - Score interface: fn(mdp, shared, **params) -> Sequence[float]
# - W&B and saving are controlled by the caller.

from __future__ import annotations

import hashlib
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import ray
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
    h.update(str(int(master_seed)).encode())
    for t in tags:
        h.update(b"::")
        h.update(str(t).encode())
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
        precomputed_portables: Optional[List[Dict[str, Any]]] = None,
    ):
        self.base_ref = MDPNetwork.from_portable(base_portable)
        self.whitelist: Set[EdgeTriple] = set(tuple(x) for x in whitelist)
        self.ops = dict(ops or {})
        self.distance = {
            "max_hops": distance_cfg.get(
                "dist_max_hops", distance_cfg.get("max_hops", None)
            ),
            "node_cap": distance_cfg.get(
                "dist_node_cap", distance_cfg.get("node_cap", None)
            ),
            "weight_eps": float(
                distance_cfg.get(
                    "dist_weight_eps", distance_cfg.get("weight_eps", 1e-9)
                )
            ),
            "unreachable": float(
                distance_cfg.get(
                    "dist_unreachable", distance_cfg.get("unreachable", 1e6)
                )
            ),
        }
        self.solver = dict(solver or {})
        self.precomputed_portables = precomputed_portables

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

        # Apply mutations (all protected by whitelist)
        for _ in range(int(self.ops.get("add_edge_attempts_per_child", 2))):
            _mutation_add_edge(
                ind,
                rng,
                self.base_ref,
                self.ops,
                self.distance,
                whitelist=self.whitelist,
            )
        _mutation_prob_pairwise(
            ind,
            rng,
            self.ops,
            whitelist=self.whitelist,
        )
        if int(self.ops.get("reward_tweak_edges_per_child", 50)) > 0:
            _mutation_reward_smallstep(
                ind, rng, self.ops
            )  # reward tweak doesn't remove edges

        if self.ops.get("prune_prob_threshold", None) is not None:
            _prune_low_prob_transitions(
                ind,
                float(self.ops["prune_prob_threshold"]),
                whitelist=self.whitelist,
                prob_floor=prob_floor,
            )

        return ind.to_portable()

    def score_batch(
        self, portables: List[Dict[str, Any]], score_spec: Any
    ) -> List[List[float]]:
        """
        Evaluate a batch of MDPs.
        Concatenate outputs in the given order.
        """
        try:
            fns_spec = _normalize_score_spec(score_spec)
            shared = {"solver": self.solver, "precomputed": self.precomputed_portables}
            results: List[List[float]] = []
            for p in portables:
                mdp = MDPNetwork.from_portable(p)
                obj: List[float] = []
                for name, params in fns_spec:
                    fn = SCORE_FNS[name]
                    vals = fn(mdp, shared, **params)
                    obj.extend([float(x) for x in vals])
                results.append(obj)
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


def run_ga(
    *,
    base_mdp: MDPNetwork,
    # ---- core GA params (expanded) ----
    population_size: int,
    generations: int,
    seed: int,
    tournament_k: int = 2,
    elitism: int = 8,
    crossover_rate: float = 1.0,
    # ---- saving/logging (expanded) ----
    output_dir: Optional[str] = None,
    wandb_writer: Optional[ActorHandle] = None,
    # ---- grouped dicts ----
    ops: Optional[Dict[str, Any]] = None,
    distance: Optional[Dict[str, Any]] = None,
    solver: Optional[Dict[str, Any]] = None,
    score: Any = None,  # simplified shapes accepted
) -> Tuple[List[MDPNetwork], List[List[float]], List[MDPNetwork], List[List[float]]]:
    """
    Genetic Algorithm driver.
    W&B policy:
      - Bind all "ga/*" series to x-axis "ga/gen".
      - Do NOT pass an explicit `step`; include "ga/gen" in the payload.

    `score` accepts:
      - None -> defaults to "obj_multi_perf"
      - "name"
      - ("name", {params})
      - ["name", ("name", {params}), ...]
    """
    ops = dict(ops or {})
    distance = dict(distance or {})
    solver = dict(solver or {})
    score = score or "obj_multi_perf"

    # logger setup
    logger = logging.getLogger("ga")
    if not logger.handlers:
        _h = logging.StreamHandler(sys.stdout)
        _h.setFormatter(
            logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(_h)
    logger.setLevel(logging.INFO)

    # ray init
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # --- W&B axis binding for GA (MOST-SPECIFIC RULE) ---
    if wandb_writer is not None:
        try:
            wandb_writer.define_metric.remote("ga/*", step_metric="ga/gen")
        except Exception:
            pass

        # ===== Precompute baseline policy/occupancy (used by score fns) =====
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
        precomputed = [
            base_policy.to_portable(),
            base_occupancy.to_portable(),
            base_mdp.to_portable(),
        ]
        t1 = time.perf_counter()

    if wandb_writer is not None:
        try:
            wandb_writer.log.remote(
                {"ga/time/precompute_sec": float(t1 - t0), "ga/gen": -1}
            )
        except Exception:
            pass

    # ===== Common materials to construct ephemeral workers =====
    whitelist = _list_all_triples(base_mdp)
    base_portable = base_mdp.to_portable()

    def _spawn_worker():
        """Create a fresh GAWorker actor (num_cpus=1) for a single task."""
        return GAWorker.options(num_cpus=1).remote(
            base_portable=base_portable,
            whitelist=whitelist,
            ops=ops,
            distance_cfg=distance,
            solver=solver,
            precomputed_portables=precomputed,
        )

    rng_drv = np.random.default_rng(_derive_seed(seed, "driver"))

    # ===== Init population (fan-out) =====
    pop: List[MDPNetwork] = [base_mdp.clone()]
    need = population_size - 1
    if need > 0:
        futs = []
        for i in range(need):
            # Each child is produced by its own short-lived worker
            futs.append(
                _spawn_worker().mutate.remote(seed=_derive_seed(seed, "init", i))
            )
        children_portables = ray.get(futs)
        pop.extend([MDPNetwork.from_portable(p) for p in children_portables])

    # ===== Evaluate population (fan-out scoring; preserve order) =====
    def _score_portables(portables: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Fan-out: spawn one worker per portable (batch size 1).
        Ray ensures global concurrency against cluster CPU capacity.
        The result order matches the input order (ray.get preserves order).
        """
        if not portables:
            return []
        futs = []
        for p in portables:
            futs.append(_spawn_worker().score_batch.remote([p], score))
        parts = ray.get(futs)  # List[List[List[float]]] (each is a 1-element batch)
        out: List[List[float]] = []
        for one in parts:
            out.append([float(x) for x in one[0]])
        return out

    objs: List[List[float]] = _score_portables([m.to_portable() for m in pop])

    # ===== Initial logs =====
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
        if elite_k > 0:
            order_prev = sorted(
                range(len(pop)), key=lambda i: (ranks[i], -crowding.get(i, 0.0))
            )
            elite_parent_idxs = set(order_prev[:elite_k])
        else:
            elite_parent_idxs = set()

        # --- tournament selection ---
        parents_pairs: List[Tuple[MDPNetwork, MDPNetwork]] = []
        for k in range(population_size):
            # first parent
            idxs = rng_drv.choice(len(pop), size=int(tournament_k), replace=False)
            best = int(idxs[0])
            for j in idxs[1:]:
                j = int(j)
                if ranks[j] < ranks[best] or (
                    ranks[j] == ranks[best]
                    and crowding.get(j, 0.0) > crowding.get(best, 0.0)
                ):
                    best = j
            # second parent
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

        # --- offspring (fan-out mutation/crossover) ---
        futs = []
        for k, (pa, pb) in enumerate(parents_pairs):
            do_x = rng_drv.random() < float(crossover_rate)
            futs.append(
                _spawn_worker().mutate.remote(
                    seed=_derive_seed(seed, "child", gen, k),
                    pa_portable=pa.to_portable(),
                    pb_portable=pb.to_portable(),
                    do_crossover=bool(do_x),
                )
            )
        child_portables = ray.get(futs)
        children = [MDPNetwork.from_portable(p) for p in child_portables]

        # --- evaluate children (fan-out scoring) ---
        child_objs = _score_portables([c.to_portable() for c in children])

        # --- environmental selection with locked elites ---
        union_pop = pop + children
        union_objs = objs + child_objs
        union_fronts = _fast_non_dominated_sort(union_objs)

        locked = set(int(i) for i in elite_parent_idxs)
        new_pop: List[MDPNetwork] = [union_pop[i] for i in locked]
        new_objs: List[List[float]] = [union_objs[i] for i in locked]

        for F in union_fronts:
            F_remaining = [i for i in F if i not in locked]
            if len(new_pop) + len(F_remaining) <= population_size:
                new_pop.extend([union_pop[i] for i in F_remaining])
                new_objs.extend([union_objs[i] for i in F_remaining])
            else:
                dist = _compute_crowding_distance(union_objs, F_remaining)
                sorted_F = sorted(
                    F_remaining, key=lambda i: dist.get(i, 0.0), reverse=True
                )
                remain = population_size - len(new_pop)
                chosen = sorted_F[:remain]
                new_pop.extend([union_pop[i] for i in chosen])
                new_objs.extend([union_objs[i] for i in chosen])
                break

        pop, objs = new_pop, new_objs

        # --- refresh ranks & crowding ---
        fronts = _fast_non_dominated_sort(objs)
        ranks = [0] * len(pop)
        for r, F in enumerate(fronts):
            for i in F:
                ranks[i] = r
        crowding = {}
        for F in fronts:
            crowding.update(_compute_crowding_distance(objs, F))

        # --- logging (generation k -> "ga/gen" = k+1) ---
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

    # optional save
    if output_dir:
        mdp_out_dir = Path(output_dir) / "ga" / "mdps"
        ensure_dir(mdp_out_dir)
        for i, (m, objv) in enumerate(zip(pareto_mdps, pareto_objs)):
            tag = "_".join(f"{v:.4f}" for v in objv)
            p = mdp_out_dir / f"pareto_{i}_objs_{tag}.json"
            m.export_to_json(str(p))
            logger.info("[GA] Saved PF[%d] -> %s", i, p.name)

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

    return pareto_mdps, pareto_objs, pop, objs
