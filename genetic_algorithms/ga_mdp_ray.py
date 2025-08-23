# genetic_algorithms/ga_mdp_search.py
# Ray-based NSGA-II over MDPNetwork
# - No GAConfig dataclass; top-level GA params + grouped dicts.
# - Stable seeding (deterministic w.r.t. master seed & tags), independent of concurrency.
# - One GAWorker actor type (mutate + score); driver orchestrates selection/offspring/eval.
# - Score interface unified: fn(mdp, shared, **params) -> Sequence[float]
# - Optional saving and W&B are controlled externally by the caller.

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import networkx as nx
import ray

import wandb
from ray.actor import ActorHandle

from mdp_network import MDPNetwork
from mdp_network.mdp_tables import (
    q_table_to_policy,
    PolicyTable,
    ValueTable,
    create_random_policy,
    blend_policies,
)
from mdp_network.metrics import kl_policies, performance_curve_and_integral
from mdp_network.solvers import optimal_value_iteration, compute_occupancy_measure


def _derive_seed(master_seed: int, *tags: Any) -> int:
    """
    Deterministic 64-bit seed derived from (master_seed, tags*).
    Independent of task ordering/concurrency.
    """
    h = hashlib.sha256()
    h.update(str(int(master_seed)).encode())
    for t in tags:
        h.update(b"::")
        h.update(str(t).encode())
    return int.from_bytes(h.digest()[:8], "little", signed=False)


ScoreFn = Callable[[MDPNetwork, Dict[str, Any]], Sequence[float]]  # but we add **params via wrapper

_SCORE_REGISTRY: Dict[str, Callable[[MDPNetwork, Dict[str, Any], Dict[str, Any]], Sequence[float]]] = {}


def register_score_fn(name: str, fn: Callable[[MDPNetwork, Dict[str, Any], Any], Sequence[float]]) -> None:
    """
    Register a score function with unified signature:
        fn(mdp, shared, **params) -> Sequence[float]
    - shared: {"precomputed": [portable...], "solver": {...}, ...}
    - params: per-function params from score spec
    """
    _S = str(name)
    if _S in _SCORE_REGISTRY:
        raise ValueError(f"Score function '{_S}' already registered.")
    # Wrap to ensure **params dict passing
    def _wrapped(mdp: MDPNetwork, shared: Dict[str, Any], params: Dict[str, Any]) -> Sequence[float]:
        return fn(mdp, shared, **(params or {}))
    _SCORE_REGISTRY[_S] = _wrapped


def get_score_fn(name: str):
    if name not in _SCORE_REGISTRY:
        raise KeyError(f"Score function '{name}' is not registered.")
    return _SCORE_REGISTRY[name]


# =========================================================
# Graph helpers & mutation ops (dict-based config)
# =========================================================

State = int
Action = int
EdgeTriple = Tuple[State, Action, State]


def _directed_prob_distance(
    mdp: MDPNetwork,
    s: int,
    sp: int,
    *,
    max_hops: Optional[int],
    node_cap: Optional[int],
    weight_eps: float,
    unreachable: float,
) -> float:
    """Dijkstra on directed graph with edge weight w(u->v)=max(weight_eps, 2 - max_a P(v|u,a))."""
    if s == sp:
        return 0.0
    G = mdp.graph

    # Reachable scope
    if max_hops is not None:
        hop_dist = nx.single_source_shortest_path_length(G, source=s, cutoff=max_hops)
        allowed = set(hop_dist.keys())
    else:
        allowed = {s}
        q = [s]
        while q and (node_cap is None or len(allowed) < node_cap):
            u = q.pop(0)
            for v in G.successors(u):
                if v not in allowed:
                    allowed.add(v)
                    q.append(v)

    if node_cap is not None and len(allowed) > node_cap:
        if "hop_dist" in locals():
            kept = sorted(allowed, key=lambda x: hop_dist.get(x, 10**9))[:node_cap]
            allowed = set(kept)
        else:
            allowed = set(list(allowed)[:node_cap])

    if sp not in allowed:
        return float(unreachable)

    import heapq
    INF = float("inf")
    dist: Dict[int, float] = {s: 0.0}
    heap: List[Tuple[float, int]] = [(0.0, s)]
    while heap:
        du, u = heapq.heappop(heap)
        if du > dist.get(u, INF):
            continue
        if u == sp:
            return float(du)
        for v in G.successors(u):
            if v not in allowed:
                continue
            edata = G[u][v]
            if "transitions" not in edata or not edata["transitions"]:
                continue
            pmax = 0.0
            for _a, ar in edata["transitions"].items():
                pmax = max(pmax, float(ar["p"]))
            w = max(weight_eps, 2.0 - pmax)
            nd = du + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return float(unreachable)


def _get_outgoing_for_action(mdp: MDPNetwork, s: int, a: int) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    for sp in mdp.graph.successors(s):
        edata = mdp.graph[s][sp]
        if "transitions" in edata and a in edata["transitions"]:
            p = float(edata["transitions"][a]["p"])
            r = float(edata["transitions"][a]["r"])
            out[int(sp)] = (p, r)
    return out


def _set_outgoing_for_action(
    mdp: MDPNetwork,
    s: int,
    a: int,
    new_map: Dict[int, Tuple[float, float]],
    *,
    whitelist: Optional[Set[EdgeTriple]] = None,
    prob_floor: float = 1e-6,
):
    # Collect currently existing (s,a,*) to preserve protected ones
    existing = _get_outgoing_for_action(mdp, s, a)
    final_map = dict(new_map)

    if whitelist:
        for sp, (p_cur, r_cur) in existing.items():
            if (s, a, sp) in whitelist and sp not in final_map:
                final_map[sp] = (max(prob_floor, float(p_cur)), float(r_cur))

    # Drop all (s,a,*) and rebuild from final_map
    for sp in list(mdp.graph.successors(s)):
        edata = mdp.graph[s][sp]
        if "transitions" in edata and a in edata["transitions"]:
            del edata["transitions"][a]
            if not edata["transitions"]:
                mdp.graph.remove_edge(s, sp)

    for sp, (p, r) in final_map.items():
        mdp.add_transition(s, int(sp), a, probability=float(p), reward=float(r))

    mdp.renormalize_action(s, a)

def _list_all_action_pairs(mdp: MDPNetwork) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for s in mdp.states:
        if s in mdp.terminal_states:
            continue
        for a in range(mdp.num_actions):
            pairs.append((int(s), int(a)))
    return pairs


def _list_all_triples(mdp: MDPNetwork) -> List[EdgeTriple]:
    triples: List[EdgeTriple] = []
    for s in mdp.states:
        for sp in mdp.graph.successors(s):
            edata = mdp.graph[s][sp]
            if "transitions" not in edata:
                continue
            for a in edata["transitions"].keys():
                triples.append((int(s), int(a), int(sp)))
    return triples


def _allowed_nodes_within_scope(base_ref: MDPNetwork, s: int, distance_cfg: Dict[str, Any]) -> Set[int]:
    if distance_cfg.get("max_hops") is None:
        return set(base_ref.states)
    hop_dist = nx.single_source_shortest_path_length(base_ref.graph, source=s, cutoff=int(distance_cfg["max_hops"]))
    allowed = set(hop_dist.keys())
    node_cap = distance_cfg.get("node_cap", None)
    if node_cap is not None and len(allowed) > int(node_cap):
        kept = sorted(allowed, key=lambda x: hop_dist.get(x, 10**9))[: int(node_cap)]
        allowed = set(kept)
    return allowed


def _prune_low_prob_transitions(
    mdp: MDPNetwork,
    threshold: float,
    *,
    whitelist: Optional[Set[EdgeTriple]] = None,
    prob_floor: float = 1e-6,
):
    """
    Remove (s,a,sp) with p < threshold, EXCEPT those in whitelist.
    Whitelisted edges are kept as-is (we do NOT bump to threshold; only clamp by prob_floor on re-write paths).
    """
    thr = float(threshold)
    for s in mdp.states:
        if s in mdp.terminal_states:
            continue
        for a in range(mdp.num_actions):
            out_map = _get_outgoing_for_action(mdp, s, a)
            if not out_map:
                continue
            kept: Dict[int, Tuple[float, float]] = {}
            for sp, (p, r) in out_map.items():
                if p >= thr or (whitelist and (s, a, sp) in whitelist):
                    kept[sp] = (p, r)
            # Use protected setter so that any present whitelisted edges are preserved even if missing in `kept`
            _set_outgoing_for_action(
                mdp, s, a, kept,
                whitelist=whitelist,
                prob_floor=prob_floor,
            )


def _mutation_add_edge(
    mdp: MDPNetwork,
    rng: np.random.Generator,
    base_ref: MDPNetwork,
    ops: Dict[str, Any],
    distance_cfg: Dict[str, Any],
    *,
    whitelist: Optional[Set[EdgeTriple]] = None,
):
    max_out_degree = int(ops.get("max_out_degree", 8))
    allow_self_loops = bool(ops.get("allow_self_loops", True))
    add_edge_allow_out_of_scope = bool(ops.get("add_edge_allow_out_of_scope", True))
    epsilon_new_prob = float(ops.get("epsilon_new_prob", 0.02))
    gamma_sample = float(ops.get("gamma_sample", 1.0))
    gamma_prob = float(ops.get("gamma_prob", 0.0))
    prob_floor = float(ops.get("prob_floor", 1e-6))

    candidates_sa = [(s, a) for (s, a) in _list_all_action_pairs(mdp)
                     if sum(1 for _sp in _get_outgoing_for_action(mdp, s, a)) < max_out_degree]
    if not candidates_sa:
        return
    s, a = candidates_sa[rng.integers(0, len(candidates_sa))]
    existing = set(_get_outgoing_for_action(mdp, s, a).keys())
    sp_candidates = [sp for sp in mdp.states if (allow_self_loops or sp != s) and sp not in existing]

    if not add_edge_allow_out_of_scope:
        allowed = _allowed_nodes_within_scope(base_ref, s, distance_cfg)
        sp_candidates = [sp for sp in sp_candidates if sp in allowed]
    if not sp_candidates:
        return

    def dist(_s, _sp):
        return _directed_prob_distance(
            base_ref, _s, _sp,
            max_hops=distance_cfg.get("max_hops", None),
            node_cap=distance_cfg.get("node_cap", None),
            weight_eps=float(distance_cfg.get("weight_eps", 1e-9)),
            unreachable=float(distance_cfg.get("unreachable", 1e6)),
        )

    weights = np.asarray(
        [1.0 if gamma_sample <= 0.0 else math.exp(-gamma_sample * dist(s, sp))
         for sp in sp_candidates],
        dtype=float
    )
    if weights.sum() == 0.0:
        return
    weights /= weights.sum()

    sp_new = int(rng.choice(sp_candidates, p=weights))
    d_new = dist(s, sp_new)
    p_new = epsilon_new_prob if gamma_prob <= 0.0 else min(epsilon_new_prob, epsilon_new_prob * math.exp(-gamma_prob * d_new))

    # pick reward by inbound mean (fallback to default)
    def inbound_reward_mean(sp: int, fallback: float) -> float:
        vals: List[float] = []
        for ss in mdp.graph.predecessors(sp):
            edata = mdp.graph[ss][sp]
            if "transitions" not in edata:
                continue
            for _a, ar in edata["transitions"].items():
                vals.append(float(ar["r"]))
        return float(np.mean(vals)) if vals else float(fallback)

    r_new = inbound_reward_mean(sp_new, fallback=mdp.default_reward)

    out_map = _get_outgoing_for_action(mdp, s, a)
    for k in out_map:
        p, r = out_map[k]
        out_map[k] = (max(prob_floor, p * (1.0 - p_new)), r)
    out_map[sp_new] = (max(prob_floor, p_new), r_new)

    _set_outgoing_for_action(
        mdp, s, a, out_map,
        whitelist=whitelist,
        prob_floor=prob_floor,
    )


def _mutation_prob_pairwise(
    mdp: MDPNetwork,
    rng: np.random.Generator,
    ops: Dict[str, Any],
    *,
    whitelist: Optional[Set[EdgeTriple]] = None,
):
    k_actions = int(ops.get("prob_tweak_actions_per_child", 20))
    step = float(ops.get("prob_pairwise_step", 0.02))
    prob_floor = float(ops.get("prob_floor", 1e-6))
    pairs_sa = _list_all_action_pairs(mdp)
    if not pairs_sa:
        return
    for _ in range(k_actions):
        s, a = pairs_sa[rng.integers(0, len(pairs_sa))]
        out_map = _get_outgoing_for_action(mdp, s, a)
        if len(out_map) < 2:
            continue
        succs = list(out_map.keys())
        i, j = rng.choice(len(succs), size=2, replace=False)
        sp_i, sp_j = int(succs[i]), int(succs[j])
        p_i, r_i = out_map[sp_i]
        p_j, r_j = out_map[sp_j]
        delta_max = min(step, max(0.0, p_j - prob_floor))
        if delta_max <= 0:
            continue
        delta = rng.uniform(0.0, delta_max)
        out_map[sp_i] = (p_i + delta, r_i)
        out_map[sp_j] = (p_j - delta, r_j)

        # Renormalize by setter (we clamp by prob_floor when rebuilding)
        _set_outgoing_for_action(
            mdp, s, a, out_map,
            whitelist=whitelist,
            prob_floor=prob_floor,
        )


def _mutation_reward_smallstep(mdp: MDPNetwork, rng: np.random.Generator, ops: Dict[str, Any]):
    n_edges = int(ops.get("reward_tweak_edges_per_child", 50))
    k_percent = float(ops.get("reward_k_percent", 0.02))
    ref_floor = float(ops.get("reward_ref_floor", 1e-3))
    rmin = ops.get("reward_min", None)
    rmax = ops.get("reward_max", None)
    triples = _list_all_triples(mdp)
    if not triples:
        return
    for _ in range(n_edges):
        s, a, sp = triples[rng.integers(0, len(triples))]
        r_cur = mdp.get_transition_reward(s, a, sp)
        delta_max = k_percent * max(abs(r_cur), ref_floor)
        delta = rng.uniform(-delta_max, +delta_max)
        r_new = r_cur + delta
        if rmin is not None:
            r_new = max(float(rmin), r_new)
        if rmax is not None:
            r_new = min(float(rmax), r_new)
        mdp.update_transition_reward(s, a, sp, float(r_new))


def _crossover_action_block(
    pa: MDPNetwork,
    pb: MDPNetwork,
    rng: np.random.Generator,
    *,
    whitelist: Optional[Set[EdgeTriple]] = None,
    prob_floor: float = 1e-6,
) -> MDPNetwork:
    """
    For each (s,a) copy the entire outgoing map from either parent,
    but do NOT drop any (s,a,sp) that is already present in the child and is whitelisted.
    """
    child = pa.clone()
    for s in child.states:
        if s in child.terminal_states:
            continue
        for a in range(child.num_actions):
            src = pa if (rng.random() < 0.5) else pb
            src_map = _get_outgoing_for_action(src, s, a)
            if not src_map:
                continue
            _set_outgoing_for_action(
                child, s, a, src_map,
                whitelist=whitelist,
                prob_floor=prob_floor,
            )
    return child


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


def _compute_crowding_distance(objs: List[List[float]], idxs: List[int]) -> Dict[int, float]:
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
    def __init__(self,
                 base_portable: Dict[str, Any],
                 whitelist: List[Tuple[int, int, int]],
                 ops: Dict[str, Any],
                 distance_cfg: Dict[str, Any],
                 solver: Dict[str, Any],
                 precomputed_portables: Optional[List[Dict[str, Any]]] = None):
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
        self.precomputed_portables = precomputed_portables

    def mutate(self,
               seed: int,
               pa_portable: Optional[Dict[str, Any]] = None,
               pb_portable: Optional[Dict[str, Any]] = None,
               do_crossover: bool = False) -> Dict[str, Any]:
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
                        pa, pb, rng,
                        whitelist=self.whitelist,
                        prob_floor=prob_floor,
                    ) if do_crossover else
                    (pa if rng.random() < 0.5 else pb).clone()
                )

        # Apply mutations (all protected by whitelist)
        for _ in range(int(self.ops.get("add_edge_attempts_per_child", 2))):
            _mutation_add_edge(
                ind, rng, self.base_ref, self.ops, self.distance,
                whitelist=self.whitelist,
            )
        _mutation_prob_pairwise(
            ind, rng, self.ops,
            whitelist=self.whitelist,
        )
        if int(self.ops.get("reward_tweak_edges_per_child", 50)) > 0:
            _mutation_reward_smallstep(ind, rng, self.ops)  # reward tweak doesn't remove edges

        if self.ops.get("prune_prob_threshold", None) is not None:
            _prune_low_prob_transitions(
                ind,
                float(self.ops["prune_prob_threshold"]),
                whitelist=self.whitelist,
                prob_floor=prob_floor,
            )

        return ind.to_portable()


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
    workers: int,
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
    score: Optional[Dict[str, Any]] = None,
) -> Tuple[List[MDPNetwork], List[List[float]], List[MDPNetwork], List[List[float]]]:

    """
    Returns: pareto_mdps, pareto_objs, pop, pop_objs
    """
    ops = dict(ops or {})
    distance = dict(distance or {})
    solver = dict(solver or {})
    score = dict(score or {"fns": [{"name": "obj_multi_perf", "params": {}}]})

    # small util
    def _ensure_dir(p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    # logger
    logger = logging.getLogger("ga")
    if not logger.handlers:
        _h = logging.StreamHandler(sys.stdout)
        _h.setFormatter(logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(_h)
        logger.setLevel(logging.INFO)

    # init ray once
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # ===== Precompute baseline policy/occupancy =====
    gamma = float(solver.get("vi_gamma", 0.99))
    theta = float(solver.get("vi_theta", 1e-6))
    max_iters = int(solver.get("vi_max_iterations", 1000))
    policy_temp = float(solver.get("policy_temperature", 1.0))
    policy_mixing = tuple(solver.get("policy_mixing", (0.0, 1.0, 0.0)))
    tie_tol = float(solver.get("policy_tie_tol", 1e-6))

    t0 = time.perf_counter()
    _, Q = optimal_value_iteration(base_mdp, gamma=gamma, theta=theta, max_iterations=max_iters)
    base_policy = q_table_to_policy(
        Q,
        states=list(base_mdp.states),
        num_actions=base_mdp.num_actions,
        mixing=policy_mixing,
        temperature=policy_temp,
        tie_tol=tie_tol,
    )
    base_occupancy = compute_occupancy_measure(base_mdp, base_policy, gamma=gamma, theta=theta, max_iterations=max_iters)
    precomputed = [base_policy.to_portable(), base_occupancy.to_portable()]
    t1 = time.perf_counter()
    # Log precompute timing through WandbWriter (optional).
    if wandb_writer is not None:
        wandb_writer.log.remote({"gen": -1, "time/precompute_sec": float(t1 - t0)})

    # ===== Build worker pool =====
    whitelist = _list_all_triples(base_mdp)
    base_portable = base_mdp.to_portable()
    pool = [
        GAWorker.options(num_cpus=1).remote(
            base_portable=base_portable,
            whitelist=whitelist,
            ops=ops,
            distance_cfg=distance,
            solver=solver,
            precomputed_portables=precomputed,
        ) for _ in range(max(1, int(workers)))
    ]

    # master RNG for driver-only randomness (deterministic)
    rng_drv = np.random.default_rng(_derive_seed(seed, "driver"))

    # ===== Init population =====
    pop: List[MDPNetwork] = [base_mdp.clone()]
    need = population_size - 1
    if need > 0:
        futs = []
        for i in range(need):
            actor = pool[i % len(pool)]
            futs.append(actor.mutate.remote(seed=_derive_seed(seed, "init", i)))
        children_portables = ray.get(futs)
        pop.extend([MDPNetwork.from_portable(p) for p in children_portables])

    # ===== Evaluate population=====
    def _score_portables(portables: List[Dict[str, Any]]) -> List[List[float]]:
        W = len(pool)
        if W == 0:
            return []
        chunks: List[List[Dict[str, Any]]] = [[] for _ in range(W)]
        idx_chunks: List[List[int]] = [[] for _ in range(W)]
        for i, p in enumerate(portables):
            w = i % W
            chunks[w].append(p)
            idx_chunks[w].append(i)
        futs = []
        active_wids: List[int] = []
        for wid, ch in enumerate(chunks):
            if ch:
                futs.append(pool[wid].score_batch.remote(ch, score))
                active_wids.append(wid)
        parts = ray.get(futs)
        out: List[Optional[List[float]]] = [None] * len(portables)
        for part, wid in zip(parts, active_wids):
            for j, obj in zip(idx_chunks[wid], part):
                out[j] = obj
        return [list(map(float, row)) for row in out]  # type: ignore

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
        payload = {"gen": 0, "init/pop_size": int(len(pop))}
        M = len(objs[0]) if objs else 0
        for m in range(M):
            payload[f"init/obj{m}_min"] = init_stats.get(f"min_{m}", float("nan"))
            payload[f"init/obj{m}_mean"] = init_stats.get(f"mean_{m}", float("nan"))
            payload[f"init/obj{m}_max"] = init_stats.get(f"max_{m}", float("nan"))
        wandb_writer.log.remote(payload)

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
            order_prev = sorted(range(len(pop)),
                                key=lambda i: (ranks[i], -crowding.get(i, 0.0)))
            elite_parent_idxs = set(order_prev[:elite_k])
        else:
            elite_parent_idxs = set()

        # --- parent selection (tournament) ---
        parents_pairs: List[Tuple[MDPNetwork, MDPNetwork]] = []
        for k in range(population_size):
            # first parent
            idxs = rng_drv.choice(len(pop), size=int(tournament_k), replace=False)
            best = int(idxs[0])
            for j in idxs[1:]:
                j = int(j)
                if ranks[j] < ranks[best] or (ranks[j] == ranks[best] and crowding.get(j, 0.0) > crowding.get(best, 0.0)):
                    best = j
            # second parent
            idxs2 = rng_drv.choice(len(pop), size=int(tournament_k), replace=False)
            best2 = int(idxs2[0])
            for j in idxs2[1:]:
                j = int(j)
                if ranks[j] < ranks[best2] or (ranks[j] == ranks[best2] and crowding.get(j, 0.0) > crowding.get(best2, 0.0)):
                    best2 = j
            parents_pairs.append((pop[best], pop[best2]))
        t_sel = time.perf_counter()

        # --- offspring (parallel) ---
        futs = []
        for k, (pa, pb) in enumerate(parents_pairs):
            actor = pool[k % len(pool)]
            do_x = (rng_drv.random() < float(crossover_rate))
            futs.append(
                actor.mutate.remote(
                    seed=_derive_seed(seed, "child", gen, k),
                    pa_portable=pa.to_portable(),
                    pb_portable=pb.to_portable(),
                    do_crossover=bool(do_x),
                )
            )
        child_portables = ray.get(futs)
        children = [MDPNetwork.from_portable(p) for p in child_portables]
        t_child = time.perf_counter()

        # --- evaluate children  ---
        child_objs = _score_portables([c.to_portable() for c in children])
        t_eval = time.perf_counter()

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
                sorted_F = sorted(F_remaining, key=lambda i: dist.get(i, 0.0), reverse=True)
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

        # --- logging ---
        gen_stats = _summ_stats(objs)
        logger.info(
            "[Gen %d/%d] pop=%d | %s | F1=%d",
            gen + 1, generations, len(pop),
            " | ".join(
                f"obj{m}: min={gen_stats.get(f'min_{m}', float('nan')):.4f} "
                f"mean={gen_stats.get(f'mean_{m}', float('nan')):.4f} "
                f"max={gen_stats.get(f'max_{m}', float('nan')):.4f}"
                for m in range(len(objs[0]) if objs else 0)
            )
            or "NA",
            len(fronts[0]) if fronts else 0,
        )
        gend = time.perf_counter()
        if wandb_writer is not None:
            payload = {
                "gen": gen + 1,
                "pop/size": int(len(pop)),
                "pop/F1_size": int(len(fronts[0]) if fronts else 0),
                "time/selection_sec": float(t_child - t_sel),
                "time/offspring_sec": float(t_eval - t_child),
                "time/eval_sec": float(gend - t_eval),
                "time/total_gen_sec": float(gend - gstart),
            }
            M = len(objs[0]) if objs else 0
            for m in range(M):
                payload[f"pop/obj{m}_min"] = gen_stats.get(f"min_{m}", float("nan"))
                payload[f"pop/obj{m}_mean"] = gen_stats.get(f"mean_{m}", float("nan"))
                payload[f"pop/obj{m}_max"] = gen_stats.get(f"max_{m}", float("nan"))
            wandb_writer.log.remote(payload)

    # ===== Final Pareto =====
    final_fronts = _fast_non_dominated_sort(objs)
    F1 = final_fronts[0] if final_fronts else list(range(len(pop)))
    pareto_mdps = [pop[i].clone() for i in F1]
    pareto_objs = [objs[i][:] for i in F1]

    # optional: save to disk
    if output_dir:
        mdp_out_dir = Path(output_dir) / "ga" / "mdps"
        _ensure_dir(mdp_out_dir)
        for i, (m, objv) in enumerate(zip(pareto_mdps, pareto_objs)):
            tag = "_".join(f"{v:.4f}" for v in objv)
            p = mdp_out_dir / f"pareto_{i}_objs_{tag}.json"
            m.export_to_json(str(p))
            logger.info("[GA] Saved PF[%d] -> %s", i, p.name)

    if wandb_writer is not None:
        payload = {"gen": generations, "final/F1_size": int(len(F1))}
        M = len(objs[0]) if objs else 0
        fstats = _summ_stats([objs[i] for i in F1] if F1 else objs)
        for m in range(M):
            payload[f"final/obj{m}_min"] = fstats.get(f"min_{m}", float("nan"))
            payload[f"final/obj{m}_mean"] = fstats.get(f"mean_{m}", float("nan"))
            payload[f"final/obj{m}_max"] = fstats.get(f"max_{m}", float("nan"))
        wandb_writer.log.remote(payload)

    return pareto_mdps, pareto_objs, pop, objs


# =========================================================
# Example score functions (new unified signature)
# =========================================================

def obj_multi_kl_and_perf(mdp: MDPNetwork, shared: Dict[str, Any], *,
                          kl_delta: float = 1e-3) -> Sequence[float]:
    """
    Returns [ -KL(baseline || current), performance_integral ].
    Uses shared['solver'] defaults and shared['precomputed'] = [base_policy, base_occupancy].
    """
    solver = shared.get("solver", {})
    gamma = float(solver.get("vi_gamma", 0.99))
    theta = float(solver.get("vi_theta", 1e-6))
    max_iter = int(solver.get("vi_max_iterations", 1000))
    temperature = float(solver.get("policy_temperature", 1.0))
    mixing = tuple(solver.get("policy_mixing", (0.0, 1.0, 0.0)))
    tie_tol = float(solver.get("policy_tie_tol", 1e-6))

    pgamma = float(solver.get("perf_gamma", gamma))
    ptheta = float(solver.get("perf_theta", theta))
    pmax_iter = int(solver.get("perf_max_iterations", max_iter))
    numpoints = int(solver.get("perf_numpoints", 100))

    pre = shared.get("precomputed", None) or []
    base_policy = PolicyTable.from_portable(pre[0]) if len(pre) >= 1 else None
    base_occupancy = ValueTable.from_portable(pre[1]) if len(pre) >= 2 else None

    _, Q2 = optimal_value_iteration(mdp, gamma=gamma, theta=theta, max_iterations=max_iter)
    policy2: PolicyTable = q_table_to_policy(
        Q2, states=list(mdp.states), num_actions=mdp.num_actions,
        mixing=mixing, temperature=temperature, tie_tol=tie_tol,
    )
    occupancy2: ValueTable = compute_occupancy_measure(mdp, policy=policy2, gamma=gamma, theta=theta, max_iterations=max_iter)

    if base_policy is not None and base_occupancy is not None:
        kl = kl_policies(policy1=base_policy, occupancy1=base_occupancy,
                         policy2=policy2, occupancy2=occupancy2, delta=float(kl_delta))
        obj1 = -float(kl)
    else:
        obj1 = 0.0

    prior = create_random_policy(mdp)  # NOTE: may use its own RNG; same行为与旧实现一致
    _curve, integral = performance_curve_and_integral(
        prior_policy=prior, target_policy=policy2, mdp_network=mdp,
        numpoints=numpoints, gamma=pgamma, theta=ptheta, max_iterations=pmax_iter,
    )
    return [obj1, float(integral)]


def obj_multi_perf(mdp: MDPNetwork, shared: Dict[str, Any], *,
                   blend_weight: float = 0.8) -> Sequence[float]:
    """
    Returns two integrals:
      1) integral( prior = blend(policy2, random, w), target = baseline_policy )
      2) integral( prior = random, target = policy2 )
    """
    solver = shared.get("solver", {})
    gamma = float(solver.get("vi_gamma", 0.99))
    theta = float(solver.get("vi_theta", 1e-6))
    max_iter = int(solver.get("vi_max_iterations", 1000))
    temperature = float(solver.get("policy_temperature", 1.0))
    mixing = tuple(solver.get("policy_mixing", (0.0, 1.0, 0.0)))
    tie_tol = float(solver.get("policy_tie_tol", 1e-6))

    pgamma = float(solver.get("perf_gamma", gamma))
    ptheta = float(solver.get("perf_theta", theta))
    pmax_iter = int(solver.get("perf_max_iterations", max_iter))
    numpoints = int(solver.get("perf_numpoints", 100))

    pre = shared.get("precomputed", None) or []
    base_policy = PolicyTable.from_portable(pre[0]) if len(pre) >= 1 else None

    _, Q2 = optimal_value_iteration(mdp, gamma=gamma, theta=theta, max_iterations=max_iter)
    policy2: PolicyTable = q_table_to_policy(
        Q2, states=list(mdp.states), num_actions=mdp.num_actions,
        mixing=mixing, temperature=temperature, tie_tol=tie_tol,
    )

    prior_rand = create_random_policy(mdp)
    blended = blend_policies(policy2, prior_rand, weight=float(blend_weight))

    if base_policy is not None:
        _curve, integral1 = performance_curve_and_integral(
            prior_policy=blended, target_policy=base_policy, mdp_network=mdp,
            numpoints=numpoints, gamma=pgamma, theta=ptheta, max_iterations=pmax_iter,
        )
    else:
        integral1 = 0.0

    _curve, integral2 = performance_curve_and_integral(
        prior_policy=prior_rand, target_policy=policy2, mdp_network=mdp,
        numpoints=numpoints, gamma=pgamma, theta=ptheta, max_iterations=pmax_iter,
    )
    return [float(integral1), float(integral2)]


# Default registrations
register_score_fn("obj_multi_kl_and_perf", obj_multi_kl_and_perf)
register_score_fn("obj_multi_perf", obj_multi_perf)
