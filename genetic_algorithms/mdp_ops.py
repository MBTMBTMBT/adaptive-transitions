from __future__ import annotations

import math
from typing import Tuple, Optional, Dict, List, Set, Any

import networkx as nx
import numpy as np

from mdp_network import MDPNetwork

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
    """Dijkstra on directed graph with edge weight w(u->v) = max(weight_eps, 2 - max_a P(v|u,a))."""
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
    """Rewrite all (s,a,*) with new_map; preserve whitelisted ones; clamp by prob_floor; renormalize."""
    existing = _get_outgoing_for_action(mdp, s, a)
    final_map = dict(new_map)

    if whitelist:
        for sp, (p_cur, r_cur) in existing.items():
            if (s, a, sp) in whitelist and sp not in final_map:
                final_map[sp] = (max(prob_floor, float(p_cur)), float(r_cur))

    # drop & rebuild
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
    """Return states within hop-based scope (optional capped by node_cap)."""
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
    """
    Add a new (s,a,sp_new) with small mass p_new, favoring closer nodes by a distance bias.
    This version caches dist(s->sp) within the call to avoid repeated shortest-path solves.
    """
    # ---- knobs ----
    max_out_degree = int(ops.get("max_out_degree", 8))
    allow_self_loops = bool(ops.get("allow_self_loops", True))
    add_edge_allow_out_of_scope = bool(ops.get("add_edge_allow_out_of_scope", True))
    epsilon_new_prob = float(ops.get("epsilon_new_prob", 0.02))
    gamma_sample = float(ops.get("gamma_sample", 1.0))
    gamma_prob = float(ops.get("gamma_prob", 0.0))
    prob_floor = float(ops.get("prob_floor", 1e-6))

    # ---- choose (s,a) with room ----
    candidates_sa = [
        (s, a) for (s, a) in _list_all_action_pairs(mdp)
        if sum(1 for _sp in _get_outgoing_for_action(mdp, s, a)) < max_out_degree
    ]
    if not candidates_sa:
        return
    s, a = candidates_sa[rng.integers(0, len(candidates_sa))]
    existing = set(_get_outgoing_for_action(mdp, s, a).keys())

    # ---- candidate sp set ----
    sp_candidates = [sp for sp in mdp.states if (allow_self_loops or sp != s) and (sp not in existing)]
    if not sp_candidates:
        return
    if not add_edge_allow_out_of_scope:
        allowed = _allowed_nodes_within_scope(base_ref, s, distance_cfg)
        sp_candidates = [sp for sp in sp_candidates if sp in allowed]
        if not sp_candidates:
            return

    # ---- cache distances ----
    dist_cache: Dict[int, float] = {}

    def dist_cached(dst: int) -> float:
        if dst in dist_cache:
            return dist_cache[dst]
        d = _directed_prob_distance(
            base_ref, s, dst,
            max_hops=distance_cfg.get("max_hops", None),
            node_cap=distance_cfg.get("node_cap", None),
            weight_eps=float(distance_cfg.get("weight_eps", 1e-9)),
            unreachable=float(distance_cfg.get("unreachable", 1e6)),
        )
        dist_cache[dst] = d
        return d

    # ---- sampling weights over sp ----
    if gamma_sample <= 0.0:
        weights = np.full(len(sp_candidates), 1.0 / float(len(sp_candidates)), dtype=float)
    else:
        raw = np.asarray([math.exp(-gamma_sample * dist_cached(sp)) for sp in sp_candidates], dtype=float)
        total = float(raw.sum())
        if total <= 0.0 or not np.isfinite(total):
            return
        weights = raw / total

    sp_new = int(rng.choice(sp_candidates, p=weights))
    d_new = dist_cached(sp_new)

    # ---- assign probability mass ----
    p_new = epsilon_new_prob if gamma_prob <= 0.0 else min(epsilon_new_prob, epsilon_new_prob * math.exp(-gamma_prob * d_new))

    # ---- heuristic reward for new edge: inbound mean fallback ----
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

    # ---- rebuild (s,a) map: shrink others, add new ----
    out_map = _get_outgoing_for_action(mdp, s, a)
    for k in list(out_map.keys()):
        p_k, r_k = out_map[k]
        out_map[k] = (max(prob_floor, p_k * (1.0 - p_new)), r_k)
    out_map[sp_new] = (max(prob_floor, p_new), r_new)

    _set_outgoing_for_action(mdp, s, a, out_map, whitelist=whitelist, prob_floor=prob_floor)


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
