# ga_mdp_search.py
# Genetic search system for MDPNetwork (structure + probability + reward)
# with optional parallel scoring and parallel offspring mutation using a process pool (Linux, Python 3.10).
#
# W&B integration (simplified):
# - This module DOES NOT call wandb.init(); pass an external run via MDPEvolutionGA(..., wb_run=run).
# - We only log obvious running status (scalars) and real-time curves:
#   * Per-generation scalar stats: min/mean/max for each objective, population size, F1 size, timings.
#   * Per-generation scatter of current population if num_objectives < 3:
#       - 1D -> scatter(ind_idx, obj0)
#       - 2D -> scatter(obj0, obj1)
#   * If num_objectives >= 3 -> no plots; we log a compact per-gen table (ind_idx + obj*).
# - One-off outputs (e.g., final Pareto front) are saved as compact tables; no figures.
# - No artifacts, no frequent large tables, no silent try/except around logging.
#
# NSGA-II multi-objective + multi-output score support + per-score constants:
# - A score function may return either a float OR a sequence of floats.
# - register_score_fn(..., const=dict) lets you pass per-function constant params at registration time.
#   They are injected as kwargs["score_const"] in workers.
# - Serial precompute on base MDP: VI -> Q -> softmax policy + occupancy, broadcast to workers.
# - Distance and search scope are ALWAYS computed on the base MDP (initial environment).

from __future__ import annotations

from typing import Callable, Dict, Tuple, List, Optional, Set, Any
from dataclasses import dataclass
import math
import os
import sys
import time
import logging

import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor

import wandb  # required by request; we don't init here

from mdp_network.mdp_tables import q_table_to_policy, PolicyTable, ValueTable, create_random_policy, blend_policies
from mdp_network.metrics import kl_policies, performance_curve_and_integral
from mdp_network.solvers import optimal_value_iteration, compute_occupancy_measure
from serialisable import Serialisable
from mdp_network import MDPNetwork

# -------------------------------
# Types
# -------------------------------

State = int
Action = int
EdgeTriple = Tuple[State, Action, State]
ScoreFn = Callable[['MDPNetwork', Any], Any]  # may return float or sequence
DistanceFn = Callable[['MDPNetwork', State, State], float]

# -------------------------------
# Score function registry
# -------------------------------

SCORE_FN_REGISTRY: Dict[str, ScoreFn] = {}

def register_score_fn(name: str, fn: ScoreFn) -> None:
    """
    Register a score function.
    - name: registry key used in GAConfig.score_fn_names
    - fn:   callable(mdp, *args, **kwargs) -> float | sequence[float]
    """
    SCORE_FN_REGISTRY[name] = fn

def get_registered_score_fn(name: str) -> ScoreFn:
    if name not in SCORE_FN_REGISTRY:
        raise KeyError(f"Score function '{name}' is not registered.")
    return SCORE_FN_REGISTRY[name]

# -------------------------------
# Distance (ALWAYS computed on base MDP via the caller)
# -------------------------------

def directed_prob_distance(
    mdp: 'MDPNetwork',
    s: State,
    sp: State,
    *,
    max_hops: Optional[int],
    node_cap: Optional[int],
    weight_eps: float,
    unreachable: float,
) -> float:
    """Dijkstra on a directed graph with edge weights w(u->v)=max(weight_eps, 2.0 - max_a P(v|u,a))."""
    if s == sp:
        return 0.0
    G = mdp.graph

    # BFS-reachable scope
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

    # Optional cap
    if node_cap is not None and len(allowed) > node_cap:
        if 'hop_dist' in locals():
            kept = sorted(allowed, key=lambda x: hop_dist.get(x, 10**9))[:node_cap]
            allowed = set(kept)
        else:
            allowed = set(list(allowed)[:node_cap])

    if sp not in allowed:
        return float(unreachable)

    import heapq
    INF = float('inf')
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

def _allowed_nodes_within_scope(mdp_ref: 'MDPNetwork', s: State, cfg: 'GAConfig') -> Set[int]:
    """Nodes inside the search scope on the REFERENCE graph (base MDP). If dist_max_hops is None -> all states."""
    if cfg.dist_max_hops is None:
        return set(mdp_ref.states)
    hop_dist = nx.single_source_shortest_path_length(mdp_ref.graph, source=s, cutoff=cfg.dist_max_hops)
    allowed = set(hop_dist.keys())
    if cfg.dist_node_cap is not None and len(allowed) > cfg.dist_node_cap:
        kept = sorted(allowed, key=lambda x: hop_dist.get(x, 10**9))[:cfg.dist_node_cap]
        allowed = set(kept)
    return allowed

# -------------------------------
# GA config
# -------------------------------

@dataclass
class GAConfig:
    # Population / evolution
    population_size: int = 80
    generations: int = 100
    tournament_k: int = 2
    elitism_num: int = 8
    crossover_rate: float = 1.0

    # Structure constraints
    allow_self_loops: bool = True
    min_out_degree: int = 1
    max_out_degree: int = 8
    prob_floor: float = 1e-6

    # Add-edge mutation
    add_edge_attempts_per_child: int = 2
    epsilon_new_prob: float = 0.02
    gamma_sample: float = 1.0
    gamma_prob: float = 0.0
    # Forbid adding edges to out-of-scope nodes (scope defined on base MDP). If True -> allow all.
    add_edge_allow_out_of_scope: bool = True

    # Hard pruning by threshold
    prune_prob_threshold: Optional[float] = None

    # Probability tweaks
    prob_tweak_actions_per_child: int = 20
    prob_pairwise_step: float = 0.02

    # Reward tweaks
    reward_tweak_edges_per_child: int = 50
    reward_k_percent: float = 0.02
    reward_ref_floor: float = 1e-3
    reward_min: Optional[float] = None
    reward_max: Optional[float] = None

    # Parallel evaluation (multi-objective)
    n_workers: int = 1
    score_fn_names: Optional[List[str]] = None
    score_args: Optional[Tuple[Any, ...]] = None
    score_kwargs: Optional[Dict[str, Any]] = None

    # Parallel mutation
    mutation_n_workers: int = 1

    # Distance parameters (applied on base MDP)
    dist_max_hops: Optional[float] = None
    dist_node_cap: Optional[int] = None
    dist_weight_eps: float = 1e-9
    dist_unreachable: float = 1e6

    # Randomness
    seed: Optional[int] = None

    # Solver / score params
    vi_gamma: float = 0.99
    vi_theta: float = 1e-6
    vi_max_iterations: int = 1000

    policy_temperature: float = 1.0
    policy_mixing: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    policy_tie_tol: float = 1e-6

    perf_numpoints: int = 100
    perf_gamma: Optional[float] = None
    perf_theta: Optional[float] = None
    perf_max_iterations: Optional[int] = None

    # -------------- Simplified W&B controls --------------
    wandb_enabled: bool = True  # A wandb run MUST be passed into MDPEvolutionGA

# -------------------------------
# Utilities over MDPNetwork
# -------------------------------

def get_outgoing_for_action(mdp: 'MDPNetwork', s: State, a: Action) -> Dict[State, Tuple[float, float]]:
    out: Dict[State, Tuple[float, float]] = {}
    for sp in mdp.graph.successors(s):
        edata = mdp.graph[s][sp]
        if "transitions" in edata and a in edata["transitions"]:
            p = float(edata["transitions"][a]["p"])
            r = float(edata["transitions"][a]["r"])
            out[int(sp)] = (p, r)
    return out

def set_outgoing_for_action(mdp: 'MDPNetwork', s: State, a: Action, new_map: Dict[State, Tuple[float, float]]):
    for sp in list(mdp.graph.successors(s)):
        edata = mdp.graph[s][sp]
        if "transitions" in edata and a in edata["transitions"]:
            del edata["transitions"][a]
            if not edata["transitions"]:
                mdp.graph.remove_edge(s, sp)
    for sp, (p, r) in new_map.items():
        mdp.add_transition(s, sp, a, probability=float(p), reward=float(r))
    mdp.renormalize_action(s, a)

def inbound_reward_mean(mdp: 'MDPNetwork', sp: State, fallback: float) -> float:
    vals: List[float] = []
    for s in mdp.graph.predecessors(sp):
        edata = mdp.graph[s][sp]
        if "transitions" not in edata:
            continue
        for a, ar in edata["transitions"].items():
            vals.append(float(ar["r"]))
    return float(np.mean(vals)) if vals else float(fallback)

def action_out_degree(mdp: 'MDPNetwork', s: State, a: Action) -> int:
    return sum(
        1 for sp in mdp.graph.successors(s)
        if "transitions" in mdp.graph[s][sp] and a in mdp.graph[s][sp]["transitions"]
    )

def list_all_action_pairs(mdp: 'MDPNetwork') -> List[Tuple[State, Action]]:
    pairs: List[Tuple[State, Action]] = []
    for s in mdp.states:
        if s in mdp.terminal_states:
            continue
        for a in range(mdp.num_actions):
            pairs.append((s, a))
    return pairs

def list_all_triples(mdp: 'MDPNetwork') -> List[EdgeTriple]:
    triples: List[EdgeTriple] = []
    for s in mdp.states:
        for sp in mdp.graph.successors(s):
            edata = mdp.graph[s][sp]
            if "transitions" not in edata:
                continue
            for a in edata["transitions"].keys():
                triples.append((int(s), int(a), int(sp)))
    return triples

def build_original_whitelist(mdp: 'MDPNetwork') -> Set[EdgeTriple]:
    return set(list_all_triples(mdp))

# -------------------------------
# Mutation operators
# -------------------------------

def _dist_from_cfg(mdp: 'MDPNetwork', s: State, sp: State, cfg: GAConfig) -> float:
    # NOTE: callers ensure `mdp` is the REFERENCE (base MDP).
    return directed_prob_distance(
        mdp, s, sp,
        max_hops=cfg.dist_max_hops,
        node_cap=cfg.dist_node_cap,
        weight_eps=cfg.dist_weight_eps,
        unreachable=cfg.dist_unreachable,
    )

def mutation_add_edge(mdp: 'MDPNetwork',
                      rng: np.random.Generator,
                      dist_fn: DistanceFn,
                      cfg: GAConfig,
                      scope_fn: Optional[Callable[[State], Set[int]]] = None):
    """
    Add one edge with distance-weighted sampling of target states.
    Distance and scope MUST be computed on the base MDP via dist_fn/scope_fn.
    """
    candidates_sa = [(s, a) for (s, a) in list_all_action_pairs(mdp)
                     if action_out_degree(mdp, s, a) < cfg.max_out_degree]
    if not candidates_sa:
        return

    s, a = candidates_sa[rng.integers(0, len(candidates_sa))]
    existing = set(get_outgoing_for_action(mdp, s, a).keys())

    sp_candidates = [sp for sp in mdp.states
                     if (cfg.allow_self_loops or sp != s) and sp not in existing]

    # Restrict to scope on the reference graph if requested
    if not cfg.add_edge_allow_out_of_scope and scope_fn is not None:
        allowed = scope_fn(s)
        sp_candidates = [sp for sp in sp_candidates if sp in allowed]

    if not sp_candidates:
        return

    # Distance-weighted sampling using reference distance
    weights = []
    for sp in sp_candidates:
        d = dist_fn(mdp, s, sp)
        w = 1.0 if cfg.gamma_sample <= 0.0 else math.exp(-cfg.gamma_sample * d)
        weights.append(w)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0.0:
        return
    weights /= weights.sum()

    sp_new = int(rng.choice(sp_candidates, p=weights))
    d_new = dist_fn(mdp, s, sp_new)
    p_new = cfg.epsilon_new_prob
    if cfg.gamma_prob > 0.0:
        p_new = min(cfg.epsilon_new_prob, cfg.epsilon_new_prob * math.exp(-cfg.gamma_prob * d_new))
    r_new = inbound_reward_mean(mdp, sp_new, fallback=mdp.default_reward)

    out_map = get_outgoing_for_action(mdp, s, a)
    for k in out_map:
        p, r = out_map[k]
        out_map[k] = (max(cfg.prob_floor, p * (1.0 - p_new)), r)
    out_map[sp_new] = (max(cfg.prob_floor, p_new), r_new)
    set_outgoing_for_action(mdp, s, a, out_map)

def prune_low_prob_transitions(mdp: 'MDPNetwork', threshold: float):
    thr = float(threshold)
    for s in mdp.states:
        if s in mdp.terminal_states:
            continue
        for a in range(mdp.num_actions):
            out_map = get_outgoing_for_action(mdp, s, a)
            if not out_map:
                continue
            kept = {sp: (p, r) for sp, (p, r) in out_map.items() if p >= thr}
            if len(kept) != len(out_map):
                set_outgoing_for_action(mdp, s, a, kept)

def mutation_prob_pairwise(mdp: 'MDPNetwork', rng: np.random.Generator, cfg: GAConfig):
    pairs_sa = list_all_action_pairs(mdp)
    if not pairs_sa:
        return
    for _ in range(cfg.prob_tweak_actions_per_child):
        s, a = pairs_sa[rng.integers(0, len(pairs_sa))]
        out_map = get_outgoing_for_action(mdp, s, a)
        if len(out_map) < 2:
            continue
        succs = list(out_map.keys())
        i, j = rng.choice(len(succs), size=2, replace=False)
        sp_i, sp_j = int(succs[i]), int(succs[j])
        p_i, r_i = out_map[sp_i]
        p_j, r_j = out_map[sp_j]
        delta_max = min(cfg.prob_pairwise_step, max(0.0, p_j - cfg.prob_floor))
        if delta_max <= 0:
            continue
        delta = rng.uniform(0.0, delta_max)
        out_map[sp_i] = (p_i + delta, r_i)
        out_map[sp_j] = (p_j - delta, r_j)
        total = sum(p for (p, _) in out_map.values())
        if total > 0:
            for sp in out_map:
                p, r = out_map[sp]
                out_map[sp] = (max(cfg.prob_floor, p / total), r)
        set_outgoing_for_action(mdp, s, a, out_map)

def mutation_reward_smallstep(mdp: 'MDPNetwork', rng: np.random.Generator, cfg: GAConfig):
    triples = list_all_triples(mdp)
    if not triples:
        return
    for _ in range(cfg.reward_tweak_edges_per_child):
        s, a, sp = triples[rng.integers(0, len(triples))]
        r_cur = mdp.get_transition_reward(s, a, sp)
        delta_max = cfg.reward_k_percent * max(abs(r_cur), cfg.reward_ref_floor)
        delta = rng.uniform(-delta_max, +delta_max)
        r_new = r_cur + delta
        if cfg.reward_min is not None:
            r_new = max(cfg.reward_min, r_new)
        if cfg.reward_max is not None:
            r_new = min(cfg.reward_max, r_new)
        mdp.update_transition_reward(s, a, sp, float(r_new))

# -------------------------------
# Crossover
# -------------------------------

def crossover_action_block(parent_a: 'MDPNetwork', parent_b: 'MDPNetwork', rng: np.random.Generator) -> 'MDPNetwork':
    child = parent_a.clone()
    for s in child.states:
        if s in child.terminal_states:
            continue
        for a in range(child.num_actions):
            src = parent_a if (rng.random() < 0.5) else parent_b
            src_map = get_outgoing_for_action(src, s, a)
            if not src_map:
                continue
            set_outgoing_for_action(child, s, a, src_map)
    return child

# -------------------------------
# NSGA-II tools (maximization)
# -------------------------------

def _dominates_max(a: List[float], b: List[float]) -> bool:
    ge = True; gt = False
    for ai, bi in zip(a, b):
        if ai < bi:
            ge = False; break
        if ai > bi:
            gt = True
    return ge and gt

def fast_non_dominated_sort(objs: List[List[float]]) -> List[List[int]]:
    N = len(objs)
    S = [set() for _ in range(N)]
    n = [0] * N
    fronts: List[List[int]] = [[]]
    for p in range(N):
        for q in range(N):
            if p == q: continue
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

def compute_crowding_distance(objs: List[List[float]], idxs: List[int]) -> Dict[int, float]:
    M = len(objs[0]) if objs else 0
    Nf = len(idxs)
    if Nf == 0: return {}
    distance = {i: 0.0 for i in idxs}
    if Nf <= 2:
        for i in idxs: distance[i] = float('inf')
        return distance
    for m in range(M):
        vals = [objs[i][m] for i in idxs]
        order = [x for _, x in sorted(zip(vals, idxs))]
        vmin, vmax = vals[np.argmin(vals)], vals[np.argmax(vals)]
        if vmax == vmin:
            continue
        distance[order[0]] = float('inf')
        distance[order[-1]] = float('inf')
        for k in range(1, Nf - 1):
            i_prev, i_next = order[k - 1], order[k + 1]
            i_mid = order[k]
            gap = (objs[i_next][m] - objs[i_prev][m]) / (vmax - vmin)
            distance[i_mid] += gap
    return distance

def tournament_select_mo(pop: List['MDPNetwork'], rng: np.random.Generator, k: int,
                         ranks: List[int], crowding: Dict[int, float]) -> 'MDPNetwork':
    idxs = rng.choice(len(pop), size=k, replace=False)
    best = int(idxs[0])
    for j in idxs[1:]:
        j = int(j)
        if ranks[j] < ranks[best]:
            best = j
        elif ranks[j] == ranks[best] and crowding.get(j, 0.0) > crowding.get(best, 0.0):
            best = j
    return pop[best]

# -------------------------------
# Parallel scoring (multi-output aware + per-fn const)
# -------------------------------

def _flatten_objectives(val: Any) -> List[float]:
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    if isinstance(val, np.ndarray):
        return [float(x) for x in val.ravel().tolist()]
    return [float(val)]

def _score_worker_multi(payload: Tuple[Dict[str, Any], List[str], Tuple[Any, ...], Dict[str, Any], Optional[List[Dict[str, Any]]]]) -> List[float]:
    """
    payload = (mdp_portable, score_fn_names, args, kwargs, precomputed_portables)
    Returns objective vector (each fn may return float or sequence).
    """
    portable, fn_names, args, kwargs, precomputed_portables = payload
    mdp = MDPNetwork.from_portable(portable)

    # Base kwargs
    if precomputed_portables is not None and "precomputed_portables" not in kwargs:
        base_kwargs = dict(kwargs); base_kwargs["precomputed_portables"] = precomputed_portables
    else:
        base_kwargs = dict(kwargs)

    out: List[float] = []
    for name in fn_names:
        fn = get_registered_score_fn(name)
        val = fn(mdp, *args, **base_kwargs)  # constants now only via base_kwargs["score_const"]
        out.extend(_flatten_objectives(val))
    return out

def evaluate_mdp_objectives(
    mdps: List['MDPNetwork'],
    *,
    score_fn_names: List[str],
    n_workers: int,
    score_args: Optional[Tuple[Any, ...]] = None,
    score_kwargs: Optional[Dict[str, Any]] = None,
    precomputed_portables: Optional[List[Dict[str, Any]]] = None,
) -> List[List[float]]:
    if not score_fn_names:
        raise ValueError("score_fn_names must be a non-empty list (>=1).")
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1.")
    args = score_args or ()
    kwargs = score_kwargs or {}
    payloads = [(m.to_portable(), score_fn_names, args, kwargs, precomputed_portables) for m in mdps]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_score_worker_multi, payloads))
    return [list(map(float, r)) for r in results]

# -------------------------------
# Parallel offspring worker
# -------------------------------

def _apply_mutations(ind: 'MDPNetwork',
                     rng: np.random.Generator,
                     whitelist: Set[EdgeTriple],
                     cfg: GAConfig,
                     dist_mdp_ref: 'MDPNetwork'):
    """
    Apply mutations to 'ind'. All distance/scope calculations use 'dist_mdp_ref' (the base MDP).
    """
    # Build closures that reference the base MDP
    dist_fn = lambda _m, _s, _sp: _dist_from_cfg(dist_mdp_ref, _s, _sp, cfg)
    scope_fn = lambda _s: _allowed_nodes_within_scope(dist_mdp_ref, _s, cfg)

    for _ in range(cfg.add_edge_attempts_per_child):
        mutation_add_edge(ind, rng, dist_fn, cfg, scope_fn=scope_fn)

    mutation_prob_pairwise(ind, rng, cfg)

    if cfg.reward_tweak_edges_per_child > 0:
        mutation_reward_smallstep(ind, rng, cfg)

    if cfg.prune_prob_threshold is not None:
        prune_low_prob_transitions(ind, cfg.prune_prob_threshold)

def _child_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg: GAConfig = payload["cfg"]
    whitelist: Set[EdgeTriple] = set(payload["whitelist"])
    rng = np.random.default_rng(payload["seed"])
    def _mk(p: Dict[str, Any]) -> MDPNetwork: return MDPNetwork.from_portable(p)

    # Build base reference ONCE; all distances/scopes use this graph
    base_ref = _mk(payload["base_portable"])

    if "pa_portable" in payload:
        pa = _mk(payload["pa_portable"])
        pb = _mk(payload["pb_portable"])
        do_crossover: bool = payload["do_crossover"]
        child = crossover_action_block(pa, pb, rng) if do_crossover else (pa if rng.random() < 0.5 else pb).clone()
        _apply_mutations(child, rng, whitelist, cfg, dist_mdp_ref=base_ref)
        return child.to_portable()
    else:
        ind = _mk(payload["base_portable"])
        _apply_mutations(ind, rng, whitelist, cfg, dist_mdp_ref=base_ref)
        return ind.to_portable()

# -------------------------------
# GA main class (NSGA-II)
# -------------------------------

class MDPEvolutionGA:
    """NSGA-II on MDPNetwork with serial precompute; score fns may return multi-output.

    W&B usage (simplified):
      - You MUST pass `wb_run` from an external script; otherwise a ValueError will be raised.
      - We log only essential per-gen scalars and, if feasible, a per-gen scatter of the population.
      - For >=3 objectives, we log a compact per-gen table instead of plots.
      - At the end, a compact table of the final Pareto front is logged.
    """

    def __init__(self, base_mdp: 'MDPNetwork', cfg: GAConfig,
                 wb_run: Optional[Any] = None,
                 logger: Optional[logging.Logger] = None):
        if not cfg.score_fn_names or len(cfg.score_fn_names) < 1:
            raise ValueError("NSGA-II requires score_fn_names (>=1).")
        if cfg.n_workers < 1 or cfg.mutation_n_workers < 1:
            raise ValueError("n_workers and mutation_n_workers must be >= 1.")

        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.base_mdp = base_mdp.clone()
        self.whitelist: Set[EdgeTriple] = build_original_whitelist(self.base_mdp)
        self._base_portable = self.base_mdp.to_portable()

        self.precomputed_artifacts: Optional[List[Serialisable]] = None
        self._precomputed_portables: Optional[List[Dict[str, Any]]] = None

        # W&B & logging
        self.wb_run = wb_run if (cfg.wandb_enabled and wb_run is not None) else None
        if self.wb_run is None:
            raise ValueError("W&B run is required. Pass wb_run=wandb.init(...).")

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                _h = logging.StreamHandler(sys.stdout)
                _h.setFormatter(logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s"))
                self.logger.addHandler(_h)
            self.logger.setLevel(logging.INFO)

        # Define gen as the step metric for our logged series
        self.wb_run.define_metric("gen", summary="max")
        for prefix in ["pop", "init", "final", "time"]:
            self.wb_run.define_metric(f"{prefix}/*", step_metric="gen")

    # ------------- helpers --------------

    def _wb_log(self, data: Dict[str, Any]) -> None:
        """Direct W&B logging; relies on 'gen' field in data for step alignment."""
        self.wb_run.log(data)

    @staticmethod
    def _summ_stats(objs: List[List[float]]) -> Dict[str, float]:
        """Return flattened min/mean/max per objective as a dict."""
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

    @staticmethod
    def _build_pop_table(pop_objs: List[List[float]]) -> "wandb.Table":
        """Build a minimal table for the current population: [ind_idx, obj*]."""
        if not pop_objs:
            return wandb.Table(columns=["ind_idx"])
        M = len(pop_objs[0])
        columns = ["ind_idx"] + [f"obj{m}" for m in range(M)]
        table = wandb.Table(columns=columns)
        for i, vec in enumerate(pop_objs):
            row = [i] + [float(x) for x in vec]
            table.add_data(*row)
        return table

    # ------------- core GA -------------

    def _init_population(self) -> List['MDPNetwork']:
        pop: List[MDPNetwork] = [self.base_mdp.clone()]
        need = self.cfg.population_size - 1
        if need <= 0:
            return pop
        payloads: List[Dict[str, Any]] = []
        for _ in range(need):
            payloads.append({
                "cfg": self.cfg,
                "whitelist": list(self.whitelist),
                "seed": int(self.rng.integers(0, 2 ** 63 - 1)),
                "base_portable": self._base_portable,
            })
        with ProcessPoolExecutor(max_workers=self.cfg.mutation_n_workers) as ex:
            results = list(ex.map(_child_worker, payloads))
        for portable in results:
            pop.append(MDPNetwork.from_portable(portable))
        return pop

    def _make_children_parallel(self, parents_pairs: List[Tuple['MDPNetwork', 'MDPNetwork']]) -> List['MDPNetwork']:
        payloads: List[Dict[str, Any]] = []
        for (pa, pb) in parents_pairs:
            payloads.append({
                "cfg": self.cfg,
                "whitelist": list(self.whitelist),
                "seed": int(self.rng.integers(0, 2 ** 63 - 1)),
                "pa_portable": pa.to_portable(),
                "pb_portable": pb.to_portable(),
                "do_crossover": bool(self.rng.random() < self.cfg.crossover_rate),
                "base_portable": self._base_portable,
            })
        with ProcessPoolExecutor(max_workers=self.cfg.mutation_n_workers) as ex:
            results = list(ex.map(_child_worker, payloads))
        children: List[MDPNetwork] = []
        for portable in results:
            children.append(MDPNetwork.from_portable(portable))
        return children

    def _evaluate_population(self, pop: List['MDPNetwork']) -> List[List[float]]:
        # Default kwargs for score fns
        kw = dict(self.cfg.score_kwargs or {})
        kw.setdefault("vi_gamma", self.cfg.vi_gamma)
        kw.setdefault("vi_theta", self.cfg.vi_theta)
        kw.setdefault("vi_max_iterations", self.cfg.vi_max_iterations)
        kw.setdefault("policy_temperature", self.cfg.policy_temperature)
        kw.setdefault("policy_mixing", self.cfg.policy_mixing)
        kw.setdefault("policy_tie_tol", self.cfg.policy_tie_tol)
        kw.setdefault("perf_numpoints", self.cfg.perf_numpoints)
        kw.setdefault("perf_gamma", self.cfg.perf_gamma if self.cfg.perf_gamma is not None else self.cfg.vi_gamma)
        kw.setdefault("perf_theta", self.cfg.perf_theta if self.cfg.perf_theta is not None else self.cfg.vi_theta)
        kw.setdefault("perf_max_iterations", self.cfg.perf_max_iterations if self.cfg.perf_max_iterations is not None else self.cfg.vi_max_iterations)

        return evaluate_mdp_objectives(
            mdps=pop,
            score_fn_names=self.cfg.score_fn_names or [],
            n_workers=self.cfg.n_workers,
            score_args=self.cfg.score_args,
            score_kwargs=kw,
            precomputed_portables=self._precomputed_portables,
        )

    def run(self) -> Tuple[List['MDPNetwork'], List[List[float]], List['MDPNetwork'], List[List[float]]]:
        """
        Execute NSGA-II search.
        Returns:
            pareto_mdps: final non-dominated front (F1)
            pareto_objs: objective vectors for pareto_mdps
            pop:         final population
            pop_objs:    objective vectors for final population
        """
        # ---------------- Precompute baseline ----------------
        t0 = time.perf_counter()
        if self.precomputed_artifacts is not None:
            self._precomputed_portables = [obj.to_portable() for obj in self.precomputed_artifacts]
            self.logger.info(f"[Precompute] Using provided artifacts: {len(self._precomputed_portables)} item(s).")
            self._wb_log({"gen": -1, "precompute/ok": 1})
        else:
            V, Q = optimal_value_iteration(
                self.base_mdp,
                gamma=self.cfg.vi_gamma,
                theta=self.cfg.vi_theta,
                max_iterations=self.cfg.vi_max_iterations
            )
            base_policy: PolicyTable = q_table_to_policy(
                Q,
                states=list(self.base_mdp.states),
                num_actions=self.base_mdp.num_actions,
                mixing=self.cfg.policy_mixing,
                temperature=self.cfg.policy_temperature,
                tie_tol=self.cfg.policy_tie_tol,
            )
            base_occupancy: ValueTable = compute_occupancy_measure(
                self.base_mdp, base_policy,
                gamma=self.cfg.vi_gamma,
                theta=self.cfg.vi_theta,
                max_iterations=self.cfg.vi_max_iterations
            )
            self._precomputed_portables = [base_policy.to_portable(), base_occupancy.to_portable()]
            self.logger.info("[Precompute] Built baseline policy+occupancy and broadcast to workers.")
            self._wb_log({"gen": -1, "precompute/ok": 1})

        t1 = time.perf_counter()
        self._wb_log({"gen": -1, "precompute/time_sec": float(t1 - t0)})

        # ---------------- Init population ----------------
        pop = self._init_population()
        objs = self._evaluate_population(pop)

        # Initial summary for console + W&B
        init_stats = self._summ_stats(objs)
        msg_stats = " | ".join([f"obj{m}: min={init_stats.get(f'min_{m}', float('nan')):.4f} "
                                f"mean={init_stats.get(f'mean_{m}', float('nan')):.4f} "
                                f"max={init_stats.get(f'max_{m}', float('nan')):.4f}"
                                for m in range(len(objs[0]) if objs else 0)])
        self.logger.info(f"[Init] pop={len(pop)} | {msg_stats if msg_stats else 'NA'}")
        self._wb_log({
            "gen": 0,
            "init/pop_size": int(len(pop)),
            **{f"init/obj{m}_min": init_stats.get(f"min_{m}", float("nan")) for m in
               range(len(objs[0]) if objs else 0)},
            **{f"init/obj{m}_mean": init_stats.get(f"mean_{m}", float("nan")) for m in
               range(len(objs[0]) if objs else 0)},
            **{f"init/obj{m}_max": init_stats.get(f"max_{m}", float("nan")) for m in
               range(len(objs[0]) if objs else 0)},
        })

        # Ranks/crowding for init
        fronts = fast_non_dominated_sort(objs)
        ranks = [0] * len(pop)
        for r, F in enumerate(fronts):
            for i in F:
                ranks[i] = r
        crowding: Dict[int, float] = {}
        for F in fronts:
            crowding.update(compute_crowding_distance(objs, F))

        # ---------- Per-gen viz at gen 0 (ALWAYS save table; plot depends on M) ----------
        M = len(objs[0]) if objs else 0
        pop_tbl = self._build_pop_table(objs)
        self._wb_log({"gen": 0, "tables/pop": pop_tbl})  # always log table
        if M == 1:
            # Bar chart: index vs obj0
            bar = wandb.plot.bar(pop_tbl, "ind_idx", "obj0", title="Gen 0: obj0 (bar)")
            self._wb_log({"gen": 0, "plots/pop_bar": bar})
        elif M == 2:
            # Scatter: obj0 vs obj1
            scatter = wandb.plot.scatter(pop_tbl, "obj0", "obj1", title="Gen 0: obj0 vs obj1")
            self._wb_log({"gen": 0, "plots/pop_scatter": scatter})
        # M >= 3 -> table only

        # ---------------- Generations loop ----------------
        for gen in range(self.cfg.generations):
            gstart = time.perf_counter()

            # Parent selection
            parents_pairs: List[Tuple[MDPNetwork, MDPNetwork]] = []
            for _ in range(self.cfg.population_size):
                p1 = tournament_select_mo(pop, self.rng, self.cfg.tournament_k, ranks, crowding)
                p2 = tournament_select_mo(pop, self.rng, self.cfg.tournament_k, ranks, crowding)
                parents_pairs.append((p1, p2))
            t_sel = time.perf_counter()

            # Offspring generation
            children = self._make_children_parallel(parents_pairs)
            t_child = time.perf_counter()

            # Evaluate children
            child_objs = self._evaluate_population(children)
            t_eval = time.perf_counter()

            # Environmental selection (union -> next population)
            union_pop = pop + children
            union_objs = objs + child_objs

            union_fronts = fast_non_dominated_sort(union_objs)
            new_pop: List[MDPNetwork] = []
            new_objs: List[List[float]] = []

            for F in union_fronts:
                if len(new_pop) + len(F) <= self.cfg.population_size:
                    new_pop.extend([union_pop[i] for i in F])
                    new_objs.extend([union_objs[i] for i in F])
                else:
                    dist = compute_crowding_distance(union_objs, F)
                    sorted_F = sorted(F, key=lambda i: dist.get(i, 0.0), reverse=True)
                    remain = self.cfg.population_size - len(new_pop)
                    chosen = sorted_F[:remain]
                    new_pop.extend([union_pop[i] for i in chosen])
                    new_objs.extend([union_objs[i] for i in chosen])
                    break

            pop, objs = new_pop, new_objs

            # Recompute ranks & crowding for current population
            fronts = fast_non_dominated_sort(objs)
            ranks = [0] * len(pop)
            for r, F in enumerate(fronts):
                for i in F:
                    ranks[i] = r
            crowding = {}
            for F in fronts:
                crowding.update(compute_crowding_distance(objs, F))

            # Console logging
            gen_stats = self._summ_stats(objs)
            msg_stats = " | ".join([f"obj{m}: min={gen_stats.get(f'min_{m}', float('nan')):.4f} "
                                    f"mean={gen_stats.get(f'mean_{m}', float('nan')):.4f} "
                                    f"max={gen_stats.get(f'max_{m}', float('nan')):.4f}"
                                    for m in range(len(objs[0]) if objs else 0)])
            self.logger.info(
                f"[Gen {gen + 1}/{self.cfg.generations}] pop={len(pop)} | {msg_stats if msg_stats else 'NA'} | F1={len(fronts[0]) if fronts else 0}")

            # W&B scalars (use 'gen' as the step axis)
            gend = time.perf_counter()
            self._wb_log({
                "gen": gen + 1,
                "pop/size": int(len(pop)),
                "pop/F1_size": int(len(fronts[0]) if fronts else 0),
                **{f"pop/obj{m}_min": gen_stats.get(f"min_{m}", float("nan")) for m in
                   range(len(objs[0]) if objs else 0)},
                **{f"pop/obj{m}_mean": gen_stats.get(f"mean_{m}", float("nan")) for m in
                   range(len(objs[0]) if objs else 0)},
                **{f"pop/obj{m}_max": gen_stats.get(f"max_{m}", float("nan")) for m in
                   range(len(objs[0]) if objs else 0)},
                "time/selection_sec": float(t_child - t_sel),
                "time/offspring_sec": float(t_eval - t_child),
                "time/eval_sec": float(gend - t_eval),
                "time/total_gen_sec": float(gend - gstart),
            })

            # ---------- Per-gen viz at gen+1 (ALWAYS save table; plot depends on M) ----------
            M = len(objs[0]) if objs else 0
            pop_tbl = self._build_pop_table(objs)
            self._wb_log({"gen": gen + 1, "tables/pop": pop_tbl})  # always log table
            if M == 1:
                bar = wandb.plot.bar(pop_tbl, "ind_idx", "obj0", title=f"Gen {gen + 1}: obj0 (bar)")
                self._wb_log({"gen": gen + 1, "plots/pop_bar": bar})
            elif M == 2:
                scatter = wandb.plot.scatter(pop_tbl, "obj0", "obj1", title=f"Gen {gen + 1}: obj0 vs obj1")
                self._wb_log({"gen": gen + 1, "plots/pop_scatter": scatter})
            # M >= 3 -> table only

        # ---------------- Final Pareto front ----------------
        final_fronts = fast_non_dominated_sort(objs)
        F1 = final_fronts[0] if final_fronts else list(range(len(pop)))
        pareto_mdps = [pop[i].clone() for i in F1]
        pareto_objs = [objs[i][:] for i in F1]

        # Final summary logging
        final_stats = self._summ_stats([objs[i] for i in F1] if F1 else objs)
        self._wb_log({
            "gen": self.cfg.generations,
            "final/F1_size": int(len(F1)),
            **{f"final/obj{m}_min": final_stats.get(f"min_{m}", float("nan")) for m in
               range(len(objs[0]) if objs else 0)},
            **{f"final/obj{m}_mean": final_stats.get(f"mean_{m}", float("nan")) for m in
               range(len(objs[0]) if objs else 0)},
            **{f"final/obj{m}_max": final_stats.get(f"max_{m}", float("nan")) for m in
               range(len(objs[0]) if objs else 0)},
        })

        # Final compact table of F1 (no figures)
        if F1:
            M = len(objs[0])
            columns = ["idx"] + [f"obj{m}" for m in range(M)]
            f1_tbl = wandb.Table(columns=columns)
            for i in F1:
                f1_tbl.add_data(int(i), *[float(x) for x in objs[i]])
            self._wb_log({"gen": self.cfg.generations, "tables/final_pareto": f1_tbl})

        return pareto_mdps, pareto_objs, pop, objs

# -------------------------------
# Example multi-output objectives
# -------------------------------

def obj_multi_kl_and_perf(mdp: 'MDPNetwork', *args, **kwargs) -> List[float]:
    """
    Return [ -KL(baseline || current), performance_integral ].
    Shared VI -> policy -> occupancy for current MDP.
    """
    const = kwargs.get("score_const", {}) or {}
    gamma = float(kwargs.get("vi_gamma", 0.99))
    theta = float(kwargs.get("vi_theta", 1e-6))
    max_iter = int(kwargs.get("vi_max_iterations", 1000))
    temperature = float(kwargs.get("policy_temperature", 1.0))
    mixing = tuple(const.get("policy_mixing", kwargs.get("policy_mixing", (0.0, 1.0, 0.0))))
    tie_tol = float(const.get("policy_tie_tol", kwargs.get("policy_tie_tol", 1e-6)))
    delta = float(const.get("kl_delta", kwargs.get("kl_delta", 1e-3)))

    pgamma = float(kwargs.get("perf_gamma", gamma))
    ptheta = float(kwargs.get("perf_theta", theta))
    pmax_iter = int(kwargs.get("perf_max_iterations", max_iter))
    numpoints = int(kwargs.get("perf_numpoints", 100))

    # Baseline
    base_policy = None
    base_occupancy = None
    _pre = kwargs.get("precomputed_portables", None)
    if _pre and len(_pre) >= 2:
        base_policy = PolicyTable.from_portable(_pre[0])
        base_occupancy = ValueTable.from_portable(_pre[1])

    # Solve current
    _, Q2 = optimal_value_iteration(mdp, gamma=gamma, theta=theta, max_iterations=max_iter)
    policy2: PolicyTable = q_table_to_policy(
        Q2,
        states=list(mdp.states),
        num_actions=mdp.num_actions,
        mixing=mixing,
        temperature=temperature,
        tie_tol=tie_tol,
    )
    occupancy2: ValueTable = compute_occupancy_measure(mdp, policy=policy2,
                                                       gamma=gamma, theta=theta, max_iterations=max_iter)

    # Objective 1: -KL
    if base_policy is not None and base_occupancy is not None:
        kl = kl_policies(policy1=base_policy, occupancy1=base_occupancy,
                         policy2=policy2, occupancy2=occupancy2, delta=delta)
        obj1 = -float(kl)
    else:
        obj1 = 0.0

    # Objective 2: performance integral (random prior -> current)
    prior_policy: PolicyTable = create_random_policy(mdp)
    _curve, integral = performance_curve_and_integral(
        prior_policy=prior_policy,
        target_policy=policy2,
        mdp_network=mdp,
        numpoints=numpoints,
        gamma=pgamma,
        theta=ptheta,
        max_iterations=pmax_iter,
    )
    obj2 = float(integral)

    return [obj1, obj2]

def obj_multi_perf(mdp: 'MDPNetwork', *args, **kwargs) -> List[float]:
    """
    Example that uses per-function constants via kwargs["score_const"].
    Expects const like {"blend_weight": 0.8}.
    Returns two integrals:
      1) integral(prior = blend(policy2, random, w), target = baseline_policy)
      2) integral(prior = random, target = policy2)
    """
    const = kwargs.get("score_const", {}) or {}
    blend_w = float(const.get("blend_weight", kwargs.get("blend_weight", 0.8)))

    gamma = float(kwargs.get("vi_gamma", 0.99))
    theta = float(kwargs.get("vi_theta", 1e-6))
    max_iter = int(kwargs.get("vi_max_iterations", 1000))
    temperature = float(kwargs.get("policy_temperature", 1.0))
    mixing = tuple(const.get("policy_mixing", kwargs.get("policy_mixing", (0.0, 1.0, 0.0))))
    tie_tol = float(const.get("policy_tie_tol", kwargs.get("policy_tie_tol", 1e-6)))

    pgamma = float(kwargs.get("perf_gamma", gamma))
    ptheta = float(kwargs.get("perf_theta", theta))
    pmax_iter = int(kwargs.get("perf_max_iterations", max_iter))
    numpoints = int(kwargs.get("perf_numpoints", 100))

    _pre = kwargs.get("precomputed_portables", None)
    base_policy = PolicyTable.from_portable(_pre[0]) if (_pre and len(_pre) >= 1) else None

    _, Q2 = optimal_value_iteration(mdp, gamma=gamma, theta=theta, max_iterations=max_iter)
    policy2: PolicyTable = q_table_to_policy(
        Q2,
        states=list(mdp.states),
        num_actions=mdp.num_actions,
        mixing=mixing,
        temperature=temperature,
        tie_tol=tie_tol,
    )

    # Integral 1: blended prior -> baseline target
    blended_policy2 = blend_policies(policy2, create_random_policy(mdp), weight=blend_w)
    if base_policy is not None:
        _curve, integral1 = performance_curve_and_integral(
            prior_policy=blended_policy2,
            target_policy=base_policy,
            mdp_network=mdp,
            numpoints=numpoints,
            gamma=pgamma,
            theta=ptheta,
            max_iterations=pmax_iter,
        )
    else:
        integral1 = 0.0

    # Integral 2: random prior -> current target
    prior_policy: PolicyTable = create_random_policy(mdp)
    _curve, integral2 = performance_curve_and_integral(
        prior_policy=prior_policy,
        target_policy=policy2,
        mdp_network=mdp,
        numpoints=numpoints,
        gamma=pgamma,
        theta=ptheta,
        max_iterations=pmax_iter,
    )
    return [float(integral1), float(integral2)]


register_score_fn("obj_multi_kl_and_perf", obj_multi_kl_and_perf)
register_score_fn("obj_multi_perf", obj_multi_perf)
