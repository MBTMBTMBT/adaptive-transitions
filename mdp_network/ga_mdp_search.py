# ga_mdp_search.py
# Genetic search system for MDPNetwork (structure + probability + reward)
# with optional parallel scoring and parallel offspring mutation using a process pool (Linux, Python 3.10).
#
# NSGA-II multi-objective version + simple serial precompute:
# - Precompute on the base MDP: Value Iteration -> Q -> softmax Policy (temperature) + Occupancy
# - Broadcast those artifacts to scoring workers in kwargs["precomputed_portables"]
# - Two objective templates:
#     1) obj_policy_kl_similarity: -KL between (precomputed policy, occupancy) and (current policy, occupancy)
#     2) obj_perf_integral:         performance curve integral using random prior vs target policy (current MDP)
#
# Minimal-invasive changes: keep your process model and APIs; add a few GAConfig fields.

from __future__ import annotations

from typing import Callable, Dict, Tuple, List, Optional, Set, Any
from dataclasses import dataclass
import math
import os

import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor

from mdp_network.mdp_tables import q_table_to_policy, PolicyTable, ValueTable, create_random_policy
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
ScoreFn = Callable[['MDPNetwork', Any], float]
DistanceFn = Callable[['MDPNetwork', State, State], float]

# -------------------------------
# Score function registry
# -------------------------------

SCORE_FN_REGISTRY: Dict[str, ScoreFn] = {}

def register_score_fn(name: str, fn: ScoreFn) -> None:
    SCORE_FN_REGISTRY[name] = fn

def get_registered_score_fn(name: str) -> ScoreFn:
    if name not in SCORE_FN_REGISTRY:
        raise KeyError(f"Score function '{name}' is not registered.")
    return SCORE_FN_REGISTRY[name]

# -------------------------------
# Single distance function
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
    """Dijkstra on a directed graph with edge weights w(u->v)=1-max_a P(v|u,a)."""
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
    # cap
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
            w = max(weight_eps, 1.0 - pmax)
            nd = du + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return float(unreachable)

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

    # Distance parameters
    dist_max_hops: Optional[int] = None
    dist_node_cap: Optional[int] = None
    dist_weight_eps: float = 1e-9
    dist_unreachable: float = 1e6

    # Randomness
    seed: Optional[int] = None

    # --------- NEW: algorithms/score parameters (with sensible defaults) ---------
    # Value-iteration (also used for occupancy unless perf_* overrides)
    vi_gamma: float = 0.99
    vi_theta: float = 1e-6
    vi_max_iterations: int = 1000

    # Softmax policy temperature for q_table_to_policy
    policy_temperature: float = 1.0

    # KL small constant
    kl_delta: float = 1e-3

    # Performance curve & integral params
    perf_numpoints: int = 100
    perf_gamma: Optional[float] = None
    perf_theta: Optional[float] = None
    perf_max_iterations: Optional[int] = None

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

def set_outgoing_for_action(mdp: 'MDPNetwork',
                            s: State,
                            a: Action,
                            new_map: Dict[State, Tuple[float, float]]):
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
                      cfg: GAConfig):
    candidates_sa = [(s, a) for (s, a) in list_all_action_pairs(mdp)
                     if action_out_degree(mdp, s, a) < cfg.max_out_degree]
    if not candidates_sa:
        return
    s, a = candidates_sa[rng.integers(0, len(candidates_sa))]
    existing = set(get_outgoing_for_action(mdp, s, a).keys())
    sp_candidates = [sp for sp in mdp.states
                     if (cfg.allow_self_loops or sp != s)
                     and sp not in existing]
    if not sp_candidates:
        return
    weights = []
    for sp in sp_candidates:
        d = _dist_from_cfg(mdp, s, sp, cfg)
        w = math.exp(-cfg.gamma_sample * d) if cfg.gamma_sample > 0.0 else 1.0
        weights.append(w)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0.0:
        return
    weights /= weights.sum()
    sp_new = int(rng.choice(sp_candidates, p=weights))
    d_new = _dist_from_cfg(mdp, s, sp_new, cfg)
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
    """Remove ALL transitions with prob < threshold; renormalize residuals."""
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

def mutation_prob_pairwise(mdp: 'MDPNetwork',
                           rng: np.random.Generator,
                           cfg: GAConfig):
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

def mutation_reward_smallstep(mdp: 'MDPNetwork',
                              rng: np.random.Generator,
                              cfg: GAConfig):
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

def crossover_action_block(parent_a: 'MDPNetwork',
                           parent_b: 'MDPNetwork',
                           rng: np.random.Generator) -> 'MDPNetwork':
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

def tournament_select_mo(pop: List['MDPNetwork'],
                         rng: np.random.Generator,
                         k: int,
                         ranks: List[int],
                         crowding: Dict[int, float]) -> 'MDPNetwork':
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
# Parallel scoring (multi-objective)
# -------------------------------

def _score_worker_multi(payload: Tuple[Dict[str, Any], List[str], Tuple[Any, ...], Dict[str, Any], Optional[List[Dict[str, Any]]]]) -> List[float]:
    portable, fn_names, args, kwargs, precomputed_portables = payload
    mdp = MDPNetwork.from_portable(portable)
    if precomputed_portables is not None and "precomputed_portables" not in kwargs:
        local_kwargs = dict(kwargs); local_kwargs["precomputed_portables"] = precomputed_portables
    else:
        local_kwargs = kwargs
    vals: List[float] = []
    for name in fn_names:
        fn = get_registered_score_fn(name)
        vals.append(float(fn(mdp, *args, **local_kwargs)))
    return vals

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

def _apply_mutations(ind: 'MDPNetwork', rng: np.random.Generator, whitelist: Set[EdgeTriple], cfg: GAConfig):
    for _ in range(cfg.add_edge_attempts_per_child):
        mutation_add_edge(ind, rng, lambda _m, _s, _sp: _dist_from_cfg(ind, _s, _sp, cfg), cfg)
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

    if "pa_portable" in payload:
        pa = _mk(payload["pa_portable"])
        pb = _mk(payload["pb_portable"])
        do_crossover: bool = payload["do_crossover"]
        child = crossover_action_block(pa, pb, rng) if do_crossover else (pa if rng.random() < 0.5 else pb).clone()
        _apply_mutations(child, rng, whitelist, cfg)
        return child.to_portable()
    else:
        ind = _mk(payload["base_portable"])
        _apply_mutations(ind, rng, whitelist, cfg)
        return ind.to_portable()

# -------------------------------
# GA main class (NSGA-II)
# -------------------------------

class MDPEvolutionGA:
    """
    NSGA-II on MDPNetwork with optional serial precompute (policy+occupancy) broadcast to workers.
    """

    def __init__(self, base_mdp: 'MDPNetwork', cfg: GAConfig):
        if not cfg.score_fn_names or len(cfg.score_fn_names) < 1:
            raise ValueError("NSGA-II requires score_fn_names (>=1).")
        if cfg.n_workers < 1 or cfg.mutation_n_workers < 1:
            raise ValueError("n_workers and mutation_n_workers must be >= 1.")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.base_mdp = base_mdp.clone()
        self.whitelist: Set[EdgeTriple] = build_original_whitelist(self.base_mdp)
        self._base_portable = self.base_mdp.to_portable()

        # Precompute slots
        self.precomputed_artifacts: Optional[List[Serialisable]] = None
        self._precomputed_portables: Optional[List[Dict[str, Any]]] = None

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
            })
        with ProcessPoolExecutor(max_workers=self.cfg.mutation_n_workers) as ex:
            results = list(ex.map(_child_worker, payloads))
        children: List[MDPNetwork] = []
        for portable in results:
            children.append(MDPNetwork.from_portable(portable))
        return children

    def _evaluate_population(self, pop: List['MDPNetwork']) -> List[List[float]]:
        # Inject default kwargs from config so users don't have to repeat them manually.
        kw = dict(self.cfg.score_kwargs or {})
        kw.setdefault("vi_gamma", self.cfg.vi_gamma)
        kw.setdefault("vi_theta", self.cfg.vi_theta)
        kw.setdefault("vi_max_iterations", self.cfg.vi_max_iterations)
        kw.setdefault("policy_temperature", self.cfg.policy_temperature)
        kw.setdefault("kl_delta", self.cfg.kl_delta)
        kw.setdefault("perf_numpoints", self.cfg.perf_numpoints)
        # perf_* fall back to vi_* if not explicitly set
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
        Returns:
            pareto_mdps:  List[MDPNetwork]  -- final non-dominated front (F1)
            pareto_objs:  List[List[float]] -- objective vectors aligned with pareto_mdps
            pop:          List[MDPNetwork]  -- final whole population
            pop_objs:     List[List[float]] -- objective vectors for the final population
        """
        # ----- PRECOMPUTE (serial, once) -----
        if self.precomputed_artifacts is not None:
            self._precomputed_portables = [obj.to_portable() for obj in self.precomputed_artifacts]
            print(f"[Precompute] Using provided artifacts: {len(self._precomputed_portables)} item(s).")
        else:
            V, Q = optimal_value_iteration(
                self.base_mdp,
                gamma=self.cfg.vi_gamma,
                theta=self.cfg.vi_theta,
                max_iterations=self.cfg.vi_max_iterations
            )
            base_policy: PolicyTable = q_table_to_policy(
                Q, states=list(self.base_mdp.states),
                num_actions=self.base_mdp.num_actions,
                temperature=self.cfg.policy_temperature
            )
            base_occupancy: ValueTable = compute_occupancy_measure(
                self.base_mdp, base_policy,
                gamma=self.cfg.vi_gamma,
                theta=self.cfg.vi_theta,
                max_iterations=self.cfg.vi_max_iterations
            )
            self._precomputed_portables = [base_policy.to_portable(), base_occupancy.to_portable()]
            print("[Precompute] Built baseline policy+occupancy and broadcast to workers.")

        # ----- Init -----
        pop = self._init_population()
        objs = self._evaluate_population(pop)
        M = len(objs[0]) if objs else 0

        def _summ(o: List[List[float]]) -> str:
            if not o: return "NA"
            arr = np.array(o, dtype=float)
            stats = []
            for m in range(arr.shape[1]):
                stats.append(f"obj{m}: min={arr[:,m].min():.4f} mean={arr[:,m].mean():.4f} max={arr[:,m].max():.4f}")
            return " | ".join(stats)

        print(f"[Init] pop={len(pop)} | { _summ(objs) }")

        # Prepare selection metrics
        fronts = fast_non_dominated_sort(objs)
        ranks = [0] * len(pop)
        for r, F in enumerate(fronts):
            for i in F:
                ranks[i] = r
        crowding: Dict[int, float] = {}
        for F in fronts:
            crowding.update(compute_crowding_distance(objs, F))

        # ----- Generations -----
        for gen in range(self.cfg.generations):
            target = self.cfg.population_size
            parents_pairs: List[Tuple[MDPNetwork, MDPNetwork]] = []
            for _ in range(target):
                p1 = tournament_select_mo(pop, self.rng, self.cfg.tournament_k, ranks, crowding)
                p2 = tournament_select_mo(pop, self.rng, self.cfg.tournament_k, ranks, crowding)
                parents_pairs.append((p1, p2))

            children = self._make_children_parallel(parents_pairs)
            child_objs = self._evaluate_population(children)

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

            fronts = fast_non_dominated_sort(objs)
            ranks = [0] * len(pop)
            for r, F in enumerate(fronts):
                for i in F:
                    ranks[i] = r
            crowding = {}
            for F in fronts:
                crowding.update(compute_crowding_distance(objs, F))

            print(f"[Gen {gen + 1}/{self.cfg.generations}] pop={len(pop)} | { _summ(objs) } | F1={len(fronts[0])}")

        # Final Pareto front
        final_fronts = fast_non_dominated_sort(objs)
        F1 = final_fronts[0] if final_fronts else list(range(len(pop)))
        pareto_mdps = [pop[i].clone() for i in F1]
        pareto_objs = [objs[i][:] for i in F1]
        return pareto_mdps, pareto_objs, pop, objs

# -------------------------------
# Objective functions required by your spec
# -------------------------------

def obj_policy_kl_similarity(mdp: 'MDPNetwork', *args, **kwargs) -> float:
    """
    Maximize (-KL) between the baseline (precomputed) and the current MDP:
    - Baseline: policy1, occupancy1 come from kwargs["precomputed_portables"]
    - Current:  run VI -> Q -> softmax policy2 (temperature), then occupancy2
    Returns: negative KL (so that larger is better => more similar).
    """
    # Read algorithm params
    gamma = float(kwargs.get("vi_gamma", 0.99))
    theta = float(kwargs.get("vi_theta", 1e-6))
    max_iter = int(kwargs.get("vi_max_iterations", 1000))
    temperature = float(kwargs.get("policy_temperature", 1.0))
    delta = float(kwargs.get("kl_delta", 1e-3))

    # Rebuild baseline artifacts
    _pre = kwargs.get("precomputed_portables", None)
    base_policy = None
    base_occupancy = None
    if _pre and len(_pre) >= 2:
        base_policy = PolicyTable.from_portable(_pre[0])
        base_occupancy = ValueTable.from_portable(_pre[1])

    # Current MDP solve
    V2, Q2 = optimal_value_iteration(mdp, gamma=gamma, theta=theta, max_iterations=max_iter)
    policy2: PolicyTable = q_table_to_policy(Q2, states=list(mdp.states),
                                             num_actions=mdp.num_actions,
                                             temperature=temperature)
    occupancy2: ValueTable = compute_occupancy_measure(mdp, policy2, gamma=gamma, theta=theta, max_iterations=max_iter)

    # If baseline missing (shouldn't happen if GA.run() did precompute), gracefully fallback to 0 loss
    if base_policy is None or base_occupancy is None:
        return 0.0

    kl = kl_policies(
        policy1=base_policy,
        occupancy1=base_occupancy,
        policy2=policy2,
        occupancy2=occupancy2,
        delta=delta,
    )
    return -float(kl)  # maximize similarity

def obj_perf_integral(mdp: 'MDPNetwork', *args, **kwargs) -> float:
    """
    Maximize the integral under performance curve:
    - Target policy: VI -> Q -> softmax on the CURRENT MDP.
    - Prior policy:  random policy on the CURRENT MDP.
    """
    # Target via VI
    vi_gamma = float(kwargs.get("vi_gamma", 0.99))
    vi_theta = float(kwargs.get("vi_theta", 1e-6))
    vi_max_iter = int(kwargs.get("vi_max_iterations", 1000))
    temperature = float(kwargs.get("policy_temperature", 1.0))

    # Perf params (fallback to VI params)
    pgamma = float(kwargs.get("perf_gamma", vi_gamma))
    ptheta = float(kwargs.get("perf_theta", vi_theta))
    pmax_iter = int(kwargs.get("perf_max_iterations", vi_max_iter))
    numpoints = int(kwargs.get("perf_numpoints", 100))

    V, Q = optimal_value_iteration(mdp, gamma=vi_gamma, theta=vi_theta, max_iterations=vi_max_iter)
    target_policy: PolicyTable = q_table_to_policy(Q, states=list(mdp.states),
                                                   num_actions=mdp.num_actions,
                                                   temperature=temperature)
    prior_policy: PolicyTable = create_random_policy(mdp)

    _curve, integral = performance_curve_and_integral(
        prior_policy=prior_policy,
        target_policy=target_policy,
        mdp_network=mdp,
        numpoints=numpoints,
        gamma=pgamma,
        theta=ptheta,
        max_iterations=pmax_iter,
    )
    return float(integral)

# Register the two objectives
register_score_fn("obj_policy_kl_similarity", obj_policy_kl_similarity)
register_score_fn("obj_perf_integral", obj_perf_integral)

# -------------------------------
# Tiny demo
# -------------------------------

if __name__ == "__main__":
    # Mini MDP demo
    nS, nA = 5, 2
    cfg_demo: Dict[str, Any] = {
        "num_actions": nA,
        "states": list(range(nS)),
        "start_states": [0],
        "terminal_states": [4],
        "default_reward": 0.0,
        "transitions": {
            "0": {"0": {"1": {"p": 0.8, "r": 0.0}, "0": {"p": 0.2, "r": 0.0}}, "1": {"2": {"p": 1.0, "r": 0.0}}},
            "1": {"0": {"2": {"p": 0.7, "r": 0.0}, "3": {"p": 0.3, "r": 0.0}}, "1": {"1": {"p": 1.0, "r": 0.0}}},
            "2": {"0": {"3": {"p": 1.0, "r": 0.0}}, "1": {"2": {"p": 1.0, "r": 0.0}}},
            "3": {"0": {"4": {"p": 0.9, "r": 1.0}, "3": {"p": 0.1, "r": 0.0}}, "1": {"1": {"p": 1.0, "r": 0.0}}},
        },
    }
    base_mdp = MDPNetwork(config_data=cfg_demo)

    # NSGA-II with your two objectives
    cfg = GAConfig(
        population_size=64,
        generations=8,
        tournament_k=2,
        crossover_rate=1.0,
        allow_self_loops=True,
        min_out_degree=1,
        max_out_degree=6,
        prob_floor=1e-6,
        add_edge_attempts_per_child=1,
        prune_prob_threshold=1e-3,
        prob_tweak_actions_per_child=8,
        prob_pairwise_step=0.02,
        reward_tweak_edges_per_child=8,
        reward_k_percent=0.02,
        reward_ref_floor=1e-3,

        # Parallel
        n_workers=max(1, os.cpu_count() or 1),
        score_fn_names=["obj_policy_kl_similarity", "obj_perf_integral"],
        mutation_n_workers=max(1, os.cpu_count() or 1),

        # VI / policy / KL / perf params
        vi_gamma=0.99,
        vi_theta=1e-6,
        vi_max_iterations=1000,
        policy_temperature=1.0,
        kl_delta=1e-3,
        perf_numpoints=64,
        perf_gamma=None,          # None -> fall back to vi_gamma
        perf_theta=None,          # None -> fall back to vi_theta
        perf_max_iterations=None, # None -> fall back to vi_max_iterations

        seed=123,
    )

    ga = MDPEvolutionGA(base_mdp=base_mdp, cfg=cfg)
    pareto_mdps, pareto_objs, pop, pop_objs = ga.run()

    print("\n=== NSGA-II finished ===")
    print(f"Pareto front size = {len(pareto_mdps)}")
    for i, o in enumerate(pareto_objs[:10]):
        print(f"  PF[{i}] objs = {list(map(lambda x: round(x, 6), o))}")
    print(f"Final population size = {len(pop)}")
