# ga_mdp_search.py
# Genetic search system for MDPNetwork (structure + probability + reward) with
# optional parallel scoring and parallel offspring mutation using a process pool (Linux, Python 3.10).

from __future__ import annotations

from typing import Callable, Dict, Tuple, List, Optional, Set, Any
from dataclasses import dataclass
import math
import os

import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor

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
# Single distance function (fixed signature via GAConfig)
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
    """
    Directed distance using edge weights w(u->v) = 1 - max_a P(v | u, a).
    Search is restricted to a subgraph reachable from s within `max_hops` hops (unweighted).
    If `node_cap` is set, cap total expanded nodes. Outside subgraph -> `unreachable`.
    """
    if s == sp:
        return 0.0

    G = mdp.graph  # directed

    # 1) Build allowed node set via BFS (unweighted) within max_hops
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

    # Optional hard cap
    if node_cap is not None and len(allowed) > node_cap:
        if 'hop_dist' in locals():
            kept = sorted(allowed, key=lambda x: hop_dist.get(x, 10**9))[:node_cap]
            allowed = set(kept)
        else:
            allowed = set(list(allowed)[:node_cap])

    if sp not in allowed:
        return float(unreachable)

    # 2) Dijkstra within allowed subgraph; edge weight = 1 - max_a p(u->v|a)
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
    # Population and evolution
    population_size: int = 80
    generations: int = 100
    tournament_k: int = 2
    elitism_num: int = 8
    crossover_rate: float = 1.0

    # Structural constraints
    allow_self_loops: bool = True
    min_out_degree: int = 1
    max_out_degree: int = 8
    prob_floor: float = 1e-6

    # Add-edge parameters
    add_edge_attempts_per_child: int = 2
    epsilon_new_prob: float = 0.02
    gamma_sample: float = 1.0
    gamma_prob: float = 0.0

    # Delete-edge parameters (prune by threshold; no stochastic deletion anymore)
    prune_prob_threshold: Optional[float] = None

    # Probability tweak parameters
    prob_tweak_actions_per_child: int = 20
    prob_pairwise_step: float = 0.02

    # Reward tweak parameters
    reward_tweak_edges_per_child: int = 50
    reward_k_percent: float = 0.02
    reward_ref_floor: float = 1e-3
    reward_min: Optional[float] = None
    reward_max: Optional[float] = None

    # Parallel evaluation (scores)
    n_workers: int = 1                    # >=1 = process pool (1 means a single worker process)
    score_fn_name: Optional[str] = None   # required if n_workers > 1
    score_args: Optional[Tuple[Any, ...]] = None
    score_kwargs: Optional[Dict[str, Any]] = None

    # Parallel mutation (offspring creation)
    mutation_n_workers: int = 1           # 1 = serial; >1 = parallel child creation

    # Distance parameters (applied consistently in main & workers)
    dist_max_hops: Optional[int] = None
    dist_node_cap: Optional[int] = None
    dist_weight_eps: float = 1e-9
    dist_unreachable: float = 1e6

    # Randomness
    seed: Optional[int] = None


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
    """
    Remove ALL transitions with probability < threshold, action by action.
    - Ignores whitelist and min_out_degree.
    - After pruning each (s,a), probabilities are renormalized over the remaining successors.
    - If nothing remains for (s,a), that action becomes empty (no outgoing succs); this is allowed.
    """
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
# Crossover and selection
# -------------------------------

def crossover_action_block(parent_a: 'MDPNetwork',
                           parent_b: 'MDPNetwork',
                           rng: np.random.Generator) -> 'MDPNetwork':
    child = parent_a.clone()
    for s in child.states:
        if s in child.terminal_states:
            continue
        for a in range(child.num_actions):
            use_a = (rng.random() < 0.5)
            src = parent_a if use_a else parent_b
            src_map = get_outgoing_for_action(src, s, a)
            if not src_map:
                continue
            set_outgoing_for_action(child, s, a, src_map)
    return child


def tournament_select(pop: List['MDPNetwork'],
                      scores: List[float],
                      rng: np.random.Generator,
                      k: int) -> 'MDPNetwork':
    idxs = rng.choice(len(pop), size=k, replace=False)
    best_idx = int(idxs[0])
    best_score = scores[best_idx]
    for i in idxs[1:]:
        si = scores[int(i)]
        if si > best_score:
            best_idx = int(i)
            best_score = si
    return pop[best_idx]


# -------------------------------
# Parallel scoring workers (name-based only)
# -------------------------------

def _score_worker_by_name(payload: Tuple[Dict[str, Any], str, Tuple[Any, ...], Dict[str, Any], Optional[List[Dict[str, Any]]]]) -> float:
    """
    payload = (mdp_portable, score_fn_name, args, kwargs, precomputed_portables)
    precomputed_portables is a list of dicts produced by Serialisable.to_portable() in main process.
    """
    portable, fn_name, args, kwargs, precomputed_portables = payload
    mdp = MDPNetwork.from_portable(portable)
    fn = get_registered_score_fn(fn_name)

    # Inject precomputed into kwargs (non-destructive)
    if precomputed_portables is not None and "precomputed_portables" not in kwargs:
        local_kwargs = dict(kwargs)
        local_kwargs["precomputed_portables"] = precomputed_portables
    else:
        local_kwargs = kwargs

    return float(fn(mdp, *args, **local_kwargs))


def evaluate_mdp_list(
    mdps: List['MDPNetwork'],
    *,
    score_fn_name: str,
    n_workers: int,
    score_args: Optional[Tuple[Any, ...]] = None,
    score_kwargs: Optional[Dict[str, Any]] = None,
    precomputed_portables: Optional[List[Dict[str, Any]]] = None,  # PRECOMPUTE: pass-through
) -> List[float]:
    if score_fn_name is None:
        raise ValueError("score_fn_name is required (parallel-only).")
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1.")
    args = score_args or ()
    kwargs = score_kwargs or {}

    payloads = [
        (m.to_portable(), score_fn_name, args, kwargs, precomputed_portables)
        for m in mdps
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_score_worker_by_name, payloads))
    return [float(x) for x in results]


# -------------------------------
# Parallel offspring worker (init or mate)
# -------------------------------

def _apply_mutations(ind: 'MDPNetwork', rng: np.random.Generator, whitelist: Set[EdgeTriple], cfg: GAConfig):
    # Distance-weighted add edge
    for _ in range(cfg.add_edge_attempts_per_child):
        mutation_add_edge(ind, rng, lambda _m, _s, _sp: _dist_from_cfg(ind, _s, _sp, cfg), cfg)

    # Probability local pairwise tweaks
    mutation_prob_pairwise(ind, rng, cfg)

    # Reward small-step (optional)
    if cfg.reward_tweak_edges_per_child > 0:
        mutation_reward_smallstep(ind, rng, cfg)

    # Hard prune of low-prob transitions (optional)
    if cfg.prune_prob_threshold is not None:
        prune_low_prob_transitions(ind, cfg.prune_prob_threshold)


def _child_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns child's portable dict from to_portable().
    Two modes:
      - Init: payload has "base_portable"
      - Mate: payload has "pa_portable", "pb_portable", "do_crossover"
    """
    cfg: GAConfig = payload["cfg"]
    whitelist: Set[EdgeTriple] = set(payload["whitelist"])
    rng = np.random.default_rng(payload["seed"])

    def _mk_from_portable(p: Dict[str, Any]) -> MDPNetwork:
        return MDPNetwork.from_portable(p)

    if "pa_portable" in payload:
        # Mate path
        pa = _mk_from_portable(payload["pa_portable"])
        pb = _mk_from_portable(payload["pb_portable"])
        do_crossover: bool = payload["do_crossover"]

        if do_crossover:
            child = crossover_action_block(pa, pb, rng)
        else:
            child = (pa if rng.random() < 0.5 else pb).clone()

        _apply_mutations(child, rng, whitelist, cfg)
        return child.to_portable()

    else:
        # Init path
        ind = _mk_from_portable(payload["base_portable"])
        _apply_mutations(ind, rng, whitelist, cfg)
        return ind.to_portable()


# -------------------------------
# GA main class
# -------------------------------

class MDPEvolutionGA:
    """
    GA over MDPNetwork with:
    - distance-weighted add-edge (directed_prob_distance),
    - probability pairwise tweaks,
    - reward bounded small-step tweaks,
    - hard pruning of low-prob transitions (optional),
    - parallel evaluation and parallel offspring creation only.

    Precompute notes:
    - Users may set `self.precompute_hook` to a callable: (MDPNetwork) -> List[Serialisable]
      OR set `self.precomputed_artifacts` to an already-built list of Serialisable.
    - We intentionally do NOT place these on GAConfig to keep GAConfig picklable for workers.
    """

    def __init__(self,
                 base_mdp: 'MDPNetwork',
                 cfg: GAConfig):
        if cfg.score_fn_name is None:
            raise ValueError("score_fn_name is required (parallel-only).")
        if cfg.n_workers < 1 or cfg.mutation_n_workers < 1:
            raise ValueError("n_workers and mutation_n_workers must be >= 1.")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Baseline and whitelist
        self.base_mdp = base_mdp.clone()
        self.whitelist: Set[EdgeTriple] = build_original_whitelist(self.base_mdp)

        # Single portable blob for cross-process transport
        self._base_portable = self.base_mdp.to_portable()

        # --- PRECOMPUTE hook slots (simple, main-process only; never sent to workers) ---
        self.precomputed_artifacts: Optional[List[Serialisable]] = None

        # Will hold the final JSON-friendly list of dicts ready for workers
        self._precomputed_portables: Optional[List[Dict[str, Any]]] = None

    # ---------- internal helpers ----------

    def _init_population(self) -> List['MDPNetwork']:
        pop: List[MDPNetwork] = [self.base_mdp.clone()]
        need = self.cfg.population_size - 1
        if need <= 0:
            return pop

        payloads: List[Dict[str, Any]] = []
        for _ in range(need):
            payloads.append({
                "cfg": self.cfg,  # GAConfig must remain pickle-friendly
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

    def _evaluate_population(self, pop: List['MDPNetwork']) -> List[float]:
        return evaluate_mdp_list(
            mdps=pop,
            score_fn_name=self.cfg.score_fn_name,
            n_workers=self.cfg.n_workers,
            score_args=self.cfg.score_args,
            score_kwargs=self.cfg.score_kwargs,
            precomputed_portables=self._precomputed_portables,  # PRECOMPUTE: broadcast to workers
        )

    # ---------- public API ----------

    def run(self) -> Tuple['MDPNetwork', float, List[float]]:
        # ----- PRECOMPUTE: run once (serial, main process) -----
        # This block is intentionally simple. Users can assign:
        #   - self.precomputed_artifacts = [obj1, obj2, ...]
        # If both are None, we skip precompute and workers receive None.
        if self.precomputed_artifacts is not None:
            self._precomputed_portables = [obj.to_portable() for obj in self.precomputed_artifacts]
            print(f"[Precompute] Using provided artifacts: {len(self._precomputed_portables)} item(s).")
        else:
            self._precomputed_portables = None
            print("[Precompute] Skipped (no artifacts).")

        # ----- Init population & evaluate -----
        pop = self._init_population()
        scores = self._evaluate_population(pop)

        best_idx = int(np.argmax(scores))
        best_mdp = pop[best_idx].clone()
        best_score = float(scores[best_idx])
        history: List[float] = [best_score]

        print(f"[Init] pop={len(pop)} | best={best_score:.6f} | "
              f"mean={np.mean(scores):.6f} | std={np.std(scores):.6f}")

        # ----- Generations -----
        for gen in range(self.cfg.generations):
            # Elitism
            elite_count = min(self.cfg.elitism_num, len(pop))
            elite_idxs = list(np.argsort(scores)[-elite_count:])
            elites = [pop[int(i)].clone() for i in elite_idxs]

            # Children via parallel path
            target = self.cfg.population_size - elite_count
            if target <= 0:
                new_pop: List[MDPNetwork] = elites
            else:
                parents_pairs: List[Tuple[MDPNetwork, MDPNetwork]] = []
                for _ in range(target):
                    p1 = tournament_select(pop, scores, self.rng, self.cfg.tournament_k)
                    p2 = tournament_select(pop, scores, self.rng, self.cfg.tournament_k)
                    parents_pairs.append((p1, p2))

                children = self._make_children_parallel(parents_pairs)
                new_pop = elites + children

            # Next generation
            pop = new_pop
            scores = self._evaluate_population(pop)

            # Track best
            gen_best_idx = int(np.argmax(scores))
            gen_best_score = float(scores[gen_best_idx])

            improved = gen_best_score > best_score
            if improved:
                best_score = gen_best_score
                best_mdp = pop[gen_best_idx].clone()

            history.append(best_score)

            print(f"[Gen {gen + 1}/{self.cfg.generations}] elites={elite_count} | "
                  f"gen_best={gen_best_score:.6f} | mean={np.mean(scores):.6f} | "
                  f"std={np.std(scores):.6f} | best_so_far={best_score:.6f} | "
                  f"improved={'YES' if improved else 'no'}")

        return best_mdp, best_score, history


# -------------------------------
# Example score fn + registration
# -------------------------------

def example_score_fn(mdp: 'MDPNetwork', *args, **kwargs) -> float:
    """
    Example: simple sum of expected rewards over all (s, a), skipping terminals.
    Demonstrates how to read precomputed_portables if present.
    """
    _pre = kwargs.get("precomputed_portables", None)  # List[dict] or None
    # If you had a QTable portable at _pre[0], you could hydrate it here, e.g.:
    # from mdp_network.mdp_tables import QTable
    # q = QTable.from_portable(_pre[0])  # if your QTable implements Serialisable
    # ... use q in scoring ...

    total = 0.0
    for s in mdp.states:
        if s in mdp.terminal_states:
            continue
        for a in range(mdp.num_actions):
            total += mdp.compute_expected_reward(s, a)
    return float(total)


register_score_fn("example", example_score_fn)


if __name__ == "__main__":
    # ----- build a tiny demo MDP (5 states, 2 actions) -----
    nS, nA = 5, 2
    cfg_demo: Dict[str, Any] = {
        "num_actions": nA,
        "states": list(range(nS)),
        "start_states": [0],
        "terminal_states": [4],
        "default_reward": 0.0,
        "transitions": {
            "0": {
                "0": {"1": {"p": 0.8, "r": 0.0}, "0": {"p": 0.2, "r": 0.0}},
                "1": {"2": {"p": 1.0, "r": 0.0}},
            },
            "1": {
                "0": {"2": {"p": 0.7, "r": 0.0}, "3": {"p": 0.3, "r": 0.0}},
                "1": {"1": {"p": 1.0, "r": 0.0}},
            },
            "2": {
                "0": {"3": {"p": 1.0, "r": 0.0}},
                "1": {"2": {"p": 1.0, "r": 0.0}},
            },
            "3": {
                "0": {"4": {"p": 0.9, "r": 1.0}, "3": {"p": 0.1, "r": 0.0}},
                "1": {"1": {"p": 1.0, "r": 0.0}},
            },
        },
    }
    base_mdp = MDPNetwork(config_data=cfg_demo)

    # Ensure example score fn is registered
    register_score_fn("example", example_score_fn)

    # ----- GA config -----
    cfg = GAConfig(
        population_size=64,
        generations=10,
        tournament_k=2,
        elitism_num=4,
        crossover_rate=1.0,
        allow_self_loops=True,
        min_out_degree=1,
        max_out_degree=6,
        prob_floor=1e-6,
        add_edge_attempts_per_child=1,
        epsilon_new_prob=0.02,
        gamma_sample=1.0,
        gamma_prob=0.0,
        prune_prob_threshold=1e-3,
        prob_tweak_actions_per_child=8,
        prob_pairwise_step=0.02,
        reward_tweak_edges_per_child=16,
        reward_k_percent=0.02,
        reward_ref_floor=1e-3,
        reward_min=None,
        reward_max=None,

        # Parallel scoring (name-based only)
        n_workers=max(1, os.cpu_count() or 1),
        score_fn_name="example",
        score_args=None,
        score_kwargs=None,

        # Parallel offspring
        mutation_n_workers=max(1, os.cpu_count() or 1),

        # Distance params (used both in main & workers)
        dist_max_hops=3,
        dist_node_cap=1000,
        dist_weight_eps=1e-6,
        dist_unreachable=1e6,

        seed=123,
    )

    ga = MDPEvolutionGA(
        base_mdp=base_mdp,
        cfg=cfg,
    )

    best_mdp, best_score, history = ga.run()
    print("\n=== GA finished ===")
    print("Best score:", best_score)
    print("History:", [round(x, 6) for x in history])

    # Optionally export the best model
    out_dir = "./outputs_ga_demo"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "best_mdp_demo.json")
    best_mdp.export_to_json(out_path)
    print("Saved best MDP to:", out_path)
