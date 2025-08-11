# ga_mdp_search.py
# Genetic search system for MDPNetwork (structure + probability + reward) with
# optional parallel scoring using a process pool (Linux, Python 3.10).
#
# Key points:
# - Simple tournament selection and (s,a)-block crossover.
# - Add-edge sampling is distance-weighted; new-edge reward = mean inbound rewards to s'.
# - Deletion honors a whitelist of original transitions (never delete originals).
# - Small-step mutations with fixed hyperparameters (no annealing).
# - Parallel evaluation API to score a whole list of MDPs and return a list of scores.
# - Multiple score functions supported via a registry (name -> callable).
#
# NOTE: For parallel mode, it's safest to register your score function under a name
# and pass that name so worker processes can look it up reliably.

from __future__ import annotations

from typing import Callable, Dict, Tuple, List, Optional, Set, Any
from dataclasses import dataclass
import math
import pickle
import os

import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed

from mdp_network import MDPNetwork

# -------------------------------
# Types
# -------------------------------

State = int
Action = int
EdgeTriple = Tuple[State, Action, State]
ScoreFn = Callable[['MDPNetwork'], float]
DistanceFn = Callable[['MDPNetwork', State, State], float]


# -------------------------------
# Score function registry
# -------------------------------

SCORE_FN_REGISTRY: Dict[str, ScoreFn] = {}


def register_score_fn(name: str, fn: ScoreFn) -> None:
    """Register a score function for use in parallel workers by name."""
    SCORE_FN_REGISTRY[name] = fn


def get_registered_score_fn(name: str) -> ScoreFn:
    if name not in SCORE_FN_REGISTRY:
        raise KeyError(f"Score function '{name}' is not registered.")
    return SCORE_FN_REGISTRY[name]


def is_picklable(obj: Any) -> bool:
    """Best-effort check for picklability."""
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


# -------------------------------
# MDP clone & (de)serialization (no deepcopy)
# -------------------------------

def mdp_to_config_and_maps(src: 'MDPNetwork') -> Tuple[Dict, Optional[Dict[int, Any]], Optional[Dict[Any, int]]]:
    """
    Build a serializable config dict (same schema as export_to_json) and return optional mappings.
    This avoids deepcopy and enables safe inter-process transport.
    """
    transitions: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for s in src.states:
        action_map: Dict[int, Dict[int, Dict[str, float]]] = {}
        for sp in src.graph.successors(s):
            edata = src.graph[s][sp]
            if "transitions" not in edata:
                continue
            for a, ar in edata["transitions"].items():
                action_map.setdefault(int(a), {})
                action_map[int(a)][int(sp)] = {"p": float(ar["p"]), "r": float(ar["r"])}
        if action_map:
            s_trans: Dict[str, Dict[str, Dict[str, float]]] = {}
            for a, sp_dict in action_map.items():
                a_str = str(int(a))
                s_trans.setdefault(a_str, {})
                for sp, ar in sp_dict.items():
                    s_trans[a_str][str(int(sp))] = {"p": float(ar["p"]), "r": float(ar["r"])}
            transitions[str(int(s))] = s_trans

    cfg: Dict = {
        "num_actions": int(src.num_actions),
        "states": [int(s) for s in src.states],
        "start_states": [int(s) for s in src.start_states],
        "terminal_states": [int(s) for s in src.terminal_states],
        "default_reward": float(src.default_reward),
        "transitions": transitions,
    }

    if src.tags:
        cfg["tags"] = {k: sorted(int(x) for x in v) for k, v in src.tags.items()}

    int_to_state = src.int_to_state if getattr(src, "has_string_mapping", False) else None
    state_to_int = src.state_to_int if getattr(src, "has_string_mapping", False) else None
    return cfg, int_to_state, state_to_int


def clone_mdp_network(src: 'MDPNetwork') -> 'MDPNetwork':
    """Clone by reconstructing from a config dict."""
    cfg, int_to_state, state_to_int = mdp_to_config_and_maps(src)
    return MDPNetwork(config_data=cfg, int_to_state=int_to_state, state_to_int=state_to_int)


# -------------------------------
# Default distance function
# -------------------------------

def default_distance_fn(mdp: 'MDPNetwork', s: State, sp: State) -> float:
    """
    Graph-based distance. Uses undirected shortest-path length on the current graph topology.
    If unreachable, returns a large constant. Distance to self is 0.
    """
    if s == sp:
        return 0.0
    G_u = mdp.graph.to_undirected(as_view=True)
    try:
        d = nx.shortest_path_length(G_u, s, sp)
        return float(d)
    except nx.NetworkXNoPath:
        return 1e6


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
    crossover_rate: float = 1.0  # apply crossover for most children

    # Structural constraints
    allow_self_loops: bool = True
    min_out_degree: int = 1
    max_out_degree: int = 8
    prob_floor: float = 1e-6

    # Add-edge parameters
    add_edge_attempts_per_child: int = 2
    epsilon_new_prob: float = 0.02  # base initial prob for a new edge
    gamma_sample: float = 1.0       # strength for distance-weighted candidate sampling
    gamma_prob: float = 0.0         # if >0, scale p_new by exp(-gamma_prob * distance)

    # Delete-edge parameters
    delete_edge_attempts_per_child: int = 1
    delete_tau: float = 1.0         # weight ~ (p+eps)^(-tau)
    delete_eps: float = 1e-9

    # Probability tweak parameters (pairwise small transfer)
    prob_tweak_actions_per_child: int = 20   # number of (s,a) to tweak per child
    prob_pairwise_step: float = 0.02         # max delta to move between two edges in a pair

    # Reward tweak parameters
    reward_tweak_edges_per_child: int = 50   # number of (s,a,s') to tweak per child
    reward_k_percent: float = 0.02           # maximum relative change per tweak (e.g., 0.02 = 2%)
    reward_ref_floor: float = 1e-3           # baseline to avoid zero step when |r|~0
    reward_min: Optional[float] = None       # clip lower bound (None = no clip)
    reward_max: Optional[float] = None       # clip upper bound (None = no clip)

    # Parallel evaluation
    n_workers: int = 1                       # 1 = serial; >1 = process pool
    score_fn_name: Optional[str] = None      # prefer using a registered name in parallel mode

    # Randomness
    seed: Optional[int] = None


# -------------------------------
# Utilities over MDPNetwork
# -------------------------------

def get_outgoing_for_action(mdp: 'MDPNetwork', s: State, a: Action) -> Dict[State, Tuple[float, float]]:
    """Return a dict: sp -> (p, r) for a given (s,a)."""
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
    """
    Overwrite the (s,a) outgoing distribution with new_map (sp -> (p, r)).
    After writing, renormalizes to ensure sum p == 1 (if total>0).
    """
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
    """Mean reward over all incoming edges to target sp; fallback if none."""
    vals: List[float] = []
    for s in mdp.graph.predecessors(sp):
        edata = mdp.graph[s][sp]
        if "transitions" not in edata:
            continue
        for a, ar in edata["transitions"].items():
            vals.append(float(ar["r"]))
    return float(np.mean(vals)) if vals else float(fallback)


def action_out_degree(mdp: 'MDPNetwork', s: State, a: Action) -> int:
    """Number of next states with non-zero probability for (s,a)."""
    return sum(
        1 for sp in mdp.graph.successors(s)
        if "transitions" in mdp.graph[s][sp] and a in mdp.graph[s][sp]["transitions"]
    )


def list_all_action_pairs(mdp: 'MDPNetwork') -> List[Tuple[State, Action]]:
    """List all (s,a) pairs for non-terminal s."""
    pairs: List[Tuple[State, Action]] = []
    for s in mdp.states:
        if s in mdp.terminal_states:
            continue
        for a in range(mdp.num_actions):
            pairs.append((s, a))
    return pairs


def list_all_triples(mdp: 'MDPNetwork') -> List[EdgeTriple]:
    """List all (s,a,s') that currently exist."""
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
    """Original (s,a,s') transitions; never delete these."""
    return set(list_all_triples(mdp))


# -------------------------------
# Mutation operators
# -------------------------------

def mutation_add_edge(mdp: 'MDPNetwork',
                      rng: np.random.Generator,
                      dist_fn: DistanceFn,
                      cfg: GAConfig):
    """Add a new (s,a,s') preferring nearer s' by distance weighting."""
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
        d = dist_fn(mdp, s, sp)
        w = math.exp(-cfg.gamma_sample * d) if cfg.gamma_sample > 0.0 else 1.0
        weights.append(w)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0.0:
        return
    weights /= weights.sum()
    sp_new = int(sp_candidates[int(np.random.choice(len(sp_candidates), p=weights))])

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


def mutation_delete_edge(mdp: 'MDPNetwork',
                         rng: np.random.Generator,
                         whitelist: Set[EdgeTriple],
                         cfg: GAConfig):
    """Delete a non-whitelisted edge, preferring smaller-prob edges; keep min_out_degree."""
    pairs = []
    for (s, a) in list_all_action_pairs(mdp):
        out_map = get_outgoing_for_action(mdp, s, a)
        if len(out_map) <= cfg.min_out_degree:
            continue
        deletable = [sp for sp in out_map.keys() if (s, a, sp) not in whitelist]
        if deletable:
            pairs.append((s, a, out_map, deletable))
    if not pairs:
        return

    s, a, out_map, deletable = pairs[rng.integers(0, len(pairs))]
    ps = np.array([out_map[sp][0] for sp in deletable], dtype=float)
    w = np.power(ps + cfg.delete_eps, -cfg.delete_tau)
    w /= w.sum()
    sp_del = int(deletable[int(rng.choice(len(deletable), p=w))])

    del out_map[sp_del]
    if len(out_map) < cfg.min_out_degree:
        return

    total = sum(p for (p, _) in out_map.values())
    if total <= 0:
        m = len(out_map)
        for sp in out_map:
            _, r = out_map[sp]
            out_map[sp] = (1.0 / m, r)
    else:
        for sp in out_map:
            p, r = out_map[sp]
            out_map[sp] = (max(cfg.prob_floor, p / total), r)

    set_outgoing_for_action(mdp, s, a, out_map)


def mutation_prob_pairwise(mdp: 'MDPNetwork',
                           rng: np.random.Generator,
                           cfg: GAConfig):
    """For several (s,a), move a small probability mass between two successors."""
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
    """Tweak rewards for several random (s,a,s') with a bounded relative step (<= k%)."""
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
    """
    (s,a)-level crossover: for each (s,a), copy the whole outgoing table from either A or B.
    Then renormalize per (s,a).
    Assumes both parents share the same states/actions metadata.
    """
    child = clone_mdp_network(parent_a)
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
    """k-way tournament selection. Returns a reference (not a copy)."""
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
# Parallel scoring workers
# -------------------------------

def _score_worker_by_name(payload: Tuple[Dict, Optional[Dict[int, Any]], Optional[Dict[Any, int]], str]) -> float:
    """
    Worker: rebuild MDP from config and call a registered score function by name.
    Payload: (cfg, int_to_state, state_to_int, score_fn_name)
    """
    cfg, int_to_state, state_to_int, fn_name = payload
    mdp = MDPNetwork(config_data=cfg, int_to_state=int_to_state, state_to_int=state_to_int)
    fn = get_registered_score_fn(fn_name)
    return float(fn(mdp))


def _score_worker_with_callable(payload: Tuple[Dict, Optional[Dict[int, Any]], Optional[Dict[Any, int]], ScoreFn]) -> float:
    """
    Worker: rebuild MDP and call a picklable score function (top-level).
    Payload: (cfg, int_to_state, state_to_int, score_fn_callable)
    """
    cfg, int_to_state, state_to_int, fn = payload
    mdp = MDPNetwork(config_data=cfg, int_to_state=int_to_state, state_to_int=state_to_int)
    return float(fn(mdp))


def evaluate_mdp_list(
    mdps: List['MDPNetwork'],
    score_fn: Optional[ScoreFn] = None,
    score_fn_name: Optional[str] = None,
    n_workers: int = 1,
) -> List[float]:
    """
    Evaluate a list of MDPNetwork individuals and return a list of scores.
    - If n_workers == 1: serial evaluation.
    - If n_workers > 1:
        * Preferred: provide score_fn_name of a registered function.
        * Fallback: if a picklable score_fn is given, we can parallelize as well.
        * Otherwise, falls back to serial to stay safe.
    """
    if n_workers <= 1:
        # Serial path on the live objects (no reconstruction overhead).
        if score_fn_name is not None:
            fn = get_registered_score_fn(score_fn_name)
            return [float(fn(m)) for m in mdps]
        elif score_fn is not None:
            return [float(score_fn(m)) for m in mdps]
        else:
            raise ValueError("Either score_fn_name or score_fn must be provided.")

    # Parallel path (process pool)
    if score_fn_name is not None:
        # Build payloads using configs
        payloads = []
        for m in mdps:
            cfg, int_to_state, state_to_int = mdp_to_config_and_maps(m)
            payloads.append((cfg, int_to_state, state_to_int, score_fn_name))
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_score_worker_by_name, payloads))
        return [float(x) for x in results]

    if score_fn is not None and is_picklable(score_fn):
        payloads = []
        for m in mdps:
            cfg, int_to_state, state_to_int = mdp_to_config_and_maps(m)
            payloads.append((cfg, int_to_state, state_to_int, score_fn))
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_score_worker_with_callable, payloads))
        return [float(x) for x in results]

    # Fallback: serial if we cannot safely parallelize
    if score_fn is not None:
        return [float(score_fn(m)) for m in mdps]
    raise ValueError("Parallel requested but neither a registered score_fn_name nor a picklable score_fn was provided.")


# -------------------------------
# GA main class
# -------------------------------

class MDPEvolutionGA:
    """
    Genetic algorithm over MDPNetwork with:
    - distance-weighted add-edge,
    - probability pairwise tweaks,
    - reward bounded small-step tweaks,
    - deletion preferring low-prob edges (never delete original whitelist edges).
    - Optional parallel evaluation via process pool.
    """

    def __init__(self,
                 base_mdp: 'MDPNetwork',
                 score_fn: Optional[ScoreFn],      # may be None if you use score_fn_name
                 cfg: GAConfig,
                 dist_fn: Optional[DistanceFn] = None):
        if (cfg.score_fn_name is None) and (score_fn is None):
            raise ValueError("Provide either cfg.score_fn_name (registered) or a score_fn callable.")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.score_fn = score_fn
        self.dist_fn = dist_fn or default_distance_fn

        # Baseline and whitelist
        self.base_mdp = clone_mdp_network(base_mdp)
        self.whitelist: Set[EdgeTriple] = build_original_whitelist(self.base_mdp)

    # ---------- initialization ----------

    def _init_population(self) -> List['MDPNetwork']:
        """Initial population: baseline + diversified variants."""
        pop: List[MDPNetwork] = []
        pop.append(clone_mdp_network(self.base_mdp))

        for _ in range(self.cfg.population_size - 1):
            ind = clone_mdp_network(self.base_mdp)
            for _ in range(self.cfg.add_edge_attempts_per_child):
                mutation_add_edge(ind, self.rng, self.dist_fn, self.cfg)
            for _ in range(self.cfg.delete_edge_attempts_per_child):
                mutation_delete_edge(ind, self.rng, self.whitelist, self.cfg)
            mutation_prob_pairwise(ind, self.rng, self.cfg)
            mutation_reward_smallstep(ind, self.rng, self.cfg)
            pop.append(ind)
        return pop

    def _make_child(self, parent_a: 'MDPNetwork', parent_b: 'MDPNetwork') -> 'MDPNetwork':
        """Create a child by optional crossover and then apply fixed mutations."""
        if self.rng.random() < self.cfg.crossover_rate:
            child = crossover_action_block(parent_a, parent_b, self.rng)
        else:
            child = clone_mdp_network(parent_a if self.rng.random() < 0.5 else parent_b)

        for _ in range(self.cfg.add_edge_attempts_per_child):
            mutation_add_edge(child, self.rng, self.dist_fn, self.cfg)
        for _ in range(self.cfg.delete_edge_attempts_per_child):
            mutation_delete_edge(child, self.rng, self.whitelist, self.cfg)
        mutation_prob_pairwise(child, self.rng, self.cfg)
        mutation_reward_smallstep(child, self.rng, self.cfg)
        return child

    # ---------- evaluation ----------

    def _evaluate_population(self, pop: List['MDPNetwork']) -> List[float]:
        """Evaluate population serially or in parallel as configured."""
        return evaluate_mdp_list(
            mdps=pop,
            score_fn=self.score_fn,
            score_fn_name=self.cfg.score_fn_name,
            n_workers=self.cfg.n_workers,
        )

    # ---------- public API ----------

    def run(self) -> Tuple['MDPNetwork', float, List[float]]:
        """
        Run GA for a fixed number of generations.
        Returns: (best_mdp, best_score, history_best_scores_per_generation)
        """
        pop = self._init_population()
        scores = self._evaluate_population(pop)

        best_idx = int(np.argmax(scores))
        best_mdp = clone_mdp_network(pop[best_idx])
        best_score = float(scores[best_idx])
        history: List[float] = [best_score]

        # --- print init stats ---
        print(f"[Init] pop={len(pop)} | best={best_score:.6f} | "
              f"mean={np.mean(scores):.6f} | std={np.std(scores):.6f}")

        for gen in range(self.cfg.generations):
            elite_count = min(self.cfg.elitism_num, len(pop))
            elite_idxs = list(np.argsort(scores)[-elite_count:])
            elites = [clone_mdp_network(pop[int(i)]) for i in elite_idxs]

            new_pop: List[MDPNetwork] = elites[:]
            while len(new_pop) < self.cfg.population_size:
                p1 = tournament_select(pop, scores, self.rng, self.cfg.tournament_k)
                p2 = tournament_select(pop, scores, self.rng, self.cfg.tournament_k)
                child = self._make_child(p1, p2)
                new_pop.append(child)

            pop = new_pop
            scores = self._evaluate_population(pop)

            gen_best_idx = int(np.argmax(scores))
            gen_best_score = float(scores[gen_best_idx])

            # Check improvement against historical best BEFORE updating it
            improved = gen_best_score > best_score
            if improved:
                best_score = gen_best_score
                best_mdp = clone_mdp_network(pop[gen_best_idx])

            history.append(best_score)

            # --- print per-generation stats ---
            print(f"[Gen {gen + 1}/{self.cfg.generations}] elites={elite_count} | "
                  f"gen_best={gen_best_score:.6f} | mean={np.mean(scores):.6f} | "
                  f"std={np.std(scores):.6f} | best_so_far={best_score:.6f} | "
                  f"improved={'YES' if improved else 'no'}")

        return best_mdp, best_score, history

# -------------------------------
# Example score fn + registration
# -------------------------------

def example_score_fn(mdp: 'MDPNetwork') -> float:
    """
    Placeholder scoring function.
    Replace this with your domain-specific evaluation (DP/simulation/etc.).
    Must return a single scalar (higher is better).
    """
    total = 0.0
    for s in mdp.states:
        if s in mdp.terminal_states:
            continue
        for a in range(mdp.num_actions):
            total += mdp.compute_expected_reward(s, a)
    return float(total)


# Register the example under a name so parallel workers can find it.
register_score_fn("example", example_score_fn)


# -------------------------------
# Example wiring (optional)
# -------------------------------

def main_example(base_mdp: 'MDPNetwork'):
    """
    Minimal example showing how to run the GA.
    For parallel evaluation, set cfg.n_workers > 1 and provide cfg.score_fn_name
    that was registered via register_score_fn(...).
    """
    cfg = GAConfig(
        population_size=80,
        generations=50,
        tournament_k=2,
        elitism_num=8,
        crossover_rate=1.0,
        allow_self_loops=True,
        min_out_degree=1,
        max_out_degree=8,
        prob_floor=1e-6,
        add_edge_attempts_per_child=2,
        epsilon_new_prob=0.02,
        gamma_sample=1.0,
        gamma_prob=0.0,
        delete_edge_attempts_per_child=1,
        delete_tau=1.0,
        delete_eps=1e-9,
        prob_tweak_actions_per_child=20,
        prob_pairwise_step=0.02,
        reward_tweak_edges_per_child=50,
        reward_k_percent=0.02,
        reward_ref_floor=1e-3,
        reward_min=None,
        reward_max=None,

        # ---- parallel settings ----
        n_workers=os.cpu_count() or 4,   # e.g., use all cores
        score_fn_name="example",         # use a registered fn for parallel safety
        seed=42,
    )

    ga = MDPEvolutionGA(
        base_mdp=base_mdp,
        score_fn=None,                # use name-based lookup in workers
        cfg=cfg,
        dist_fn=default_distance_fn
    )

    best_mdp, best_score, history = ga.run()
    print("Best score:", best_score)
    # best_mdp.export_to_json("best_mdp.json")


# If you want to enable running as a script, uncomment below and provide a base MDP:
# if __name__ == "__main__":
#     # Load or build your base MDPNetwork here, then call main_example(base_mdp)
#     pass
