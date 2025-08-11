# test_ga_frozenlake8x8.py
# English comments only. Minimal, self-contained sanity test for ga_mdp_search.
# - Builds an MDPNetwork from Gymnasium FrozenLake 8x8 (slippery)
# - Runs the GA for a few generations with small population
# - Uses process pool parallel scoring to validate the path
# - Prints best score and simple assertions

import os
from typing import Dict, Any, Tuple

# ---- Gymnasium FrozenLake for building the MDP ----
import gymnasium as gym

# ---- Your GA system ----
from mdp_network.ga_mdp_search import (
    GAConfig,
    MDPEvolutionGA,
    register_score_fn,
    example_score_fn,
    evaluate_mdp_list,
    clone_mdp_network,
)

# ---- Import MDPNetwork (adjust import path to your project if needed) ----
try:
    # common package-style path
    from mdp_network.mdp_network import MDPNetwork  # type: ignore
except Exception:
    # fallback: same-folder module or plain name
    from MDPNetwork import MDPNetwork  # type: ignore


def frozenlake_to_mdp_config(map_name: str = "8x8", is_slippery: bool = True) -> Tuple[Dict[str, Any], None, None]:
    """
    Build a config_data dict for MDPNetwork from Gymnasium FrozenLake-v1.
    Returns (config_data, int_to_state, state_to_int); mappings are None here.
    """
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    env.reset(seed=0)
    P = env.unwrapped.P  # shape: dict[state][action] -> list of (p, next_state, reward, done)

    nS = env.observation_space.n
    nA = env.action_space.n
    default_reward = 0.0

    # Detect terminal states: in FrozenLake, terminals self-loop with done=True and prob=1.0 for all actions.
    terminal_states = []
    for s in range(nS):
        is_terminal = True
        for a in range(nA):
            trans = P[s][a]
            # Terminal if only one outcome and it's (1.0, s, r, True)
            if not (len(trans) == 1 and trans[0][3] is True and trans[0][1] == s and abs(trans[0][0] - 1.0) < 1e-12):
                is_terminal = False
                break
        if is_terminal:
            terminal_states.append(s)

    # Build transitions dict in the schema expected by MDPNetwork
    transitions: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for s in range(nS):
        s_map: Dict[str, Dict[str, Dict[str, float]]] = {}
        for a in range(nA):
            next_map: Dict[str, Dict[str, float]] = {}
            for (p, sp, r, _done) in P[s][a]:
                # Aggregate in case multiple entries go to the same sp (should not happen here, but safe)
                entry = next_map.setdefault(str(int(sp)), {"p": 0.0, "r": 0.0})
                entry["p"] += float(p)
                # In FrozenLake, reward is deterministic per outcome; keep the last seen (all same except goal=1.0)
                entry["r"] = float(r)
            if next_map:
                s_map[str(int(a))] = next_map
        if s_map:
            transitions[str(int(s))] = s_map

    cfg = {
        "num_actions": int(nA),
        "states": list(range(nS)),
        "start_states": [0],  # FrozenLake starts at top-left
        "terminal_states": terminal_states,
        "default_reward": float(default_reward),
        "transitions": transitions,
    }
    return cfg, None, None


def build_frozenlake_mdp(map_name: str = "8x8", is_slippery: bool = True) -> MDPNetwork:
    """Create an MDPNetwork instance from FrozenLake config."""
    cfg, int_to_state, state_to_int = frozenlake_to_mdp_config(map_name, is_slippery)
    return MDPNetwork(config_data=cfg, int_to_state=int_to_state, state_to_int=state_to_int)


if __name__ == "__main__":
    out_dir = "./outputs/ga_test"

    print("\n=== Build FrozenLake 8x8 (slippery) MDPNetwork ===")
    mdp = build_frozenlake_mdp(map_name="8x8", is_slippery=True)
    print(f"|S|={len(mdp.states)}  |A|={mdp.num_actions}  terminals={len(mdp.terminal_states)}")

    # Register example score function under a name for parallel workers.
    register_score_fn("example", example_score_fn)

    # Small GA config for a quick sanity check (and enable process pool)
    cfg = GAConfig(
        population_size=256,
        generations=64,
        tournament_k=2,
        elitism_num=16,
        crossover_rate=0.5,
        allow_self_loops=True,
        min_out_degree=1,
        max_out_degree=6,
        prob_floor=1e-6,
        add_edge_attempts_per_child=10,
        epsilon_new_prob=0.1,
        gamma_sample=1.0,
        gamma_prob=0.0,
        delete_edge_attempts_per_child=1,
        delete_tau=1.0,
        delete_eps=1e-9,
        prob_tweak_actions_per_child=50,
        prob_pairwise_step=0.05,
        reward_tweak_edges_per_child=5,
        reward_k_percent=0.05,
        reward_ref_floor=1e-3,
        reward_min=None,
        reward_max=None,

        # Parallel scoring: use 2 workers to validate process pool path
        n_workers=min(2, os.cpu_count() or 2),
        score_fn_name="example",
        seed=123,
    )

    # GA driver
    ga = MDPEvolutionGA(
        base_mdp=mdp,
        score_fn=None,               # use name-based lookup for parallel workers
        cfg=cfg,
    )

    # Optional: quick check that parallel batch evaluation works on a list
    print("\n=== Parallel evaluate_mdp_list sanity check ===")
    batch = [mdp, clone_mdp_network(mdp), clone_mdp_network(mdp)]
    scores = evaluate_mdp_list(batch, score_fn_name="example", n_workers=cfg.n_workers)
    print("Batch scores:", [round(s, 6) for s in scores])
    assert isinstance(scores, list) and len(scores) == len(batch)

    # Run GA for a few generations
    print("\n=== Run GA for a few generations ===")
    best_mdp, best_score, history = ga.run()
    print("Best score:", best_score)
    print("History (best so far per generation):", [round(x, 6) for x in history])

    # Simple sanity assertions
    assert isinstance(best_mdp, MDPNetwork)
    assert len(history) == cfg.generations + 1  # includes initial evaluation
    print("\nOK: GA ran successfully on FrozenLake 8x8 with parallel scoring.")

    # Save the best MDP as JSON
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"best_mdp_seed{cfg.seed}_score{best_score:.6f}.json")
    best_mdp.export_to_json(out_path)
    print(f"Saved best MDP JSON -> {out_path}")
