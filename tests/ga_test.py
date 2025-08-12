# test_ga_frozenlake8x8.py
# English comments only. Minimal, self-contained sanity test for ga_mdp_search.
# - Builds an MDPNetwork from CustomisedFrozenLakeEnv (8x8, slippery) via get_mdp_network()
# - Runs the GA for a few generations with small population
# - Uses process pool parallel scoring to validate the path
# - Prints best score and simple assertions

import os

# ---- GA system ----
from mdp_network.ga_mdp_search import (
    GAConfig,
    MDPEvolutionGA,
    register_score_fn,
    example_score_fn,
    evaluate_mdp_list,
)

from mdp_network.mdp_network import MDPNetwork

# ---- Custom env that exposes an MDPNetwork ----
from customised_toy_text_envs.customised_frozenlake import CustomisedFrozenLakeEnv


def build_frozenlake_mdp_via_env(map_name: str = "8x8", is_slippery: bool = True) -> MDPNetwork:
    """
    Create an MDPNetwork by constructing CustomisedFrozenLakeEnv and calling get_mdp_network().
    This mirrors the setup used in test_q_learning_stoch_envs.py.
    """
    env = CustomisedFrozenLakeEnv(render_mode=None, map_name=map_name, is_slippery=is_slippery)
    env.reset(seed=0)
    mdp = env.get_mdp_network()
    return mdp


if __name__ == "__main__":
    out_dir = "./outputs/ga_test"

    print("\n=== Build FrozenLake 8x8 (slippery) MDPNetwork via CustomisedFrozenLakeEnv ===")
    mdp = build_frozenlake_mdp_via_env(map_name="8x8", is_slippery=True)
    print(f"|S|={len(mdp.states)}  |A|={mdp.num_actions}  terminals={len(mdp.terminal_states)}")

    # Register example score function under a name for parallel workers.
    register_score_fn("example", example_score_fn)

    # Choose worker counts (>=1). Use 2 if available to validate process pool path.
    workers = min(2, os.cpu_count() or 2)

    # Small GA config for a quick sanity check
    cfg = GAConfig(
        population_size=512,
        generations=64,
        tournament_k=2,
        elitism_num=64,
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

        # Parallel scoring (name-based only)
        n_workers=workers,
        score_fn_name="example",

        # Parallel offspring (process-pool only)
        mutation_n_workers=workers,

        # Distance params (applied in main & workers)
        dist_max_hops=32,
        dist_node_cap=2048,
        dist_weight_eps=1e-6,
        dist_unreachable=1e6,

        seed=123,
    )

    # GA driver
    ga = MDPEvolutionGA(base_mdp=mdp, cfg=cfg)

    # Optional: parallel evaluate_mdp_list sanity check
    print("\n=== Parallel evaluate_mdp_list sanity check ===")
    batch = [mdp, mdp.clone(), mdp.clone()]
    scores = evaluate_mdp_list(batch, score_fn_name="example", n_workers=cfg.n_workers)
    print("Batch scores:", [round(s, 6) for s in scores])
    assert isinstance(scores, list) and len(scores) == len(batch)

    # Run GA
    print("\n=== Run GA for a few generations ===")
    best_mdp, best_score, history = ga.run()
    print("Best score:", best_score)
    print("History (best so far per generation):", [round(x, 6) for x in history])

    # Simple sanity assertions
    assert isinstance(best_mdp, MDPNetwork)
    assert len(history) == cfg.generations + 1  # includes initial evaluation
    print("\nOK: GA ran successfully on FrozenLake 8x8 with parallel scoring & offspring.")

    # Save the best MDP as JSON
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"best_mdp_seed{cfg.seed}_score{best_score:.6f}.json")
    best_mdp.export_to_json(out_path)
    print(f"Saved best MDP JSON -> {out_path}")
