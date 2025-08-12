# test_ga_frozenlake8x8.py
# English comments only. Minimal, self-contained sanity test for NSGA-II GA.
# - Builds an MDPNetwork from CustomisedFrozenLakeEnv (8x8, slippery)
# - Runs the GA (NSGA-II) for a few generations with small population
# - Uses process pool parallel scoring to validate the path
# - Prints Pareto front info and simple assertions

import os

# ---- GA system (NSGA-II version) ----
from mdp_network.ga_mdp_search import (
    GAConfig,
    MDPEvolutionGA,
    register_score_fn,
    evaluate_mdp_objectives,
    obj_reward_sum,
    obj_sparse_structure,
)

from mdp_network.mdp_network import MDPNetwork

# ---- Custom env that exposes an MDPNetwork ----
from customised_toy_text_envs.customised_frozenlake import CustomisedFrozenLakeEnv


def build_frozenlake_mdp_via_env(map_name: str = "8x8", is_slippery: bool = True) -> MDPNetwork:
    """Create an MDPNetwork by constructing CustomisedFrozenLakeEnv and calling get_mdp_network()."""
    env = CustomisedFrozenLakeEnv(render_mode=None, map_name=map_name, is_slippery=is_slippery)
    env.reset(seed=0)
    mdp = env.get_mdp_network()
    return mdp


if __name__ == "__main__":
    out_dir = "./outputs/ga_test"

    print("\n=== Build FrozenLake 8x8 (slippery) MDPNetwork via CustomisedFrozenLakeEnv ===")
    mdp = build_frozenlake_mdp_via_env(map_name="8x8", is_slippery=True)
    print(f"|S|={len(mdp.states)}  |A|={mdp.num_actions}  terminals={len(mdp.terminal_states)}")

    # Register objective functions under names for parallel workers.
    register_score_fn("obj_reward_sum", obj_reward_sum)
    register_score_fn("obj_sparse_structure", obj_sparse_structure)

    # Choose worker counts (>=1). Use 2 if available to validate process pool path.
    workers = min(2, os.cpu_count() or 2)

    # Smallish GA config for a quick sanity check (feel free to tweak sizes)
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
        prune_prob_threshold=1e-3,
        prob_tweak_actions_per_child=50,
        prob_pairwise_step=0.05,
        reward_tweak_edges_per_child=5,
        reward_k_percent=0.05,
        reward_ref_floor=1e-3,
        reward_min=None,
        reward_max=None,

        # Parallel scoring (multi-objective)
        n_workers=workers,
        score_fn_names=["obj_reward_sum", "obj_sparse_structure"],
        score_args=None,
        score_kwargs=None,

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

    # Optional: parallel evaluate_mdp_objectives sanity check
    print("\n=== Parallel evaluate_mdp_objectives sanity check ===")
    batch = [mdp, mdp.clone(), mdp.clone()]
    obj_vecs = evaluate_mdp_objectives(
        batch,
        score_fn_names=cfg.score_fn_names or [],
        n_workers=cfg.n_workers,
        score_args=cfg.score_args,
        score_kwargs=cfg.score_kwargs,
        precomputed_portables=None,
    )
    print("Batch objective vectors:", [[round(x, 6) for x in v] for v in obj_vecs])
    assert isinstance(obj_vecs, list) and len(obj_vecs) == len(batch)
    assert all(isinstance(v, list) and len(v) == len(cfg.score_fn_names or []) for v in obj_vecs)

    # Run GA (NSGA-II)
    print("\n=== Run NSGA-II GA for a few generations ===")
    pareto_mdps, pareto_objs, pop, pop_objs = ga.run()

    # Print a brief summary
    print(f"\nPareto front size = {len(pareto_mdps)}")
    for i, vec in enumerate(pareto_objs[:10]):
        print(f"  PF[{i}] objs = {[round(x, 6) for x in vec]}")
    print(f"Final population size = {len(pop)}")

    # Simple sanity assertions
    assert all(isinstance(m, MDPNetwork) for m in pareto_mdps)
    assert len(pareto_objs) == len(pareto_mdps)
    assert all(len(v) == len(cfg.score_fn_names or []) for v in pareto_objs)

    print("\nOK: NSGA-II GA ran successfully on FrozenLake 8x8 with parallel scoring & offspring.")

    # Save the Pareto front MDPs as JSON (save top K or all)
    os.makedirs(out_dir, exist_ok=True)
    K = min(10, len(pareto_mdps))
    for i in range(K):
        out_path = os.path.join(out_dir, f"pareto_{i}_objs_{'_'.join(f'{v:.4f}' for v in pareto_objs[i])}.json")
        pareto_mdps[i].export_to_json(out_path)
        print(f"Saved PF[{i}] -> {out_path}")
