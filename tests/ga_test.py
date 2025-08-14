# test_ga_frozenlake8x8.py
# Minimal NSGA-II sanity test on FrozenLake 8x8 (slippery).
# - Build MDPNetwork from CustomisedFrozenLakeEnv
# - Precompute baseline (policy + occupancy)
# - Run GA with a single multi-output objective: obj_multi_perf (two integrals)
# - Parallel scoring sanity check
# - Print Pareto front and simple assertions

import os

# ---- GA system (NSGA-II version) ----
from mdp_network.ga_mdp_search import (
    GAConfig,
    MDPEvolutionGA,
    register_score_fn,
    evaluate_mdp_objectives,
    obj_multi_perf,           # single multi-output objective
)

from mdp_network.mdp_network import MDPNetwork

# ---- Tables (Serialisable) ----
from mdp_network.mdp_tables import PolicyTable, ValueTable, QTable, q_table_to_policy  # noqa: F401

# ---- Custom env that exposes an MDPNetwork ----
from customised_toy_text_envs.customised_frozenlake import CustomisedFrozenLakeEnv
from mdp_network.solvers import optimal_value_iteration, compute_occupancy_measure


def build_frozenlake_mdp_via_env(map_name: str = "8x8", is_slippery: bool = True) -> MDPNetwork:
    """Create an MDPNetwork via CustomisedFrozenLakeEnv.get_mdp_network()."""
    env = CustomisedFrozenLakeEnv(render_mode=None, map_name=map_name, is_slippery=is_slippery)
    env.reset(seed=0)
    mdp = env.get_mdp_network()
    return mdp


if __name__ == "__main__":
    out_dir = "./outputs/ga_test"

    print("\n=== Build FrozenLake 8x8 (slippery) MDPNetwork via CustomisedFrozenLakeEnv ===")
    mdp = build_frozenlake_mdp_via_env(map_name="8x8", is_slippery=True)
    print(f"|S|={len(mdp.states)}  |A|={mdp.num_actions}  terminals={len(mdp.terminal_states)}")

    workers = os.cpu_count()

    # Register the multi-output objective with per-function constants.
    # You can tweak blend_weight here without touching the GAConfig.
    register_score_fn("obj_multi_perf", obj_multi_perf, const={"blend_weight": 0.99})

    # GA config (kept as requested; only added add_edge_allow_out_of_scope=False)
    cfg = GAConfig(
        population_size=25,
        generations=200,
        tournament_k=2,
        elitism_num=5,
        crossover_rate=0.5,

        allow_self_loops=True,
        min_out_degree=1,
        max_out_degree=4,
        prob_floor=1e-6,
        add_edge_attempts_per_child=10,
        epsilon_new_prob=0.1,
        gamma_sample=1.0,
        gamma_prob=0.0,
        prune_prob_threshold=1e-3,
        prob_tweak_actions_per_child=50,
        prob_pairwise_step=0.05,
        reward_tweak_edges_per_child=0,  # don't change the reward
        reward_k_percent=0.05,
        reward_ref_floor=1e-3,

        # forbid adding transitions to out-of-scope states
        add_edge_allow_out_of_scope=False,

        # Parallel scoring (single multi-output fn)
        n_workers=workers,
        score_fn_names=["obj_multi_perf"],
        score_args=None,
        score_kwargs=None,

        # Parallel offspring
        mutation_n_workers=workers,

        # Distance / scope (used by add_edge_allow_out_of_scope)
        dist_max_hops=6.0,
        dist_node_cap=64,
        dist_weight_eps=1e-6,
        dist_unreachable=1e9,

        # VI / softmax / KL / perf defaults
        vi_gamma=0.99,
        vi_theta=1e-3,
        vi_max_iterations=1000,
        policy_temperature=0.1,
        kl_delta=1e-3,
        perf_numpoints=32,
        perf_gamma=None,          # None -> fallback to vi_gamma
        perf_theta=None,          # None -> fallback to vi_theta
        perf_max_iterations=None, # None -> fallback to vi_max_iterations

        seed=4444,
    )

    # GA driver
    ga = MDPEvolutionGA(base_mdp=mdp, cfg=cfg)

    # ----- Serial PRECOMPUTE on the BASE MDP -----
    V, Q = optimal_value_iteration(
        mdp, gamma=cfg.vi_gamma, theta=cfg.vi_theta, max_iterations=cfg.vi_max_iterations
    )
    base_policy = q_table_to_policy(
        Q, states=list(mdp.states), num_actions=mdp.num_actions, temperature=cfg.policy_temperature
    )
    base_occupancy = compute_occupancy_measure(
        mdp, base_policy, gamma=cfg.vi_gamma, theta=cfg.vi_theta, max_iterations=cfg.vi_max_iterations
    )
    # Broadcast artifacts to workers
    ga.precomputed_artifacts = [base_policy, base_occupancy]

    # ----- Optional: parallel evaluate_mdp_objectives sanity check -----
    print("\n=== Parallel evaluate_mdp_objectives sanity check ===")
    batch = [mdp, mdp.clone(), mdp.clone()]
    obj_vecs = evaluate_mdp_objectives(
        batch,
        score_fn_names=cfg.score_fn_names or [],
        n_workers=cfg.n_workers,
        score_args=cfg.score_args,
        score_kwargs={
            # mirror what GA will auto-inject
            "vi_gamma": cfg.vi_gamma,
            "vi_theta": cfg.vi_theta,
            "vi_max_iterations": cfg.vi_max_iterations,
            "policy_temperature": cfg.policy_temperature,
            "kl_delta": cfg.kl_delta,
            "perf_numpoints": cfg.perf_numpoints,
            "perf_gamma": cfg.perf_gamma if cfg.perf_gamma is not None else cfg.vi_gamma,
            "perf_theta": cfg.perf_theta if cfg.perf_theta is not None else cfg.vi_theta,
            "perf_max_iterations": cfg.perf_max_iterations if cfg.perf_max_iterations is not None else cfg.vi_max_iterations,
        },
        # same baseline artifacts as GA.run()
        precomputed_portables=[base_policy.to_portable(), base_occupancy.to_portable()],
    )
    print("Batch objective vectors:", [[round(x, 6) for x in v] for v in obj_vecs])
    assert isinstance(obj_vecs, list) and len(obj_vecs) == len(batch)
    expected_dim = len(obj_vecs[0])  # multi-output => 2
    assert expected_dim == 2
    assert all(isinstance(v, list) and len(v) == expected_dim for v in obj_vecs)

    # ----- Run GA (NSGA-II) -----
    print("\n=== Run NSGA-II GA for a few generations ===")
    pareto_mdps, pareto_objs, pop, pop_objs = ga.run()

    # Summary
    print(f"\nPareto front size = {len(pareto_mdps)}")
    for i, vec in enumerate(pareto_objs[:10]):
        print(f"  PF[{i}] objs = {[round(x, 6) for x in vec]}")
    print(f"Final population size = {len(pop)}")

    # Sanity assertions
    assert all(isinstance(m, MDPNetwork) for m in pareto_mdps)
    assert len(pareto_objs) == len(pareto_mdps)
    assert all(len(v) == expected_dim for v in pareto_objs)
    assert len(pareto_mdps) >= 1

    print("\nOK: NSGA-II GA ran successfully on FrozenLake 8x8 with parallel scoring & offspring.")

    # Save PF MDPs
    os.makedirs(out_dir, exist_ok=True)
    K = len(pareto_mdps)
    for i in range(K):
        out_path = os.path.join(out_dir, f"pareto_{i}_objs_{'_'.join(f'{v:.4f}' for v in pareto_objs[i])}.json")
        pareto_mdps[i].export_to_json(out_path)
        print(f"Saved PF[{i}] -> {out_path}")
