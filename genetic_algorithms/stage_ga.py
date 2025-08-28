from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import List

from experiment_utils.utils import ensure_dir
from genetic_algorithms.ga_mdp_search import (
    register_score_fn,
    obj_multi_perf,
    GAConfig,
    MDPEvolutionGA,
    evaluate_mdp_objectives,
)
from mdp_network import MDPNetwork
from mdp_network.mdp_tables import q_table_to_policy
from mdp_network.solvers import optimal_value_iteration, compute_occupancy_measure


def stage_ga(args, run, mdp: MDPNetwork) -> List[Path]:
    """Run GA given a pre-built base MDP (environment-agnostic)."""
    out_dir = Path(args.outdir) / "ga"
    mdp_out_dir = out_dir / "mdps"
    ensure_dir(mdp_out_dir)

    if args.skip_ga and mdp_out_dir.exists():
        files = sorted(mdp_out_dir.glob("*.json"))
        if files:
            print(f"[GA] Skipped (existing {len(files)} JSON found).")
            return files

    # NOTE: The base MDP is now supplied by the caller; no environment binding here.
    register_score_fn("obj_multi_perf", obj_multi_perf)

    workers = args.ga_workers or (os.cpu_count() or 1)
    cfg = GAConfig(
        population_size=args.ga_pop_size,
        generations=args.ga_generations,
        tournament_k=args.ga_tournament_k,
        elitism_num=args.ga_elitism,
        crossover_rate=args.ga_crossover,
        allow_self_loops=args.ga_allow_self_loops,
        min_out_degree=args.ga_min_out_degree,
        max_out_degree=args.ga_max_out_degree,
        prob_floor=args.ga_prob_floor,
        add_edge_attempts_per_child=args.ga_add_edge_attempts_per_child,
        epsilon_new_prob=args.ga_epsilon_new_prob,
        gamma_sample=args.ga_gamma_sample,
        gamma_prob=args.ga_gamma_prob,
        prune_prob_threshold=args.ga_prune_prob_threshold,
        prob_tweak_actions_per_child=args.ga_prob_tweak_actions_per_child,
        prob_pairwise_step=args.ga_prob_pairwise_step,
        reward_tweak_edges_per_child=args.ga_reward_tweak_edges_per_child,
        reward_k_percent=args.ga_reward_k_percent,
        reward_ref_floor=args.ga_reward_ref_floor,
        add_edge_allow_out_of_scope=args.ga_add_edge_allow_out_of_scope,
        n_workers=workers,
        score_fn_names=["obj_multi_perf"],
        score_args=None,
        score_kwargs={
            "policy_mixing": tuple(args.ga_policy_mix),
            "policy_tie_tol": args.ga_tie_tol,
            "blend_weight": args.ga_blend_weight,
        },
        mutation_n_workers=workers,
        dist_max_hops=args.ga_dist_max_hops,
        dist_node_cap=args.ga_dist_node_cap,
        dist_weight_eps=args.ga_dist_weight_eps,
        dist_unreachable=args.ga_dist_unreachable,
        vi_gamma=args.ga_vi_gamma,
        vi_theta=args.ga_vi_theta,
        vi_max_iterations=args.ga_vi_max_iters,
        policy_temperature=args.ga_policy_temperature,
        perf_numpoints=args.ga_perf_numpoints,
        perf_gamma=args.ga_perf_gamma,
        perf_theta=args.ga_perf_theta,
        perf_max_iterations=args.ga_perf_max_iters,
        seed=args.ga_seed,
    )

    run.config.update({"ga_config": asdict(cfg)}, allow_val_change=True)

    print("[GA] Precomputing baseline policy & occupancy…")
    _, Q = optimal_value_iteration(
        mdp,
        gamma=cfg.vi_gamma,
        theta=cfg.vi_theta,
        max_iterations=cfg.vi_max_iterations,
    )
    base_policy = q_table_to_policy(
        Q,
        states=list(mdp.states),
        num_actions=mdp.num_actions,
        mixing=tuple(args.ga_policy_mix),
        temperature=cfg.policy_temperature,
        tie_tol=args.ga_tie_tol,
    )
    base_occupancy = compute_occupancy_measure(
        mdp,
        base_policy,
        gamma=cfg.vi_gamma,
        theta=cfg.vi_theta,
        max_iterations=cfg.vi_max_iterations,
    )

    ga = MDPEvolutionGA(base_mdp=mdp, cfg=cfg, wb_run=run)
    ga.precomputed_artifacts = [base_policy, base_occupancy]

    if args.ga_sanity_batch > 0:
        print("[GA] Sanity check evaluate_mdp_objectives…")
        batch = [mdp] + [mdp.clone() for _ in range(args.ga_sanity_batch - 1)]
        obj_vecs = evaluate_mdp_objectives(
            batch,
            score_fn_names=cfg.score_fn_names or [],
            n_workers=cfg.n_workers,
            score_args=cfg.score_args,
            score_kwargs={
                "vi_gamma": cfg.vi_gamma,
                "vi_theta": cfg.vi_theta,
                "vi_max_iterations": cfg.vi_max_iterations,
                "policy_temperature": cfg.policy_temperature,
                "policy_mixing": tuple(args.ga_policy_mix),
                "policy_tie_tol": args.ga_tie_tol,
                "perf_numpoints": cfg.perf_numpoints,
                "perf_gamma": cfg.perf_gamma,
                "perf_theta": cfg.perf_theta,
                "perf_max_iterations": cfg.perf_max_iterations,
            },
            precomputed_portables=[
                base_policy.to_portable(),
                base_occupancy.to_portable(),
            ],
        )
        print(
            "  Batch objective vectors (head):",
            [[round(x, 6) for x in v] for v in obj_vecs[:3]],
        )

    print("[GA] Running NSGA-II…")
    pareto_mdps, pareto_objs, pop, _ = ga.run()
    print(f"[GA] Pareto front size = {len(pareto_mdps)}, population size = {len(pop)}")

    saved = []
    ensure_dir(mdp_out_dir)
    for i, m in enumerate(pareto_mdps):
        tag = "_".join(f"{v:.4f}" for v in pareto_objs[i])
        out_path = mdp_out_dir / f"pareto_{i}_objs_{tag}.json"
        m.export_to_json(str(out_path))
        saved.append(out_path)
        print(f"[GA] Saved PF[{i}] -> {out_path.name}")

    return saved
