#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ga_experiment.py (MiniGrid)
# English comments only.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from customised_minigrid_env import (               # your package providing maps & env
    CustomMiniGridEnv,
    list_builtin_maps,
)
from customised_minigrid_env.customised_minigrid_env import plot_minigrid_scalar_overlay, \
    plot_minigrid_scalar_diff_overlay

from experiment_utils.utils import (
    _ensure_dir,
    _str2bool,
    _parse_tuple3,
    _save_json,
    _load_json,
    _wandb_init,
    _wandb_log_image,
    _resolve_args,                                  # your generic resolver
)

from genetic_algorithms.stage_ga import stage_ga
from mdp_network.mdp_network import MDPNetwork
from mdp_network.mdp_tables import q_table_to_policy, create_random_policy
from mdp_network.solvers import (
    optimal_value_iteration,
    compute_occupancy_measure,
    policy_evaluation,
)

from two_stage_cl.stage_train import stage_train


# -------- MiniGrid-specific factories live in main (not inside stage_train) --------
TARGET_FACTORY_PATH = "experiment_utils.env_factories:make_minigrid_target"
SOURCE_FACTORY_PATH = "experiment_utils.env_factories:make_nx_env_from_mdp"


def _build_native_mdp(map_name: str) -> MDPNetwork:
    """MiniGrid-specific helper to build the native MDP from a fixed-layout map name."""
    # Use fixed layout: no random rotations/flips to keep encode/decode consistent.
    env = CustomMiniGridEnv(
        map_name=map_name,
        random_rotate=False,
        random_flip=False,
        any_key_opens_the_door=False,
        networkx_env=None,
    )
    env.reset(seed=0)
    return env.get_mdp_network()


# =============================================================================
# Visualization Stage (Overlays)
# =============================================================================

def stage_visualize(args, run, json_files: List[Path]):
    vis_out = Path(args.outdir) / "vis_minigrid"
    _ensure_dir(vis_out)

    # A native renderer env for consistent geometry (use the same map as training target)
    env = CustomMiniGridEnv(
        map_name=args.map,
        random_rotate=False,
        random_flip=False,
        any_key_opens_the_door=False,
        networkx_env=None,
    )
    env.reset()

    native_mdp = None
    native_occ_random = None
    native_V_opt_greedy = None

    if args.vis_include_native:
        native_mdp = env.get_mdp_network()

        native_policy_rand = create_random_policy(native_mdp)
        native_occ_random = compute_occupancy_measure(
            mdp_network=native_mdp, policy=native_policy_rand,
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )

        _, Q_star = optimal_value_iteration(
            mdp_network=native_mdp, gamma=args.vis_gamma, theta=args.vis_theta,
            max_iterations=args.vis_max_iters, verbose=False,
        )
        native_policy_opt_greedy = q_table_to_policy(
            q_table=Q_star, states=native_mdp.states, num_actions=native_mdp.num_actions,
            mixing=(1.0, 0.0, 0.0), temperature=1.0, tie_tol=args.vis_tie_tol,
        )
        native_V_opt_greedy = policy_evaluation(
            mdp_network=native_mdp, policy=native_policy_opt_greedy,
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )

        native_out = vis_out / f"__native_minigrid__{args.map}__"
        _ensure_dir(native_out)

        # Occupancy — Random
        plot_minigrid_scalar_overlay(
            env=env, value_map=native_occ_random, output_dir=str(native_out),
            filename_prefix="native_minigrid_occupancy_random",
            alpha=args.vis_occ_alpha, annotate=True, dpi=args.vis_dpi,
            font_scale=args.vis_occ_font_scale, cmap_name=args.vis_occ_cmap, gamma=args.vis_occ_gamma,
            min_abs_label=0.0, vmin=0.0, vmax=None,
            title="State Occupancy — Random", cbar_label="Occupancy measure",
            value_format=None,
        )

        # V(s) — Random
        V_rand = policy_evaluation(
            mdp_network=native_mdp, policy=create_random_policy(native_mdp),
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )
        plot_minigrid_scalar_overlay(
            env=env, value_map=V_rand, output_dir=str(native_out),
            filename_prefix="native_minigrid_VALUE_random",
            alpha=args.vis_val_alpha, annotate=True, dpi=args.vis_dpi,
            font_scale=args.vis_val_font_scale, cmap_name=args.vis_val_cmap, gamma=args.vis_val_gamma,
            min_abs_label=0.0, vmin=None, vmax=None,
            title="State Value V(s) — Random", cbar_label="V(s)",
            value_format=None,
        )

        # V(s) — Optimal (greedy)
        plot_minigrid_scalar_overlay(
            env=env, value_map=native_V_opt_greedy, output_dir=str(native_out),
            filename_prefix="native_minigrid_VALUE_optimal_greedy",
            alpha=args.vis_val_alpha, annotate=True, dpi=args.vis_dpi,
            font_scale=args.vis_val_font_scale, cmap_name=args.vis_val_cmap, gamma=args.vis_val_gamma,
            min_abs_label=0.0, vmin=None, vmax=None,
            title="State Value V(s) — Optimal (greedy)", cbar_label="V(s)",
            value_format=None,
        )

        for fn in sorted(native_out.glob("*.png"))[:8]:
            _wandb_log_image(run, f"images/vis/native/{fn.stem}", fn)

    if not json_files:
        print("[VIS] No JSON files; skip visualization on loop-generated MDPs.")
        return

    for jf in json_files:
        cfg = _load_json(jf)
        mdp = MDPNetwork(config_data=cfg)

        stem = jf.stem
        out_dir = vis_out / stem
        _ensure_dir(out_dir)

        # Random policy occupancy
        policy_rand = create_random_policy(mdp)
        occ_rand = compute_occupancy_measure(
            mdp_network=mdp, policy=policy_rand,
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )

        # Optimal Q*
        _, Q_star = optimal_value_iteration(
            mdp_network=mdp, gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )

        # Training policy (mixed): derived from Q* but with exploration
        policy_train_mixed = q_table_to_policy(
            q_table=Q_star, states=mdp.states, num_actions=mdp.num_actions,
            mixing=tuple(args.vis_mix_loop), temperature=args.vis_temperature, tie_tol=args.vis_tie_tol,
        )
        occ_train_mixed = compute_occupancy_measure(
            mdp_network=mdp, policy=policy_train_mixed,
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )

        # Optimal greedy policy
        policy_opt_greedy = q_table_to_policy(
            q_table=Q_star, states=mdp.states, num_actions=mdp.num_actions,
            mixing=(1.0, 0.0, 0.0), temperature=1.0, tie_tol=args.vis_tie_tol,
        )
        V_opt_greedy = policy_evaluation(
            mdp_network=mdp, policy=policy_opt_greedy,
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )

        # Occupancy — random
        plot_minigrid_scalar_overlay(
            env=env, value_map=occ_rand, output_dir=str(out_dir),
            filename_prefix=f"{stem}_occupancy_random",
            alpha=args.vis_occ_alpha, annotate=True, dpi=args.vis_dpi,
            font_scale=args.vis_occ_font_scale, cmap_name=args.vis_occ_cmap, gamma=args.vis_occ_gamma,
            min_abs_label=0.0, vmin=0.0, vmax=None,
            title="State Occupancy", cbar_label="Occupancy measure",
            value_format=None,
        )

        # Occupancy — training policy (mixed)
        mix_suffix = f"mix_g{args.vis_mix_loop[0]:.2f}_s{args.vis_mix_loop[1]:.2f}_u{args.vis_mix_loop[2]:.2f}" + \
                     (f"_T{args.vis_temperature:g}" if args.vis_mix_loop[1] > 0.0 else "")
        plot_minigrid_scalar_overlay(
            env=env, value_map=occ_train_mixed, output_dir=str(out_dir),
            filename_prefix=f"{stem}_occupancy_trainPolicy_{mix_suffix}",
            alpha=args.vis_occ_alpha, annotate=True, dpi=args.vis_dpi,
            font_scale=args.vis_occ_font_scale, cmap_name=args.vis_occ_cmap, gamma=args.vis_occ_gamma,
            min_abs_label=0.0, vmin=0.0, vmax=None,
            title="State Occupancy — Training policy (mixed)", cbar_label="Occupancy measure",
            value_format=None,
        )

        # V(s) — random
        V_rand = policy_evaluation(
            mdp_network=mdp, policy=policy_rand,
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )
        plot_minigrid_scalar_overlay(
            env=env, value_map=V_rand, output_dir=str(out_dir),
            filename_prefix=f"{stem}_VALUE_random",
            alpha=args.vis_val_alpha, annotate=True, dpi=args.vis_dpi,
            font_scale=args.vis_val_font_scale, cmap_name=args.vis_val_cmap, gamma=args.vis_val_gamma,
            min_abs_label=0.0, vmin=None, vmax=None,
            title="State Value V(s) — Random", cbar_label="V(s)",
            value_format=None,
        )

        # V(s) — optimal (greedy)
        plot_minigrid_scalar_overlay(
            env=env, value_map=V_opt_greedy, output_dir=str(out_dir),
            filename_prefix=f"{stem}_VALUE_optimal_greedy",
            alpha=args.vis_val_alpha, annotate=True, dpi=args.vis_dpi,
            font_scale=args.vis_val_font_scale, cmap_name=args.vis_val_cmap, gamma=args.vis_val_gamma,
            min_abs_label=0.0, vmin=None, vmax=None,
            title="State Value V(s) — Optimal (greedy)", cbar_label="V(s)",
            value_format=None,
        )

        # Optional: Cross/Δ pages vs native
        if native_mdp is not None:
            occ_cross_native = compute_occupancy_measure(
                mdp_network=native_mdp, policy=policy_train_mixed,
                gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
            )
            plot_minigrid_scalar_overlay(
                env=env, value_map=occ_cross_native, output_dir=str(out_dir),
                filename_prefix=f"{stem}_occupancy_trainPolicy_on_NATIVE_{mix_suffix}",
                alpha=args.vis_occ_alpha, annotate=True, dpi=args.vis_dpi,
                font_scale=args.vis_occ_font_scale, cmap_name=args.vis_occ_cmap, gamma=args.vis_occ_gamma,
                min_abs_label=0.0, vmin=0.0, vmax=None,
                title="State Occupancy — Training policy on NATIVE", cbar_label="Occupancy measure",
                value_format=None,
            )
            if native_occ_random is not None:
                plot_minigrid_scalar_diff_overlay(
                    env=env,
                    values_a=occ_cross_native,
                    values_b=native_occ_random,
                    output_dir=str(out_dir),
                    filename_prefix=f"{stem}_occupancy_DIFF_trainPolicyMINUS_nativeRandom_{mix_suffix}",
                    alpha=args.vis_occ_alpha, annotate=True, dpi=args.vis_dpi,
                    font_scale=args.vis_occ_font_scale, cmap_name="coolwarm",
                    min_abs_label=0.0, vmin=None, vmax=None,
                    title="Δ State Occupancy (training − native-random)",
                    cbar_label="Δ occupancy (A − B)",
                    value_format="+.2e",
                )

        for fn in sorted(out_dir.glob("*.png"))[:10]:
            _wandb_log_image(run, f"images/vis/{stem}/{fn.stem}", fn)


# =============================================================================
# Argparse / main
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiniGrid: GA → Curriculum → Visualization with W&B (images only).")

    # W&B
    p.add_argument("--outdir", type=str, default="./outputs_minigrid")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default="full-minigrid")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, choices=["online", "offline"], default="online")

    # Pipeline toggles
    p.add_argument("--skip-ga", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-vis", action="store_true")

    # Env
    p.add_argument("--map", type=str, default="door-key",
                   help=f"Built-in maps: {', '.join(list_builtin_maps())}")
    p.add_argument("--max-steps", type=int, default=1000)

    # GA (kept modest; MiniGrid state space is typically larger than Taxi, tune as needed)
    p.add_argument("--ga-pop-size", type=int, default=50)
    p.add_argument("--ga-generations", type=int, default=250)
    p.add_argument("--ga-tournament-k", type=int, default=2)
    p.add_argument("--ga-elitism", type=int, default=10)
    p.add_argument("--ga-crossover", type=float, default=0.5)

    p.add_argument("--ga-allow-self-loops", type=_str2bool, default=True)
    p.add_argument("--ga-min-out-degree", type=int, default=1)
    p.add_argument("--ga-max-out-degree", type=int, default=6)
    p.add_argument("--ga-prob-floor", type=float, default=1e-6)
    p.add_argument("--ga-add-edge-attempts-per-child", type=int, default=50)
    p.add_argument("--ga-epsilon-new-prob", type=float, default=0.1)
    p.add_argument("--ga-gamma-sample", type=float, default=1.0)
    p.add_argument("--ga-gamma-prob", type=float, default=0.0)
    p.add_argument("--ga-prune-prob-threshold", type=float, default=1e-3)
    p.add_argument("--ga-prob-tweak-actions-per-child", type=int, default=100)
    p.add_argument("--ga-prob-pairwise-step", type=float, default=0.05)
    p.add_argument("--ga-reward-tweak-edges-per-child", type=int, default=0)
    p.add_argument("--ga-reward-k-percent", type=float, default=0.05)
    p.add_argument("--ga-reward-ref-floor", type=float, default=1e-3)
    p.add_argument("--ga-add-edge-allow-out-of-scope", type=_str2bool, default=False)

    p.add_argument("--ga-workers", type=int, default=0, help="0=auto(cpu_count)")
    p.add_argument("--ga-sanity-batch", type=int, default=0)

    p.add_argument("--ga-dist-max-hops", type=int, default=10)
    p.add_argument("--ga-dist-node-cap", type=int, default=256)
    p.add_argument("--ga-dist-weight-eps", type=float, default=1e-6)
    p.add_argument("--ga-dist-unreachable", type=float, default=1e9)

    p.add_argument("--ga-vi-gamma", type=float, default=0.99)
    p.add_argument("--ga-vi-theta", type=float, default=1e-3)
    p.add_argument("--ga-vi-max-iters", type=int, default=1000)
    p.add_argument("--ga-policy-mix", type=_parse_tuple3, default=(0.9, 0.0, 0.1))
    p.add_argument("--ga-policy-temperature", type=float, default=0.01)
    p.add_argument("--ga-tie-tol", type=float, default=1e-2)
    p.add_argument("--ga-blend-weight", type=float, default=0.8)
    p.add_argument("--ga-perf-numpoints", type=int, default=10)
    p.add_argument("--ga-perf-gamma", type=float, default=0.99)
    p.add_argument("--ga-perf-theta", type=float, default=1e-3)
    p.add_argument("--ga-perf-max-iters", type=int, default=1000)
    p.add_argument("--ga-seed", type=int, default=0)

    # Training (flattened agent kwargs)
    p.add_argument("--agent-learning-rate", type=float, default=0.1)
    p.add_argument("--agent-gamma", type=float, default=0.99)
    p.add_argument("--agent-policy-mix", type=_parse_tuple3, default=(0.9, 0.0, 0.1),
                   help="Tuple 'g,s,u' for (greedy, softmax, uniform)")
    p.add_argument("--agent-temperature", type=float, default=0.01,
                   help="Used only if softmax weight > 0")
    p.add_argument("--agent-tie-tol", type=float, default=1e-2)
    p.add_argument("--agent-verbose", type=int, default=0)

    p.add_argument("--phase-steps", type=str, default="20000, 130000")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--n-eval-episodes", type=int, default=10)

    # Train seeds as COUNT -> seeds [0..N-1]
    p.add_argument("--train-seeds", type=int, default=50, help="Use N to get seeds [0..N-1].")
    p.add_argument("--train-workers", type=int, default=0)

    p.add_argument("--eval-seed-base-target", type=int, default=0)
    p.add_argument("--eval-seed-base-source", type=int, default=0)
    p.add_argument("--json-dir", type=str, default="")
    p.add_argument("--json-max", type=int, default=0)

    # Visualization
    p.add_argument("--vis-include-native", type=_str2bool, default=True)
    p.add_argument("--vis-dpi", type=int, default=200)
    p.add_argument("--vis-gamma", type=float, default=0.99)
    p.add_argument("--vis-theta", type=float, default=1e-6)
    p.add_argument("--vis-max-iters", type=int, default=1000)
    p.add_argument("--vis-temperature", type=float, default=1.0)
    p.add_argument("--vis-mix-loop", type=_parse_tuple3, default=(0.9, 0.0, 0.1))
    p.add_argument("--vis-tie-tol", type=float, default=1e-2)

    p.add_argument("--vis-occ-alpha", type=float, default=0.55)
    p.add_argument("--vis-occ-font-scale", type=float, default=0.11)
    p.add_argument("--vis-occ-cmap", type=str, default="magma")
    p.add_argument("--vis-occ-gamma", type=float, default=1.0)

    p.add_argument("--vis-val-alpha", type=float, default=0.60)
    p.add_argument("--vis-val-font-scale", type=float, default=0.11)
    p.add_argument("--vis-val-cmap", type=str, default="viridis")
    p.add_argument("--vis-val-gamma", type=float, default=1.0)
    return p


def main():
    parser = build_arg_parser()
    args = _resolve_args(parser)   # your resolver builds agent_kwargs, seeds, phase_steps, etc.

    # Validate map name early for clearer error messages
    if args.map not in set(list_builtin_maps()):
        raise ValueError(f"Unknown --map '{args.map}'. Available: {', '.join(list_builtin_maps())}")

    run = _wandb_init(args)
    _save_json(Path(args.outdir) / "meta" / "config.json", {k: getattr(args, k) for k in vars(args)})

    # GA Stage
    if args.json_dir:
        json_dir = Path(args.json_dir)
        json_files = sorted(json_dir.glob("*.json"))
        if args.json_max > 0:
            json_files = json_files[:args.json_max]
        print(f"[MAIN] Using external JSON dir: {json_dir} ({len(json_files)} files).")
    else:
        if not args.skip_ga:
            print(f"[GA] Building native MiniGrid MDP for map '{args.map}'…")
            base_mdp = _build_native_mdp(args.map)
            json_files = stage_ga(args, run, base_mdp)
        else:
            mdp_out_dir = Path(args.outdir) / "ga" / "mdps"
            json_files = sorted(mdp_out_dir.glob("*.json"))
            if args.json_max > 0:
                json_files = json_files[:args.json_max]
            print(f"[MAIN] GA skipped; using {len(json_files)} JSON from {mdp_out_dir}.")

    if args.json_max > 0:
        json_files = json_files[:args.json_max]

    # Training Stage (environment-agnostic)
    if not args.skip_train and json_files:
        print("[Training] Curriculum starting…")
        _ = stage_train(
            args=args,
            run=run,
            json_files=json_files,
            target_factory_path=TARGET_FACTORY_PATH,
            target_factory_kwargs=dict(map_name=args.map, max_steps=int(args.max_steps)),
            source_factory_path=SOURCE_FACTORY_PATH,
            source_env_base_kwargs=dict(max_steps=int(args.max_steps)),
            phase_steps=args.phase_steps,
            eval_seed_base_target=args.eval_seed_base_target,
            eval_seed_base_source=args.eval_seed_base_source,
            target_label=f"Target[{args.map}]",
            source_label="Source-A",
        )
    else:
        print("[MAIN] Training skipped or no JSON files; skipping trainer stage.")

    # Visualization Stage
    if not args.skip_vis:
        stage_visualize(args, run, json_files)
    else:
        print("[MAIN] Visualization skipped.")

    print("\n[MAIN] All done.")
    run.finish()


if __name__ == "__main__":
    main()
