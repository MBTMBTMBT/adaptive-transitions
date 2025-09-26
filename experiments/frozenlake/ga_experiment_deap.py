#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_full_experiment.py

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import ray
from ray.actor import ActorHandle

from customised_toy_text_envs.customised_frozenlake import (
    CustomisedFrozenLakeEnv,
    plot_frozenlake_transition_overlays,
    plot_frozenlake_scalar_overlay,
    plot_frozenlake_scalar_diff_overlay,
)

from experiment_utils.utils import (
    ensure_dir,
    str2bool,
    parse_tuple3,
    save_json,
    load_json,
    resolve_args,
    _timestamped_outdir,
)

from experiment_utils.wandb_utils import WandbActor, capture_prints_to_wandb
# New GA entry with schedules (DEAP + Ray)
from genetic_algorithms.ga_mdp_deap import run_ga_deap_ray

from mdp_network.mdp_network import MDPNetwork
from mdp_network.mdp_tables import q_table_to_policy, create_random_policy
from mdp_network.solvers import (
    optimal_value_iteration,
    compute_occupancy_measure,
    policy_evaluation,
)
from two_stage_cl.tabular_curriculum_trainer_ray import run_curriculum

# -------- FrozenLake-specific factories --------
TARGET_FACTORY_PATH = "experiment_utils.env_factories:make_frozenlake"
SOURCE_FACTORY_PATH = "experiment_utils.env_factories:make_frozenlake"

# ---- Group-level config (objectives + schedules) ----
# Keep behavior equivalent to old pipeline:
# - fitness: every generation optimize the group's objective_keys (gens=1)
# - eval: every generation monitor all metrics ("*"), and also at begin/end
OBJECTIVE_GROUPS: Dict[str, Dict[str, Any]] = {
    "auc": {
        "objective_keys": ["auc_p2"],
        "fitness_schedule": [{"gens": 1, "keys": ["auc_p2"]}],
        "eval_schedule": [{"every": 1, "offset": 0, "keys": "*"}],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "int": {
        "objective_keys": ["int_source_to_target"],
        "fitness_schedule": [{"gens": 1, "keys": ["int_source_to_target"]}],
        "eval_schedule": [{"every": 1, "offset": 0, "keys": "*"}],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "auc_auc": {
        "objective_keys": ["auc_p1_source", "auc_p2"],
        "fitness_schedule": [{"gens": 1, "keys": ["auc_p1_source", "auc_p2"]}],
        "eval_schedule": [{"every": 1, "offset": 0, "keys": "*"}],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "int_int": {
        "objective_keys": ["int_rand_to_source_on_source", "int_source_to_target"],
        "fitness_schedule": [{"gens": 1, "keys": ["int_rand_to_source_on_source", "int_source_to_target"]}],
        "eval_schedule": [{"every": 1, "offset": 0, "keys": "*"}],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "auc_kl": {
        "objective_keys": ["auc_p1_source", "minus_target_kl"],
        "fitness_schedule": [{"gens": 1, "keys": ["auc_p1_source", "minus_target_kl"]}],
        "eval_schedule": [{"every": 1, "offset": 0, "keys": "*"}],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "int_kl": {
        "objective_keys": ["int_rand_to_source_on_source", "minus_target_kl"],
        "fitness_schedule": [{"gens": 1, "keys": ["int_rand_to_source_on_source", "minus_target_kl"]}],
        "eval_schedule": [{"every": 1, "offset": 0, "keys": "*"}],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "kl_kl": {
        "objective_keys": ["minus_control_kl", "minus_target_kl"],
        "fitness_schedule": [{"gens": 1, "keys": ["minus_control_kl", "minus_target_kl"]}],
        "eval_schedule": [{"every": 1, "offset": 0, "keys": "*"}],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "auc_value_diff": {
        "objective_keys": ["auc_p1_source", "minus_value_diff"],
        "fitness_schedule": [{"gens": 1, "keys": ["auc_p1_source", "minus_value_diff"]}],
        "eval_schedule": [{"every": 1, "offset": 0, "keys": "*"}],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "auc_auc_kl": {
        "objective_keys": ["auc_p1_source", "auc_p2", "minus_target_kl"],
        "fitness_schedule": [{"gens": 1, "keys": ["auc_p1_source", "auc_p2", "minus_target_kl"]}],
        "eval_schedule": [{"every": 1, "offset": 0, "keys": "*"}],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "auc_auc_value_diff": {
        "objective_keys": ["auc_p1_source", "auc_p2", "minus_value_diff"],
        "fitness_schedule": [{"gens": 1, "keys": ["auc_p1_source", "auc_p2", "minus_value_diff"]}],
        "eval_schedule": [{"every": 1, "offset": 0, "keys": "*"}],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "comb_int_int": {
        "objective_keys": [
            "int_rand_to_source_on_source", "int_source_to_target",
        ],
        "fitness_schedule": [
            {"gens": 1, "keys": ["int_rand_to_source_on_source", "int_source_to_target",]},
        ],
        "eval_schedule": [
            {"gens": 9, "keys": [
                "int_rand_to_source_on_source", "int_rand_to_source", "int_source_to_target",
                "control_kl", "target_kl", "minus_control_kl", "minus_target_kl",
                "value_diff", "minus_value_diff"
            ]},
            {"gens": 1, "keys": "*"},
        ],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "comb_int_kl": {
        "objective_keys": [
            "int_rand_to_source_on_source", "minus_target_kl",
        ],
        "fitness_schedule": [
            {"gens": 1, "keys": ["int_rand_to_source_on_source", "minus_target_kl",]},
        ],
        "eval_schedule": [
            {"gens": 9, "keys": [
                "int_rand_to_source_on_source", "int_rand_to_source", "int_source_to_target",
                "control_kl", "target_kl", "minus_control_kl", "minus_target_kl",
                "value_diff", "minus_value_diff"
            ]},
            {"gens": 1, "keys": "*"},
        ],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "comb_auc__int": {
        "objective_keys": [
            "auc_p2",
            "int_source_to_target",
        ],
        "fitness_schedule": [
            {"gens": 9, "keys": ["int_source_to_target"]},
            {"gens": 1, "keys": ["auc_p2"]},
        ],
        "eval_schedule": [
            {"gens": 9, "keys": [
                "int_rand_to_source_on_source", "int_rand_to_source", "int_source_to_target",
                "control_kl", "target_kl", "minus_control_kl", "minus_target_kl",
                "value_diff", "minus_value_diff"
            ]},
            {"gens": 1, "keys": "*"},
        ],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "comb_auc_auc__int_int": {
        "objective_keys": [
            "auc_p1_source", "auc_p2",
            "int_rand_to_source_on_source", "int_source_to_target",
        ],
        "fitness_schedule": [
            {"gens": 9, "keys": ["int_rand_to_source_on_source", "int_source_to_target"]},
            {"gens": 1, "keys": ["auc_p1_source", "auc_p2"]},
        ],
        "eval_schedule": [
            {"gens": 9, "keys": [
                "int_rand_to_source_on_source", "int_rand_to_source", "int_source_to_target",
                "control_kl", "target_kl", "minus_control_kl", "minus_target_kl",
                "value_diff", "minus_value_diff"
            ]},
            {"gens": 1,  "keys": "*"},
        ],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "comb_auc_kl__int_kl": {
        "objective_keys": [
            "auc_p1_source",
            "int_rand_to_source_on_source",
            "minus_target_kl",
        ],
        "fitness_schedule": [
            {"gens": 9, "keys": ["int_rand_to_source_on_source", "minus_target_kl"]},
            {"gens": 1, "keys": ["auc_p1_source", "minus_target_kl"]},
        ],
        "eval_schedule": [
            {"gens": 9, "keys": [
                "int_rand_to_source_on_source", "int_rand_to_source", "int_source_to_target",
                "control_kl", "target_kl", "minus_control_kl", "minus_target_kl",
                "value_diff", "minus_value_diff"
            ]},
            {"gens": 1, "keys": "*"},
        ],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "comb_auc_valdiff__int_valdiff": {
        "objective_keys": [
            "auc_p1_source",
            "int_rand_to_source_on_source",
            "minus_value_diff",
        ],
        "fitness_schedule": [
            {"gens": 9, "keys": ["int_rand_to_source_on_source", "minus_value_diff"]},
            {"gens": 1, "keys": ["auc_p1_source", "minus_value_diff"]},
        ],
        "eval_schedule": [
            {"gens": 9, "keys": [
                "int_rand_to_source_on_source", "int_rand_to_source", "int_source_to_target",
                "control_kl", "target_kl", "minus_control_kl", "minus_target_kl",
                "value_diff", "minus_value_diff"
            ]},
            {"gens": 1, "keys": "*"},
        ],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "comb_auc_auc_kl__int_int_kl": {
        "objective_keys": [
            "auc_p1_source", "auc_p2",
            "int_rand_to_source_on_source", "int_source_to_target",
            "minus_target_kl",
        ],
        "fitness_schedule": [
            {"gens": 9, "keys": ["int_rand_to_source_on_source", "int_source_to_target", "minus_target_kl"]},
            {"gens": 1, "keys": ["auc_p1_source", "auc_p2", "minus_target_kl"]},
        ],
        "eval_schedule": [
            {"gens": 9, "keys": [
                "int_rand_to_source_on_source", "int_rand_to_source", "int_source_to_target",
                "control_kl", "target_kl", "minus_control_kl", "minus_target_kl",
                "value_diff", "minus_value_diff"
            ]},
            {"gens": 1, "keys": "*"},
        ],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
    "comb_auc_auc_valdiff__int_int_valdiff": {
        "objective_keys": [
            "auc_p1_source", "auc_p2",
            "int_rand_to_source_on_source", "int_source_to_target",
            "minus_value_diff",
        ],
        "fitness_schedule": [
            {"gens": 9, "keys": ["int_rand_to_source_on_source", "int_source_to_target", "minus_value_diff"]},
            {"gens": 1, "keys": ["auc_p1_source", "auc_p2", "minus_value_diff"]},
        ],
        "eval_schedule": [
            {"gens": 9, "keys": [
                "int_rand_to_source_on_source", "int_rand_to_source", "int_source_to_target",
                "control_kl", "target_kl", "minus_control_kl", "minus_target_kl",
                "value_diff", "minus_value_diff"
            ]},
            {"gens": 1, "keys": "*"},
        ],
        "eval_at_begin": "*",
        "eval_at_end": "*",
    },
}

# Only upload "best-of" summaries for these keys on generations when they were actually computed.
SELECTED_MAX_METRIC_KEYS = ["auc_p2", "minus_target_kl", "minus_value_diff"]


def _build_native_mdp(map_name: str, slippery: bool) -> MDPNetwork:
    """Build the native FrozenLake MDP from the environment."""
    env = CustomisedFrozenLakeEnv(
        render_mode=None, map_name=map_name, is_slippery=slippery
    )
    env.reset(seed=0)
    return env.get_mdp_network()


# =============================================================================
# Visualization Stage (Overlays) — logs images through WandbWriter
# =============================================================================


def stage_visualize(args, wandb_actor: Optional[ActorHandle], json_files: List[Path]):
    vis_out = Path(args.outdir) / "vis"
    ensure_dir(vis_out)

    env = CustomisedFrozenLakeEnv(
        render_mode="rgb_array", map_name=args.map, is_slippery=bool(args.slippery)
    )
    env.reset()

    native_mdp = None
    native_occ_random = None
    native_V_opt_greedy = None

    if args.vis_include_native:
        native_mdp = env.get_mdp_network()

        native_policy_rand = create_random_policy(native_mdp)
        native_occ_random = compute_occupancy_measure(
            mdp_network=native_mdp,
            policy=native_policy_rand,
            gamma=args.vis_gamma,
            theta=args.vis_theta,
            max_iterations=args.vis_max_iters,
            verbose=False,
        )

        _, Q_star = optimal_value_iteration(
            mdp_network=native_mdp,
            gamma=args.vis_gamma,
            theta=args.vis_theta,
            max_iterations=args.vis_max_iters,
            verbose=False,
        )
        native_policy_opt_greedy = q_table_to_policy(
            q_table=Q_star,
            states=native_mdp.states,
            num_actions=native_mdp.num_actions,
            mixing=(1.0, 0.0, 0.0),
            temperature=1.0,
            tie_tol=args.vis_tie_tol,
        )
        native_V_opt_greedy = policy_evaluation(
            mdp_network=native_mdp,
            policy=native_policy_opt_greedy,
            gamma=args.vis_gamma,
            theta=args.vis_theta,
            max_iterations=args.vis_max_iters,
            verbose=False,
        )

        native_out = vis_out / "__native_frozenlake__"
        ensure_dir(native_out)

        plot_frozenlake_transition_overlays(
            env=env,
            mdp=native_mdp,
            output_dir=str(native_out),
            filename_prefix="native_frozenlake",
            min_prob=args.vis_min_prob,
            alpha=args.vis_alpha,
            annotate=True,
            show_self_loops=args.vis_show_self_loops,
            dpi=args.vis_dpi,
        )

        plot_frozenlake_scalar_overlay(
            env=env,
            value_map=native_occ_random,
            output_dir=str(native_out),
            filename_prefix="native_frozenlake_occupancy_random",
            alpha=args.vis_occ_alpha,
            annotate=True,
            dpi=args.vis_dpi,
            target_cell_px=args.vis_occ_cell_px,
            font_scale=args.vis_occ_font_scale,
            cmap_name=args.vis_occ_cmap,
            gamma=args.vis_occ_gamma,
            min_abs_label=0.0,
            vmin=0.0,
            vmax=None,
            title="State Occupancy — Random",
            cbar_label="Occupancy measure",
            value_format=None,
        )

        V_rand = policy_evaluation(
            mdp_network=native_mdp,
            policy=create_random_policy(native_mdp),
            gamma=args.vis_gamma,
            theta=args.vis_theta,
            max_iterations=args.vis_max_iters,
            verbose=False,
        )
        plot_frozenlake_scalar_overlay(
            env=env,
            value_map=V_rand,
            output_dir=str(native_out),
            filename_prefix="native_frozenlake_VALUE_random",
            alpha=args.vis_val_alpha,
            annotate=True,
            dpi=args.vis_dpi,
            target_cell_px=args.vis_val_cell_px,
            font_scale=args.vis_val_font_scale,
            cmap_name=args.vis_val_cmap,
            gamma=args.vis_val_gamma,
            min_abs_label=0.0,
            vmin=None,
            vmax=None,
            title="State Value V(s) — Random",
            cbar_label="V(s)",
            value_format=None,
        )
        plot_frozenlake_scalar_overlay(
            env=env,
            value_map=native_V_opt_greedy,
            output_dir=str(native_out),
            filename_prefix="native_frozenlake_VALUE_optimal_greedy",
            alpha=args.vis_val_alpha,
            annotate=True,
            dpi=args.vis_dpi,
            target_cell_px=args.vis_val_cell_px,
            font_scale=args.vis_val_font_scale,
            cmap_name=args.vis_val_cmap,
            gamma=args.vis_val_gamma,
            min_abs_label=0.0,
            vmin=None,
            vmax=None,
            title="State Value V(s) — Optimal (greedy)",
            cbar_label="V(s)",
            value_format=None,
        )

        # Upload a small sample of native images to W&B
        if wandb_actor is not None:
            for fn in sorted(native_out.glob("*.png"))[:6]:
                wandb_actor.log_image.remote(f"images/vis/native/{fn.stem}", str(fn))

    if not json_files:
        print("[VIS] No JSON files; skip visualization on loops.")
        return

    for jf in json_files:
        cfg = load_json(jf)
        mdp = MDPNetwork(config_data=cfg)

        stem = jf.stem
        out_dir = vis_out / stem
        ensure_dir(out_dir)

        plot_frozenlake_transition_overlays(
            env=env,
            mdp=mdp,
            output_dir=str(out_dir),
            filename_prefix=stem,
            min_prob=args.vis_min_prob,
            alpha=args.vis_alpha,
            annotate=True,
            show_self_loops=args.vis_show_self_loops,
            dpi=args.vis_dpi,
        )

        policy_rand = create_random_policy(mdp)
        occ_rand = compute_occupancy_measure(
            mdp_network=mdp,
            policy=policy_rand,
            gamma=args.vis_gamma,
            theta=args.vis_theta,
            max_iterations=args.vis_max_iters,
            verbose=False,
        )

        _, Q_star = optimal_value_iteration(
            mdp_network=mdp,
            gamma=args.vis_gamma,
            theta=args.vis_theta,
            max_iterations=args.vis_max_iters,
            verbose=False,
        )

        # Mixed policy derived from optimal Q: this is the training policy with exploration.
        policy_opt_mixed = q_table_to_policy(
            q_table=Q_star,
            states=mdp.states,
            num_actions=mdp.num_actions,
            mixing=tuple(args.vis_mix_loop),
            temperature=args.vis_temperature,
            tie_tol=args.vis_tie_tol,
        )
        occ_opt_mixed = compute_occupancy_measure(
            mdp_network=mdp,
            policy=policy_opt_mixed,
            gamma=args.vis_gamma,
            theta=args.vis_theta,
            max_iterations=args.vis_max_iters,
            verbose=False,
        )

        # Pure greedy policy from optimal Q (kept as "optimal (greedy)").
        policy_opt_greedy = q_table_to_policy(
            q_table=Q_star,
            states=mdp.states,
            num_actions=mdp.num_actions,
            mixing=(1.0, 0.0, 0.0),
            temperature=1.0,
            tie_tol=args.vis_tie_tol,
        )
        V_opt_greedy = policy_evaluation(
            mdp_network=mdp,
            policy=policy_opt_greedy,
            gamma=args.vis_gamma,
            theta=args.vis_theta,
            max_iterations=args.vis_max_iters,
            verbose=False,
        )

        plot_frozenlake_scalar_overlay(
            env=env,
            value_map=occ_rand,
            output_dir=str(out_dir),
            filename_prefix=f"{stem}_occupancy_random",
            alpha=args.vis_occ_alpha,
            annotate=True,
            dpi=args.vis_dpi,
            target_cell_px=args.vis_occ_cell_px,
            font_scale=args.vis_occ_font_scale,
            cmap_name=args.vis_occ_cmap,
            gamma=args.vis_occ_gamma,
            min_abs_label=0.0,
            vmin=0.0,
            vmax=None,
            title="State Occupancy",
            cbar_label="Occupancy measure",
            value_format=None,
        )

        mix_suffix = (
            f"mix_g{args.vis_mix_loop[0]:.2f}_s{args.vis_mix_loop[1]:.2f}_u{args.vis_mix_loop[2]:.2f}"
            + (f"_T{args.vis_temperature:g}" if args.vis_mix_loop[1] > 0.0 else "")
        )
        plot_frozenlake_scalar_overlay(
            env=env,
            value_map=occ_opt_mixed,
            output_dir=str(out_dir),
            filename_prefix=f"{stem}_occupancy_trainPolicy_{mix_suffix}",
            alpha=args.vis_occ_alpha,
            annotate=True,
            dpi=args.vis_dpi,
            target_cell_px=args.vis_occ_cell_px,
            font_scale=args.vis_occ_font_scale,
            cmap_name=args.vis_occ_cmap,
            gamma=args.vis_occ_gamma,
            min_abs_label=0.0,
            vmin=0.0,
            vmax=None,
            title="State Occupancy — Training policy (mixed)",
            cbar_label="Occupancy measure",
            value_format=None,
        )

        V_rand = policy_evaluation(
            mdp_network=mdp,
            policy=create_random_policy(mdp),
            gamma=args.vis_gamma,
            theta=args.vis_theta,
            max_iterations=args.vis_max_iters,
            verbose=False,
        )
        plot_frozenlake_scalar_overlay(
            env=env,
            value_map=V_rand,
            output_dir=str(out_dir),
            filename_prefix=f"{stem}_VALUE_random",
            alpha=args.vis_val_alpha,
            annotate=True,
            dpi=args.vis_dpi,
            target_cell_px=args.vis_val_cell_px,
            font_scale=args.vis_val_font_scale,
            cmap_name=args.vis_val_cmap,
            gamma=args.vis_val_gamma,
            min_abs_label=0.0,
            vmin=None,
            vmax=None,
            title="State Value V(s) — Random",
            cbar_label="V(s)",
            value_format=None,
        )
        plot_frozenlake_scalar_overlay(
            env=env,
            value_map=V_opt_greedy,
            output_dir=str(out_dir),
            filename_prefix=f"{stem}_VALUE_optimal_greedy",
            alpha=args.vis_val_alpha,
            annotate=True,
            dpi=args.vis_dpi,
            target_cell_px=args.vis_val_cell_px,
            font_scale=args.vis_val_font_scale,
            cmap_name=args.vis_val_cmap,
            gamma=args.vis_val_gamma,
            min_abs_label=0.0,
            vmin=None,
            vmax=None,
            title="State Value V(s) — Optimal (greedy)",
            cbar_label="V(s)",
            value_format=None,
        )

        # Cross-visualizations against native env (if present)
        if native_mdp is not None:
            occ_cross_native = compute_occupancy_measure(
                mdp_network=native_mdp,
                policy=policy_opt_mixed,
                gamma=args.vis_gamma,
                theta=args.vis_theta,
                max_iterations=args.vis_max_iters,
                verbose=False,
            )
            plot_frozenlake_scalar_overlay(
                env=env,
                value_map=occ_cross_native,
                output_dir=str(out_dir),
                filename_prefix=f"{stem}_occupancy_trainPolicy_on_NATIVE_{mix_suffix}",
                alpha=args.vis_occ_alpha,
                annotate=True,
                dpi=args.vis_dpi,
                target_cell_px=args.vis_occ_cell_px,
                font_scale=args.vis_occ_font_scale,
                cmap_name=args.vis_occ_cmap,
                gamma=args.vis_occ_gamma,
                min_abs_label=0.0,
                vmin=0.0,
                vmax=None,
                title="State Occupancy — Training policy on NATIVE",
                cbar_label="Occupancy measure",
                value_format=None,
            )
            if native_occ_random is not None:
                plot_frozenlake_scalar_diff_overlay(
                    env=env,
                    values_a=occ_cross_native,
                    values_b=native_occ_random,
                    output_dir=str(out_dir),
                    filename_prefix=f"{stem}_occupancy_DIFF_trainPolicyMINUS_nativeRandom_{mix_suffix}",
                    alpha=args.vis_occ_alpha,
                    annotate=True,
                    dpi=args.vis_dpi,
                    target_cell_px=args.vis_occ_cell_px,
                    font_scale=args.vis_occ_font_scale,
                    cmap_name="coolwarm",
                    min_abs_label=0.0,
                    vmin=None,
                    vmax=None,
                    title="Δ State Occupancy (training − native-random)",
                    cbar_label="Δ occupancy (A − B)",
                    value_format="+.2e",
                )

        if native_V_opt_greedy is not None:
            plot_frozenlake_scalar_diff_overlay(
                env=env,
                values_a=V_opt_greedy,
                values_b=native_V_opt_greedy,
                output_dir=str(out_dir),
                filename_prefix=f"{stem}_VALUE_DIFF_optGreedyMINUS_nativeOptGreedy",
                alpha=args.vis_val_alpha,
                annotate=True,
                dpi=args.vis_dpi,
                target_cell_px=args.vis_val_cell_px,
                font_scale=args.vis_val_font_scale,
                cmap_name="coolwarm",
                min_abs_label=0.0,
                vmin=None,
                vmax=None,
                title="Δ State Value: optGreedy(loop) − optGreedy(native)",
                cbar_label="Δ V(s) (loop − native)",
                value_format="+.2f",
            )

        # Upload a small sample of loop images to W&B
        if wandb_actor is not None:
            for fn in sorted(out_dir.glob("*.png"))[:8]:
                wandb_actor.log_image.remote(f"images/vis/{stem}/{fn.stem}", str(fn))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end GA → Curriculum → Visualization with W&B (via WandbWriter)."
    )

    # W&B
    p.add_argument("--outdir", type=str, default="./outputs")
    p.add_argument("--run-name", type=str, default=None)
    # p.add_argument("--wandb-project", type=str, default="full-frozenlake")
    p.add_argument("--wandb-project", type=str, default="comb-full-frozenlake")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument(
        "--wandb-mode", type=str, choices=["online", "offline"], default="online"
    )

    # Pipeline toggles
    p.add_argument("--skip-ga", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-vis", action="store_true")

    # Env
    p.add_argument("--map", type=str, default="8x8")
    p.add_argument("--slippery", type=str2bool, default=True)
    p.add_argument("--max-steps", type=int, default=1000)

    # GA (passed directly to run_ga_deap_ray)
    p.add_argument("--ga-pop-size", type=int, default=150)
    p.add_argument("--ga-generations", type=int, default=250)
    p.add_argument("--ga-tournament-k", type=int, default=2)
    p.add_argument("--ga-crossover", type=float, default=0.5)

    p.add_argument("--ga-allow-self-loops", type=str2bool, default=True)
    p.add_argument("--ga-max-out-degree", type=int, default=5)
    p.add_argument("--ga-prob-floor", type=float, default=1e-6)
    p.add_argument("--ga-add-edge-attempts-per-child", type=int, default=5)
    p.add_argument("--ga-epsilon-new-prob", type=float, default=0.1)
    p.add_argument("--ga-gamma-sample", type=float, default=1.0)
    p.add_argument("--ga-gamma-prob", type=float, default=0.0)
    p.add_argument("--ga-prune-prob-threshold", type=float, default=1e-3)
    p.add_argument("--ga-prob-tweak-actions-per-child", type=int, default=25)
    p.add_argument("--ga-prob-pairwise-step", type=float, default=0.05)
    p.add_argument("--ga-reward-tweak-edges-per-child", type=int, default=0)
    p.add_argument("--ga-reward-k-percent", type=float, default=0.05)
    p.add_argument("--ga-reward-ref-floor", type=float, default=1e-3)
    p.add_argument("--ga-add-edge-allow-out-of-scope", type=str2bool, default=False)

    p.add_argument("--ga-seed", type=int, default=0)

    p.add_argument("--ga-dist-max-hops", type=int, default=10)
    p.add_argument("--ga-dist-node-cap", type=int, default=64)
    p.add_argument("--ga-dist-weight-eps", type=float, default=1e-6)
    p.add_argument("--ga-dist-unreachable", type=float, default=1e9)

    # Solver knobs used by GA precompute
    p.add_argument("--ga-vi-gamma", type=float, default=0.99)
    p.add_argument("--ga-vi-theta", type=float, default=1e-3)
    p.add_argument("--ga-vi-max-iters", type=int, default=1000)
    p.add_argument("--ga-policy-mix", type=parse_tuple3, default=(0.9, 0.0, 0.1))
    p.add_argument("--ga-policy-temperature", type=float, default=0.01)
    p.add_argument("--ga-tie-tol", type=float, default=1e-2)

    # Objective group name (now maps to objectives + schedules)
    p.add_argument(
        "--obj-group",
        type=str,
        default="auc_auc",
        help="Objective group name selecting objectives + schedules.",
    )

    # Training (flattened agent kwargs; fed to CL API)
    p.add_argument(
        "--agent-ctor-path",
        type=str,
        default="simple_agents.tabular_q_agent:TabularQAgent",
        help="Dotted path 'module:factory_or_class' used by CL to construct the agent.",
    )
    p.add_argument("--agent-learning-rate", type=float, default=0.1)
    p.add_argument("--agent-gamma", type=float, default=0.99)
    p.add_argument(
        "--agent-policy-mix",
        type=parse_tuple3,
        default=(0.9, 0.0, 0.1),
        help="Tuple 'g,s,u' for (greedy, softmax, uniform).",
    )
    p.add_argument(
        "--agent-temperature",
        type=float,
        default=0.01,
        help="Used only if softmax weight > 0.",
    )
    p.add_argument("--agent-tie-tol", type=float, default=1e-2)
    p.add_argument("--agent-verbose", type=int, default=1)

    p.add_argument(
        "--phase-steps",
        type=str,
        default="20000,180000",
        help="Comma-separated curriculum steps per phase; e.g., 'X,Y' means 2 phases.",
    )
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--n-eval-episodes", type=int, default=100)

    # Seeds and parallelism for training
    p.add_argument(
        "--train-seeds", type=int, default=50, help="Use N to get seeds [0..N-1]."
    )
    p.add_argument("--train-save-intermediate", type=str2bool, default=True)

    # External JSON loops (optional)
    p.add_argument("--json-dir", type=str, default="")
    p.add_argument("--json-max", type=int, default=0)

    # Visualization
    p.add_argument("--vis-include-native", type=str2bool, default=True)
    p.add_argument("--vis-min-prob", type=float, default=0.05)
    p.add_argument("--vis-alpha", type=float, default=0.65)
    p.add_argument("--vis-show-self-loops", type=str2bool, default=False)
    p.add_argument("--vis-dpi", type=int, default=200)
    p.add_argument("--vis-gamma", type=float, default=0.99)
    p.add_argument("--vis-theta", type=float, default=1e-6)
    p.add_argument("--vis-max-iters", type=int, default=1000)
    p.add_argument("--vis-temperature", type=float, default=1.0)
    p.add_argument("--vis-mix-native", type=parse_tuple3, default=(1.0, 0.0, 0.0))
    p.add_argument("--vis-mix-loop", type=parse_tuple3, default=(0.9, 0.0, 0.1))
    p.add_argument("--vis-tie-tol", type=float, default=1e-2)
    p.add_argument("--vis-occ-alpha", type=float, default=0.65)
    p.add_argument("--vis-occ-cell-px", type=int, default=240)
    p.add_argument("--vis-occ-font-scale", type=float, default=0.16)
    p.add_argument("--vis-occ-cmap", type=str, default="magma")
    p.add_argument("--vis-occ-gamma", type=float, default=1.0)
    p.add_argument("--vis-val-alpha", type=float, default=0.65)
    p.add_argument("--vis-val-cell-px", type=int, default=240)
    p.add_argument("--vis-val-font-scale", type=float, default=0.16)
    p.add_argument("--vis-val-cmap", type=str, default="viridis")
    p.add_argument("--vis-val-gamma", type=float, default=1.0)
    return p


def main():
    parser = build_arg_parser()
    args = resolve_args(parser)

    # Select group (objectives + schedules)
    if args.obj_group not in OBJECTIVE_GROUPS:
        raise ValueError(
            f"Unknown --obj-group '{args.obj_group}'. "
            f"Available: {', '.join(sorted(OBJECTIVE_GROUPS))}"
        )
    grp = OBJECTIVE_GROUPS[args.obj_group]
    selected_objective_keys: List[str] = list(grp["objective_keys"])
    fitness_schedule = list(grp["fitness_schedule"])
    eval_schedule = list(grp["eval_schedule"])
    eval_at_begin = grp["eval_at_begin"]
    eval_at_end = grp["eval_at_end"]
    selected_max_metric_keys: List[str] = SELECTED_MAX_METRIC_KEYS

    # Make run name obj_group-aware
    base_name = args.run_name or f"flk_{args.map}"
    if args.obj_group not in base_name:
        args.run_name = f"{base_name}__{args.obj_group}"
    else:
        args.run_name = base_name

    # Timestamped outdir: <outdir>/ga-frozenlake/<ts>
    run_dir = _timestamped_outdir(args.outdir, leaf="ga-frozenlake")
    ensure_dir(run_dir)
    args.outdir = str(run_dir)

    print(f"[SETUP] Results outdir: {args.outdir}")
    print(f"[SETUP] Obj-group: {args.obj_group} -> {selected_objective_keys}")

    # Save full config
    meta_dir = Path(args.outdir) / "meta"
    ensure_dir(meta_dir)
    save_json(meta_dir / "config.json", {k: getattr(args, k) for k in vars(args)})

    # Init Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # W&B actor
    init_kwargs: Dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.run_name,
        "mode": args.wandb_mode,
        "config": {k: getattr(args, k) for k in vars(args)},
    }
    init_kwargs["config"]["obj_group"] = args.obj_group
    init_kwargs["config"]["objective_keys"] = selected_objective_keys
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    env_overrides = {"WANDB_MODE": args.wandb_mode, "WANDB_START_METHOD": "thread"}
    wandb_actor: ActorHandle = WandbActor.remote(init_kwargs, env=env_overrides)
    capture_prints_to_wandb(wandb_actor)

    # ----------------------------- GA Stage -----------------------------
    if args.json_dir:
        json_dir = Path(args.json_dir)
        json_files = sorted(json_dir.glob("*.json"))
        if args.json_max > 0:
            json_files = json_files[: args.json_max]
        print(f"[MAIN] Using external JSON dir: {json_dir} ({len(json_files)} files).")
    else:
        if not args.skip_ga:
            print("[GA] Building native MDP…")
            base_mdp = _build_native_mdp(args.map, args.slippery)

            ops = {
                "allow_self_loops": args.ga_allow_self_loops,
                "max_out_degree": args.ga_max_out_degree,
                "prob_floor": args.ga_prob_floor,
                "add_edge_attempts_per_child": args.ga_add_edge_attempts_per_child,
                "epsilon_new_prob": args.ga_epsilon_new_prob,
                "gamma_sample": args.ga_gamma_sample,
                "gamma_prob": args.ga_gamma_prob,
                "prune_prob_threshold": args.ga_prune_prob_threshold,
                "prob_tweak_actions_per_child": args.ga_prob_tweak_actions_per_child,
                "prob_pairwise_step": args.ga_prob_pairwise_step,
                "reward_tweak_edges_per_child": args.ga_reward_tweak_edges_per_child,
                "reward_k_percent": args.ga_reward_k_percent,
                "reward_ref_floor": args.ga_reward_ref_floor,
                "add_edge_allow_out_of_scope": args.ga_add_edge_allow_out_of_scope,
            }
            distance = {
                "dist_max_hops": args.ga_dist_max_hops,
                "dist_node_cap": args.ga_dist_node_cap,
                "dist_weight_eps": args.ga_dist_weight_eps,
                "dist_unreachable": args.ga_dist_unreachable,
            }
            solver = {
                "vi_gamma": args.ga_vi_gamma,
                "vi_theta": args.ga_vi_theta,
                "vi_max_iterations": args.ga_vi_max_iters,
                "policy_mix": tuple(args.ga_policy_mix),
                "policy_temperature": args.ga_policy_temperature,
                "policy_tie_tol": args.ga_tie_tol,
            }

            # score_spec with per-score resources (unchanged)
            score_spec = [
                {
                    "name": "obj_cl_phase_mean",
                    "params": {
                        "target_factory_path": TARGET_FACTORY_PATH,
                        "target_cfg": {
                            "map_name": args.map,
                            "is_slippery": bool(args.slippery),
                            "max_steps": int(args.max_steps),
                        },
                        "item_factory_path": SOURCE_FACTORY_PATH,
                        "item_max_steps": int(args.max_steps),
                        "phase_steps": (20_000, 80_000),
                        "seeds": 5,
                        "agent_ctor_path": "simple_agents.tabular_q_agent:TabularQAgent",
                        "agent_kwargs": {
                            "learning_rate": 0.1, "gamma": 0.99,
                            "policy_mix": (0.9, 0.0, 0.1),
                            "temperature": 0.01, "default_q_value": 0.05,
                            "tie_tol": 1e-2, "verbose": 0,
                        },
                        "eval_every": 2500, "n_eval_episodes": 100,
                    },
                    "resources": {"cpus": 1, "gpus": 0},
                },
                {
                    "name": "obj_multi_perf",
                    "params": {"perf_numpoints": 16, "perf_gamma": 0.99, "perf_theta": 1e-3,
                               "perf_max_iterations": 1000},
                    "resources": {"cpus": 1, "gpus": 0},
                },
                {
                    "name": "obj_multi_kl",
                    "params": {"kl_gamma": 0.99, "kl_theta": 1e-3, "kl_max_iterations": 1000},
                    "resources": {"cpus": 1, "gpus": 0},
                },
                {
                    "name": "obj_val_diff",
                    "params": {"diff_gamma": 0.99, "diff_theta": 1e-3, "diff_max_iterations": 1000},
                    "resources": {"cpus": 1, "gpus": 0},
                },
            ]

            # Auto-pick outer concurrency (max_in_flight)
            avail = ray.available_resources()
            tot_cpu = int(avail.get("CPU", os.cpu_count() or 1))
            tot_gpu = float(avail.get("GPU", 0.0))
            cpu_per_ind = sum(float(s["resources"].get("cpus", 1)) for s in score_spec)
            gpu_per_ind = sum(float(s["resources"].get("gpus", 0.0)) for s in score_spec)
            cpu_bound = max(1, int(tot_cpu // max(1, math.ceil(cpu_per_ind))))
            if gpu_per_ind > 0.0:
                gpu_bound = max(1, int(tot_gpu // max(0.0001, gpu_per_ind)))
                max_in_flight = max(1, min(args.ga_pop_size, cpu_bound, gpu_bound))
            else:
                max_in_flight = max(1, min(args.ga_pop_size, cpu_bound))

            objective_keys: List[str] = selected_objective_keys
            max_metric_keys: List[str] = selected_max_metric_keys

            _ = run_ga_deap_ray(
                base_mdp=base_mdp,
                population_size=args.ga_pop_size,
                generations=args.ga_generations,
                master_seed=args.ga_seed,
                algo_type="mu_plus_lambda",
                mu=args.ga_pop_size, lambd=args.ga_pop_size, cxpb=args.ga_crossover,
                survivor="nsga2",
                parents="tournament_dcd", parent_k=args.ga_tournament_k,
                score_spec=score_spec,
                objective_keys=objective_keys,
                max_metric_keys=max_metric_keys,
                max_in_flight=max_in_flight,
                ray_init={},  # cluster: {"address": "auto"}
                ops=ops, distance=distance, solver=solver,
                output_dir=args.outdir,
                wandb_writer=wandb_actor,
                # NEW: schedules taken from group
                fitness_schedule=fitness_schedule,
                eval_schedule=eval_schedule,
                eval_at_begin=eval_at_begin,
                eval_at_end=eval_at_end,
            )

            mdp_out_dir = Path(args.outdir) / "ga" / "mdps"
            json_files = sorted(mdp_out_dir.glob("*.json"))
            if args.json_max > 0:
                json_files = json_files[: args.json_max]
            print(f"[MAIN] GA done; using {len(json_files)} JSON files from {mdp_out_dir}.")
        else:
            mdp_out_dir = Path(args.outdir) / "ga" / "mdps"
            json_files = sorted(mdp_out_dir.glob("*.json"))
            if args.json_max > 0:
                json_files = json_files[: args.json_max]
            print(f"[MAIN] GA skipped; using {len(json_files)} JSON from {mdp_out_dir}.")

    # ----------------------------- Curriculum Training -----------------------------
    if not args.skip_train and json_files:
        print("[Training] Curriculum run starting...")

        # Seeds [0..N-1]
        seeds = args.train_seeds

        # Env registry for CL
        envs: Dict[str, Any] = {
            "target": {
                "factory_path": TARGET_FACTORY_PATH,
                "cfg": {
                    "map_name": args.map,
                    "is_slippery": bool(args.slippery),
                    "max_steps": int(args.max_steps),
                },
            },
            "items": {
                jf.stem: {
                    "factory_path": SOURCE_FACTORY_PATH,
                    "cfg": {
                        "mdp_config_path": str(jf),
                        "max_steps": int(args.max_steps),
                    },
                }
                for jf in json_files
            },
        }

        # Curriculum phases
        phase_steps = args.phase_steps
        if len(phase_steps) == 1:
            baseline_phases = [{"env": "target", "steps": phase_steps[0]}]
        else:
            baseline_phases = [{"env": "target", "steps": phase_steps[0]}]

        baseline_evals = [{"name": "Target", "env": "target"}]

        # Per-item schedules and evals
        item_phases_map: Dict[str, List[Dict[str, Any]]] = {}
        for key in envs["items"].keys():
            phases: List[Dict[str, Any]] = []
            if len(phase_steps) >= 1:
                phases.append({"env": key, "steps": phase_steps[0]})
            if len(phase_steps) >= 2:
                phases.append({"env": "target", "steps": phase_steps[1]})
            item_phases_map[key] = phases

        evals_map: Dict[str, List[Dict[str, Any]]] = {}
        for key in envs["items"].keys():
            evals_map[key] = [
                {"name": key, "env": key},
                {"name": "Target", "env": "target"},
            ]

        # Agent config for CL
        agent_kwargs = {
            "learning_rate": float(args.agent_learning_rate),
            "gamma": float(args.agent_gamma),
            "policy_mix": tuple(args.agent_policy_mix),
            "temperature": float(args.agent_temperature),
            "tie_tol": float(args.agent_tie_tol),
            "verbose": int(args.agent_verbose),
        }

        # Run CL
        cl_summary = run_curriculum(
            seeds=seeds,
            envs=envs,
            baseline_phases=baseline_phases,
            baseline_evals=baseline_evals,
            item_phases_map=item_phases_map,
            evals_map=evals_map,
            agent_ctor_path=args.agent_ctor_path,
            agent_kwargs=agent_kwargs,
            eval_every=int(args.eval_every),
            n_eval_episodes=int(args.n_eval_episodes),
            output_dir=args.outdir,
            save_intermediate=bool(args.train_save_intermediate),
            wandb_actor=wandb_actor,
            media_opts={"target_size": (128, 128)},
        )
        print(f"[MAIN] Curriculum run done; summary:\n{cl_summary}")
    else:
        print("[MAIN] Training skipped or no JSON files; skipping curriculum stage.")

    # ----------------------------- Visualization -----------------------------
    if not args.skip_vis:
        stage_visualize(args, wandb_actor, json_files)  # assumed existing in your codebase
    else:
        print("[MAIN] Visualization skipped.")

    # Graceful W&B shutdown
    ray.get(wandb_actor.finish.remote())
    print("\n[MAIN] All done.")


if __name__ == "__main__":
    main()
