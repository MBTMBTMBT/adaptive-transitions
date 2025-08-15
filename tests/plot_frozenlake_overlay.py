#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch overlay plotter for FrozenLake MDPs + Occupancy & Value overlays.

(A) Native (baseline) env first:
    - Build native MDP via env.get_mdp_network().
    - Plot transition overlays.
    - Build policies:
        * Random (uniform)
        * Optimal policy for OCCUPANCY plots: by MIXING (default greedy for native).
        * Optimal policy for VALUE plots: strictly greedy (mixing=(0,0,1)), independent of the occupancy policy.
    - Plot OCCUPANCY overlays (random vs optimal-mixed).
    - Plot VALUE overlays (random vs optimal-greedy).

(B) For each JSON MDP:
    - Same as above for its own env.
    - Cross-eval occupancy on native (as before).
    - VALUE difference plot: V_opt_greedy(loop) − V_opt_greedy(native).

(C) Naming rule: if MIXING softmax weight == 0, omit temperature “T...” in filenames.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from gymnasium.envs.toy_text import FrozenLakeEnv

# Project modules (adjust paths if needed)
from customised_toy_text_envs.customised_frozenlake import (
    CustomisedFrozenLakeEnv,                 # used to get native MDP from env
    plot_frozenlake_transition_overlays,
    plot_frozenlake_scalar_overlay,          # generic scalar overlay (for occupancy or V(s))
    plot_frozenlake_scalar_diff_overlay,     # generic scalar diff overlay
)
from mdp_network import MDPNetwork
from mdp_network.mdp_tables import create_random_policy, q_table_to_policy
from mdp_network.solvers import (
    compute_occupancy_measure,
    optimal_value_iteration,
    policy_evaluation,                       # evaluate V^π
)

# -----------------------------
# Hard-coded constants
# -----------------------------
JSON_DIR: Path = Path("./outputs/ga_test")
OUTPUT_DIR: Path = Path("./outputs/ga_vis")
MAP_NAME: str = "8x8"
IS_SLIPPERY: bool = True
RECURSIVE: bool = True

# Transition overlay style
MIN_PROB: float = 0.05
ALPHA: float = 0.65
SHOW_SELF_LOOPS: bool = False
DPI: int = 200

# Occupancy computation params
OCC_GAMMA: float = 0.99
OCC_THETA: float = 1e-6
OCC_MAX_ITERS: int = 1000

# Temperature for softmax component in mixing (only used if mixing[1] > 0)
SOFTMAX_TEMPERATURE: float = 1.0

# Policy mapping (Q* -> Policy) mixing weights: (greedy，softmax, uniform,)
NATIVE_OPT_POLICY_MIXING: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # pure greedy for native occupancy plots
LOOP_OPT_POLICY_MIXING:   Tuple[float, float, float] = (0.9, 0.0, 0.1) # default eps-greedy for JSON MDP occupancy
OPT_TIE_TOL: float = 1e-2

# Overlays (occupancy)
OCC_ALPHA: float = 0.65
OCC_TARGET_CELL_PX: int = 240
OCC_FONT_SCALE: float = 0.16
OCC_CMAP: str = "magma"
OCC_COLOR_GAMMA: float = 1.0
OCC_MIN_LABEL: float = 0.0
OCC_VMIN = 0.0
OCC_VMAX = None

# Overlays (value) — reuse sizes; different colormap
VAL_ALPHA: float = OCC_ALPHA
VAL_TARGET_CELL_PX: int = OCC_TARGET_CELL_PX
VAL_FONT_SCALE: float = OCC_FONT_SCALE
VAL_CMAP: str = "viridis"
VAL_COLOR_GAMMA: float = 1.0
VAL_MIN_ABS_LABEL: float = 0.0
VAL_VMIN = None
VAL_VMAX = None

# Native output subfolder/prefix
NATIVE_SUBDIR_NAME: str = "__native_frozenlake__"
NATIVE_PREFIX: str = "native_frozenlake"


def find_json_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted([p for p in root.rglob("*.json") if p.is_file()])
    return sorted([p for p in root.glob("*.json") if p.is_file()])


def load_json(p: Path) -> Dict[str, Any]:
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON: {p}") from e


def ensure_env(map_name: str, is_slippery: bool) -> CustomisedFrozenLakeEnv:
    env = CustomisedFrozenLakeEnv(render_mode="rgb_array", map_name=map_name, is_slippery=is_slippery)
    try:
        env.reset()
        if hasattr(env, "initial_state_distrib"):
            env.s = int(env.initial_state_distrib.argmax())
    except Exception:
        pass
    return env


def states_aligned(env: FrozenLakeEnv, mdp: MDPNetwork) -> bool:
    nS = env.nrow * env.ncol
    return set(mdp.states) == set(range(nS))


def _mix_tag(mix: Tuple[float, float, float]) -> str:
    """Filename tag using (greedy, softmax, uniform)."""
    g, s, u = mix
    return f"mix_g{g:.2f}_s{s:.2f}_u{u:.2f}"

def _mix_suffix(mix: Tuple[float, float, float], temperature: float) -> str:
    """If softmax weight==0, omit temperature."""
    tag = _mix_tag(mix)
    return f"{tag}_T{temperature:g}" if mix[1] > 0.0 else tag


def _build_policy_and_values(
    mdp: MDPNetwork,
    mixing: Tuple[float, float, float],
    temperature: float,
    tie_tol: float,
    gamma: float,
    theta: float,
    max_iterations: int,
):
    """
    Return:
      policy_rand, occ_rand,
      policy_opt_mixed, occ_opt_mixed,
      policy_opt_greedy, V_opt_greedy
    """
    policy_rand = create_random_policy(mdp)
    occ_rand = compute_occupancy_measure(
        mdp_network=mdp, policy=policy_rand, gamma=gamma, theta=theta,
        max_iterations=max_iterations, verbose=False,
    )

    V_star, Q_star = optimal_value_iteration(
        mdp_network=mdp, gamma=gamma, theta=theta,
        max_iterations=max_iterations, verbose=False,
    )

    policy_opt_mixed = q_table_to_policy(
        q_table=Q_star, states=mdp.states, num_actions=mdp.num_actions,
        mixing=mixing, temperature=temperature, tie_tol=tie_tol,
    )
    occ_opt_mixed = compute_occupancy_measure(
        mdp_network=mdp, policy=policy_opt_mixed, gamma=gamma, theta=theta,
        max_iterations=max_iterations, verbose=False,
    )

    policy_opt_greedy = q_table_to_policy(
        q_table=Q_star, states=mdp.states, num_actions=mdp.num_actions,
        mixing=(1.0, 0.0, 0.0), temperature=1.0, tie_tol=tie_tol,
    )
    V_opt_greedy = policy_evaluation(
        mdp_network=mdp, policy=policy_opt_greedy,
        gamma=gamma, theta=theta, max_iterations=max_iterations, verbose=False,
    )

    return (policy_rand, occ_rand,
            policy_opt_mixed, occ_opt_mixed,
            policy_opt_greedy, V_opt_greedy)


def process_one_mdp_bundle(
    label: str,
    env: FrozenLakeEnv,
    mdp: MDPNetwork,
    out_dir: Path,
    native_mdp: Optional[MDPNetwork] = None,           # for occupancy cross-eval
    native_occ_random: Optional["ValueTable"] = None,  # baseline random occupancy on native (for diff)
    native_V_opt_greedy: Optional["ValueTable"] = None,# baseline optimal-greedy V (for diff)
    opt_policy_mixing: Tuple[float, float, float] = LOOP_OPT_POLICY_MIXING,
    softmax_temperature: float = SOFTMAX_TEMPERATURE,
    tie_tol: float = OPT_TIE_TOL,
):
    """
    - Transition overlays
    - Occupancy: random vs optimal-mixed (generic scalar overlay)
    - Value: random vs optimal-greedy (generic scalar overlay)
    - Cross-eval occupancy on native, and diff vs native random
    - Value diff vs native (opt-greedy(loop) − opt-greedy(native))
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # (1) Transition overlays
    plot_frozenlake_transition_overlays(
        env=env, mdp=mdp, output_dir=str(out_dir), filename_prefix=label,
        min_prob=MIN_PROB, alpha=ALPHA, annotate=True,
        show_self_loops=SHOW_SELF_LOOPS, dpi=DPI,
    )
    print(f"[OK] Saved transition overlays for '{label}'")

    # (2) Policies / occupancy / values
    (policy_rand, occ_rand,
     policy_opt_mixed, occ_opt_mixed,
     policy_opt_greedy, V_opt_greedy) = _build_policy_and_values(
        mdp=mdp, mixing=opt_policy_mixing, temperature=softmax_temperature,
        tie_tol=tie_tol, gamma=OCC_GAMMA, theta=OCC_THETA, max_iterations=OCC_MAX_ITERS,
    )

    # (3) OCCUPANCY overlays (generic scalar overlay)
    plot_frozenlake_scalar_overlay(
        env=env, value_map=occ_rand, output_dir=str(out_dir),
        filename_prefix=f"{label}_occupancy_random",
        alpha=OCC_ALPHA, annotate=True, dpi=DPI,
        target_cell_px=OCC_TARGET_CELL_PX, font_scale=OCC_FONT_SCALE,
        cmap_name=OCC_CMAP, gamma=OCC_COLOR_GAMMA,
        min_abs_label=OCC_MIN_LABEL, vmin=OCC_VMIN, vmax=OCC_VMAX,
        title="State Occupancy", cbar_label="Occupancy measure",
        value_format=None,
    )

    mix_suffix = _mix_suffix(opt_policy_mixing, softmax_temperature)
    plot_frozenlake_scalar_overlay(
        env=env, value_map=occ_opt_mixed, output_dir=str(out_dir),
        filename_prefix=f"{label}_occupancy_optimal_{mix_suffix}",
        alpha=OCC_ALPHA, annotate=True, dpi=DPI,
        target_cell_px=OCC_TARGET_CELL_PX, font_scale=OCC_FONT_SCALE,
        cmap_name=OCC_CMAP, gamma=OCC_COLOR_GAMMA,
        min_abs_label=OCC_MIN_LABEL, vmin=OCC_VMIN, vmax=OCC_VMAX,
        title="State Occupancy — Optimal (mixed)", cbar_label="Occupancy measure",
        value_format=None,
    )

    # (4) VALUE overlays (generic scalar overlay)
    V_rand = policy_evaluation(
        mdp_network=mdp, policy=policy_rand,
        gamma=OCC_GAMMA, theta=OCC_THETA, max_iterations=OCC_MAX_ITERS, verbose=False,
    )
    plot_frozenlake_scalar_overlay(
        env=env, value_map=V_rand, output_dir=str(out_dir),
        filename_prefix=f"{label}_VALUE_random",
        alpha=VAL_ALPHA, annotate=True, dpi=DPI,
        target_cell_px=VAL_TARGET_CELL_PX, font_scale=VAL_FONT_SCALE,
        cmap_name=VAL_CMAP, gamma=VAL_COLOR_GAMMA,
        min_abs_label=VAL_MIN_ABS_LABEL, vmin=VAL_VMIN, vmax=VAL_VMAX,
        title="State Value V(s) — Random policy", cbar_label="V(s)",
        value_format=None,
    )
    plot_frozenlake_scalar_overlay(
        env=env, value_map=V_opt_greedy, output_dir=str(out_dir),
        filename_prefix=f"{label}_VALUE_optimal_greedy",
        alpha=VAL_ALPHA, annotate=True, dpi=DPI,
        target_cell_px=VAL_TARGET_CELL_PX, font_scale=VAL_FONT_SCALE,
        cmap_name=VAL_CMAP, gamma=VAL_COLOR_GAMMA,
        min_abs_label=VAL_MIN_ABS_LABEL, vmin=VAL_VMIN, vmax=VAL_VMAX,
        title="State Value V(s) — Optimal (greedy)", cbar_label="V(s)",
        value_format=None,
    )

    # (5) Cross-eval occupancy on native + diff vs native random
    if native_mdp is not None:
        occ_cross_native = compute_occupancy_measure(
            mdp_network=native_mdp, policy=policy_opt_mixed,
            gamma=OCC_GAMMA, theta=OCC_THETA, max_iterations=OCC_MAX_ITERS, verbose=False,
        )
        plot_frozenlake_scalar_overlay(
            env=env, value_map=occ_cross_native, output_dir=str(out_dir),
            filename_prefix=f"{label}_occupancy_optPolicy_on_NATIVE_{mix_suffix}",
            alpha=OCC_ALPHA, annotate=True, dpi=DPI,
            target_cell_px=OCC_TARGET_CELL_PX, font_scale=OCC_FONT_SCALE,
            cmap_name=OCC_CMAP, gamma=OCC_COLOR_GAMMA,
            min_abs_label=OCC_MIN_LABEL, vmin=OCC_VMIN, vmax=OCC_VMAX,
            title="State Occupancy — Policy (learned) on NATIVE", cbar_label="Occupancy measure",
            value_format=None,
        )

        if native_occ_random is not None:
            plot_frozenlake_scalar_diff_overlay(
                env=env,
                values_a=occ_cross_native,
                values_b=native_occ_random,
                output_dir=str(out_dir),
                filename_prefix=f"{label}_occupancy_DIFF_optPolicyMINUS_nativeRandom_{mix_suffix}",
                alpha=OCC_ALPHA, annotate=True, dpi=DPI,
                target_cell_px=OCC_TARGET_CELL_PX, font_scale=OCC_FONT_SCALE,
                cmap_name="coolwarm", min_abs_label=0.0, vmin=None, vmax=None,
                title="Δ State Occupancy (A − B)", cbar_label="Δ occupancy (A − B)",
                value_format="+.2e",
            )

    # (6) VALUE diff vs native: V_opt_greedy(this) − V_opt_greedy(native)
    if native_V_opt_greedy is not None:
        plot_frozenlake_scalar_diff_overlay(
            env=env,
            values_a=V_opt_greedy,
            values_b=native_V_opt_greedy,
            output_dir=str(out_dir),
            filename_prefix=f"{label}_VALUE_DIFF_optGreedyMINUS_nativeOptGreedy",
            alpha=VAL_ALPHA, annotate=True, dpi=DPI,
            target_cell_px=VAL_TARGET_CELL_PX, font_scale=VAL_FONT_SCALE,
            cmap_name="coolwarm", min_abs_label=0.0, vmin=None, vmax=None,
            title="Δ State Value: optGreedy(loop) − optGreedy(native)",
            cbar_label="Δ V(s) (loop − native)",
            value_format="+.2f",
        )

    print(f"[OK] Saved overlays for '{label}' -> {out_dir}")


def main():
    json_dir = JSON_DIR.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = ensure_env(MAP_NAME, IS_SLIPPERY)
    nS = env.nrow * env.ncol
    print(f"[INFO] Env grid: {env.nrow}x{env.ncol} ({nS} states).")

    # (A) Native baseline
    native_mdp = None
    native_occ_random = None
    native_V_opt_greedy = None
    try:
        native_mdp = env.get_mdp_network()

        # Native baselines
        native_policy_rand = create_random_policy(native_mdp)
        native_occ_random = compute_occupancy_measure(
            mdp_network=native_mdp, policy=native_policy_rand,
            gamma=OCC_GAMMA, theta=OCC_THETA, max_iterations=OCC_MAX_ITERS, verbose=False,
        )

        V_star_native, Q_star_native = optimal_value_iteration(
            mdp_network=native_mdp, gamma=OCC_GAMMA, theta=OCC_THETA,
            max_iterations=OCC_MAX_ITERS, verbose=False,
        )
        native_policy_opt_greedy = q_table_to_policy(
            q_table=Q_star_native, states=native_mdp.states, num_actions=native_mdp.num_actions,
            mixing=(0.0, 0.0, 1.0), temperature=1.0, tie_tol=OPT_TIE_TOL,
        )
        native_V_opt_greedy = policy_evaluation(
            mdp_network=native_mdp, policy=native_policy_opt_greedy,
            gamma=OCC_GAMMA, theta=OCC_THETA, max_iterations=OCC_MAX_ITERS, verbose=False,
        )

        native_out = output_dir / NATIVE_SUBDIR_NAME

        process_one_mdp_bundle(
            label=NATIVE_PREFIX,
            env=env,
            mdp=native_mdp,
            out_dir=native_out,
            native_mdp=None,
            native_occ_random=None,
            native_V_opt_greedy=None,
            opt_policy_mixing=NATIVE_OPT_POLICY_MIXING,
            softmax_temperature=SOFTMAX_TEMPERATURE,
            tie_tol=OPT_TIE_TOL,
        )
    except Exception as e:
        print(f"[WARN] Failed to build or plot native FrozenLake MDP: {e}")

    # (B) JSON MDPs
    json_files = find_json_files(json_dir, RECURSIVE)
    if not json_files:
        print(f"[WARN] No JSON files found under: {json_dir}")
    else:
        print(f"[INFO] Found {len(json_files)} JSON file(s) under {json_dir}")

    for jf in json_files:
        print(f"[INFO] Processing: {jf}")
        try:
            cfg = load_json(jf)
            mdp = MDPNetwork(config_data=cfg)

            if getattr(mdp, "num_actions", None) != 4:
                print(f"[WARN] Skip (num_actions != 4): {jf}")
                continue
            if not states_aligned(env, mdp):
                print(f"[WARN] Skip (states not aligned to 0..{nS-1}): {jf}")
                continue

            stem = jf.stem
            out_subdir = output_dir / stem

            process_one_mdp_bundle(
                label=stem,
                env=env,
                mdp=mdp,
                out_dir=out_subdir,
                native_mdp=native_mdp,                   # cross-eval occupancy on native
                native_occ_random=native_occ_random,     # for occupancy diff
                native_V_opt_greedy=native_V_opt_greedy, # for value diff
                opt_policy_mixing=LOOP_OPT_POLICY_MIXING,
                softmax_temperature=SOFTMAX_TEMPERATURE,
                tie_tol=OPT_TIE_TOL,
            )

        except Exception as e:
            print(f"[ERROR] Failed on {jf}: {e}")

    print(f"[DONE] All finished. Outputs in: {output_dir}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
