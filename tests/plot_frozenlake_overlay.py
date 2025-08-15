#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch overlay plotter for FrozenLake MDPs + Occupancy overlays.

Add-ons:
(A) Before processing external JSON MDPs, this script also:
    - Builds the native FrozenLake MDP from the environment itself (via get_mdp_network()).
    - Plots transition overlays for the native MDP.
    - Computes occupancy measures for:
        (a) a random (uniform) policy, and
        (b) an optimal policy (Q*) mapped to a policy using a MIXING tuple.
            * For the native MDP: greedy (mixing=(0,0,1))
            * Temperature SOFTMAX_TEMPERATURE is kept as a tunable float (affects softmax part if used).
    - Plots occupancy overlays for both policies.

(B) For each JSON MDP, the script now:
    - Plots transition overlays for that MDP.
    - Computes occupancy for
        (a) a random (uniform) policy, and
        (b) an optimal policy (Q* -> policy via MIXING; default 0.1 uniform + 0.9 greedy).
    - Plots occupancy overlays for both policies.
    - Cross-evaluates the learned optimal mixed policy on the NATIVE MDP and plots its occupancy overlay as well.
    - NEW: Also plots an occupancy *difference* (on the NATIVE MDP):
        (learned policy on native) − (native random policy).
        Naming rule: if mixing has zero softmax weight (mix[1] == 0), omit the temperature suffix “T...”.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from gymnasium.envs.toy_text import FrozenLakeEnv

# Project modules (adjust if your paths differ)
from customised_toy_text_envs.customised_frozenlake import (
    CustomisedFrozenLakeEnv,               # used to get native MDP from env
    plot_frozenlake_transition_overlays,
    plot_frozenlake_occupancy_overlays,   # occupancy overlay renderer
    plot_frozenlake_occupancy_diff_overlay,  # NEW: occupancy difference renderer
)
from mdp_network import MDPNetwork
from mdp_network.mdp_tables import create_random_policy, q_table_to_policy
from mdp_network.solvers import compute_occupancy_measure, optimal_value_iteration


# -----------------------------
# Hard-coded constants
# -----------------------------
JSON_DIR: Path = Path("./outputs/ga_test")   # Directory containing MDP JSON files
OUTPUT_DIR: Path = Path("./outputs/ga_vis")  # Output directory for images
MAP_NAME: str = "8x8"                        # "4x4" or "8x8"
IS_SLIPPERY: bool = True                     # Background env dynamics flag
RECURSIVE: bool = True                       # Whether to search subdirectories for JSONs

# Transition overlay style
MIN_PROB: float = 0.05                        # Min probability threshold to draw arrows
ALPHA: float = 0.65                           # Transparency for arrows and labels
SHOW_SELF_LOOPS: bool = False                 # Draw self-loop arcs for s->s
DPI: int = 200                                # Figure DPI

# Occupancy computation params (for both policies)
OCC_GAMMA: float = 0.99
OCC_THETA: float = 1e-6
OCC_MAX_ITERS: int = 1000

# Temperature for softmax component in mixing (used only if mixing[1] > 0)
SOFTMAX_TEMPERATURE: float = 1.0

# Policy mapping (Q* -> Policy) mixing weights: (uniform, softmax, greedy)
# Native MDP optimal policy: pure greedy
NATIVE_OPT_POLICY_MIXING: Tuple[float, float, float] = (0.0, 0.0, 1.0)
# JSON MDP optimal policy: epsilon-greedy with epsilon=0.1 (parameterized)
LOOP_OPT_POLICY_MIXING: Tuple[float, float, float] = (0.1, 0.0, 0.9)
# Tolerance for tie-breaking in greedy/softmax
OPT_TIE_TOL: float = 1e-2

# Occupancy overlay style
OCC_ALPHA: float = 0.65
OCC_TARGET_CELL_PX: int = 240
OCC_FONT_SCALE: float = 0.16
OCC_CMAP: str = "magma"
OCC_COLOR_GAMMA: float = 1.0
OCC_MIN_LABEL: float = 0.0                    # Do not draw labels below this value
OCC_VMIN = 0.0                                # Color scale min (0 default)
OCC_VMAX = None                               # Color scale max (None -> auto)

# Native (environment-derived) output subfolder and file prefix
NATIVE_SUBDIR_NAME: str = "__native_frozenlake__"
NATIVE_PREFIX: str = "native_frozenlake"


def find_json_files(root: Path, recursive: bool) -> List[Path]:
    """Collect JSON files under a directory."""
    if recursive:
        return sorted([p for p in root.rglob("*.json") if p.is_file()])
    return sorted([p for p in root.glob("*.json") if p.is_file()])


def load_json(p: Path) -> Dict[str, Any]:
    """Load JSON config with a clear error if it fails."""
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON: {p}") from e


def ensure_env(map_name: str, is_slippery: bool) -> CustomisedFrozenLakeEnv:
    """
    Build a CustomisedFrozenLakeEnv with rgb_array capability for background rendering
    and access to get_mdp_network().
    """
    env = CustomisedFrozenLakeEnv(render_mode="rgb_array", map_name=map_name, is_slippery=is_slippery)
    try:
        env.reset()
        if hasattr(env, "initial_state_distrib"):
            env.s = int(env.initial_state_distrib.argmax())
    except Exception:
        pass
    return env


def states_aligned(env: FrozenLakeEnv, mdp: MDPNetwork) -> bool:
    """Check whether mdp.states equals {0..nS-1}."""
    nS = env.nrow * env.ncol
    return set(mdp.states) == set(range(nS))


def _mix_tag(mix: Tuple[float, float, float]) -> str:
    """Create a short tag for filenames from a mixing tuple (u,s,g)."""
    u, s, g = mix
    return f"mix_u{u:.2f}_s{s:.2f}_g{g:.2f}"


def _mix_suffix(mix: Tuple[float, float, float], temperature: float) -> str:
    """
    Filename suffix for a mixing config.
    If softmax weight == 0, omit temperature from the suffix.
    """
    tag = _mix_tag(mix)
    return f"{tag}_T{temperature:g}" if mix[1] > 0.0 else tag


def process_one_mdp_bundle(
    label: str,
    env: FrozenLakeEnv,
    mdp: MDPNetwork,
    out_dir: Path,
    native_mdp: Optional[MDPNetwork] = None,       # for cross-evaluation on the native environment
    native_occ_random: Optional["ValueTable"] = None,  # baseline random occupancy on native (for diff plots)
    opt_policy_mixing: Tuple[float, float, float] = LOOP_OPT_POLICY_MIXING,  # mapping Q*->policy on THIS mdp
    softmax_temperature: float = SOFTMAX_TEMPERATURE,
    tie_tol: float = OPT_TIE_TOL,
):
    """
    For a single MDP:
      - Plot per-action transition overlays.
      - Compute occupancy for random policy and optimal mixed policy (on this MDP).
      - Plot both occupancy overlays.
      - If native_mdp is provided: cross-evaluate the learned optimal mixed policy on the native MDP and plot.
      - NEW: If native_occ_random is provided, also plot the difference
             (occ_opt_on_native − native_random) on the native MDP.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Per-action transition overlays
    plot_frozenlake_transition_overlays(
        env=env,
        mdp=mdp,
        output_dir=str(out_dir),
        filename_prefix=label,
        min_prob=MIN_PROB,
        alpha=ALPHA,
        annotate=True,
        show_self_loops=SHOW_SELF_LOOPS,
        dpi=DPI,
    )
    print(f"[OK] Saved transition overlays for '{label}'")

    # 2) Policies and occupancy measures on THIS mdp
    # 2a) Random (uniform) policy
    policy_rand = create_random_policy(mdp)
    occ_rand = compute_occupancy_measure(
        mdp_network=mdp,
        policy=policy_rand,
        gamma=OCC_GAMMA,
        theta=OCC_THETA,
        max_iterations=OCC_MAX_ITERS,
        verbose=False,
    )

    # 2b) Optimal policy via DP (value iteration → Q*; then map with MIXING)
    V_star, Q_star = optimal_value_iteration(
        mdp_network=mdp,
        gamma=OCC_GAMMA,
        theta=OCC_THETA,
        max_iterations=OCC_MAX_ITERS,
        verbose=False,
    )
    policy_opt = q_table_to_policy(
        q_table=Q_star,
        states=mdp.states,
        num_actions=mdp.num_actions,
        mixing=opt_policy_mixing,         # mixing controls uniform/softmax/greedy weights
        temperature=softmax_temperature,  # used only if mixing softmax weight > 0
        tie_tol=tie_tol,
    )
    occ_opt = compute_occupancy_measure(
        mdp_network=mdp,
        policy=policy_opt,
        gamma=OCC_GAMMA,
        theta=OCC_THETA,
        max_iterations=OCC_MAX_ITERS,
        verbose=False,
    )

    # 3) Occupancy overlays (random vs optimal) ON THIS mdp
    plot_frozenlake_occupancy_overlays(
        env=env,
        occupancy_matrix=occ_rand,
        output_dir=str(out_dir),
        filename_prefix=f"{label}_occupancy_random",
        alpha=OCC_ALPHA,
        annotate=True,
        dpi=DPI,
        target_cell_px=OCC_TARGET_CELL_PX,
        font_scale=OCC_FONT_SCALE,
        cmap_name=OCC_CMAP,
        gamma=OCC_COLOR_GAMMA,
        min_label_value=OCC_MIN_LABEL,
        vmin=OCC_VMIN,
        vmax=OCC_VMAX,
    )

    mix_suffix = _mix_suffix(opt_policy_mixing, softmax_temperature)
    plot_frozenlake_occupancy_overlays(
        env=env,
        occupancy_matrix=occ_opt,
        output_dir=str(out_dir),
        filename_prefix=f"{label}_occupancy_optimal_{mix_suffix}",
        alpha=OCC_ALPHA,
        annotate=True,
        dpi=DPI,
        target_cell_px=OCC_TARGET_CELL_PX,
        font_scale=OCC_FONT_SCALE,
        cmap_name=OCC_CMAP,
        gamma=OCC_COLOR_GAMMA,
        min_label_value=OCC_MIN_LABEL,
        vmin=OCC_VMIN,
        vmax=OCC_VMAX,
    )

    # 4) Cross-evaluate the learned optimal mixed policy on the NATIVE MDP (if provided)
    if native_mdp is not None:
        occ_cross_native = compute_occupancy_measure(
            mdp_network=native_mdp,
            policy=policy_opt,          # learned on 'mdp', evaluated on 'native_mdp'
            gamma=OCC_GAMMA,
            theta=OCC_THETA,
            max_iterations=OCC_MAX_ITERS,
            verbose=False,
        )
        plot_frozenlake_occupancy_overlays(
            env=env,
            occupancy_matrix=occ_cross_native,
            output_dir=str(out_dir),
            filename_prefix=f"{label}_occupancy_optPolicy_on_NATIVE_{mix_suffix}",
            alpha=OCC_ALPHA,
            annotate=True,
            dpi=DPI,
            target_cell_px=OCC_TARGET_CELL_PX,
            font_scale=OCC_FONT_SCALE,
            cmap_name=OCC_CMAP,
            gamma=OCC_COLOR_GAMMA,
            min_label_value=OCC_MIN_LABEL,
            vmin=OCC_VMIN,
            vmax=OCC_VMAX,
        )

        # 5) NEW: Difference plot on native: (learned policy on native) - (native random policy)
        if native_occ_random is not None:
            plot_frozenlake_occupancy_diff_overlay(
                env=env,
                occupancy_a=occ_cross_native,
                occupancy_b=native_occ_random,
                output_dir=str(out_dir),
                filename_prefix=f"{label}_occupancy_DIFF_optPolicyMINUS_nativeRandom_{mix_suffix}",
                alpha=OCC_ALPHA,
                annotate=True,
                dpi=DPI,
                target_cell_px=OCC_TARGET_CELL_PX,
                font_scale=OCC_FONT_SCALE,
                cmap_name="coolwarm",            # diverging colormap for signed differences
                min_abs_label=0.0,               # label threshold for |Δ|
                vmin=None,                       # auto symmetric range
                vmax=None,
            )

    print(f"[OK] Saved occupancy overlays for '{label}' -> {out_dir}")


def main():
    # Resolve and prepare paths
    json_dir = JSON_DIR.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the env once; all MDPs must match this grid size.
    env = ensure_env(MAP_NAME, IS_SLIPPERY)
    nS = env.nrow * env.ncol
    print(f"[INFO] Env grid: {env.nrow}x{env.ncol} ({nS} states).")

    # === (A) Native FrozenLake MDP first ===
    native_mdp = None
    native_occ_random = None
    try:
        native_mdp = env.get_mdp_network()  # provided by CustomisedFrozenLakeEnv
        # Pre-compute native random baseline occupancy for later diffs
        native_policy_rand = create_random_policy(native_mdp)
        native_occ_random = compute_occupancy_measure(
            mdp_network=native_mdp,
            policy=native_policy_rand,
            gamma=OCC_GAMMA,
            theta=OCC_THETA,
            max_iterations=OCC_MAX_ITERS,
            verbose=False,
        )

        native_out = output_dir / NATIVE_SUBDIR_NAME

        # Native: optimal policy is GREEDY by default (mix=(0,0,1))
        process_one_mdp_bundle(
            label=NATIVE_PREFIX,
            env=env,
            mdp=native_mdp,
            out_dir=native_out,
            native_mdp=None,               # no cross-eval for native-on-native
            native_occ_random=None,        # diff not required for native bundle
            opt_policy_mixing=NATIVE_OPT_POLICY_MIXING,
            softmax_temperature=SOFTMAX_TEMPERATURE,
            tie_tol=OPT_TIE_TOL,
        )
    except Exception as e:
        print(f"[WARN] Failed to build or plot native FrozenLake MDP: {e}")

    # === (B) Then process JSON-defined MDPs ===
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

            # Sanity checks
            if getattr(mdp, "num_actions", None) != 4:
                print(f"[WARN] Skip (num_actions != 4): {jf}")
                continue
            if not states_aligned(env, mdp):
                print(f"[WARN] Skip (states not aligned to 0..{nS-1}): {jf}")
                continue

            # Dedicated output folder per JSON
            stem = jf.stem
            out_subdir = output_dir / stem

            # JSON MDP: default to epsilon-greedy 0.1/0.9 (parameterized)
            process_one_mdp_bundle(
                label=stem,
                env=env,
                mdp=mdp,
                out_dir=out_subdir,
                native_mdp=native_mdp,               # cross-evaluate learned policy on the native MDP
                native_occ_random=native_occ_random, # provide baseline random occupancy for diff
                opt_policy_mixing=LOOP_OPT_POLICY_MIXING,
                softmax_temperature=SOFTMAX_TEMPERATURE,
                tie_tol=OPT_TIE_TOL,
            )

        except Exception as e:
            print(f"[ERROR] Failed on {jf}: {e}")

    print(f"[DONE] All finished. Outputs in: {output_dir}")

    # Optional: close env
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
