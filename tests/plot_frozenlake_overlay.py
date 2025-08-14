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
        (b) an optimal policy from dynamic programming (value iteration -> Q*, then softmax with temperature SOFTMAX_TEMPERATURE).
    - Plots occupancy overlays for both policies.

(B) For each JSON MDP, the script now:
    - Plots transition overlays for that MDP.
    - Computes occupancy for
        (a) a random (uniform) policy, and
        (b) an optimal policy (Q* -> softmax with temperature SOFTMAX_TEMPERATURE) ON THE SAME JSON MDP.
    - Plots occupancy overlays for both policies.
    - NEW: Cross-evaluates the learned optimal softmax policy on the NATIVE MDP and plots its occupancy overlay as well.

All temperatures are controlled by the float constant: SOFTMAX_TEMPERATURE.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from gymnasium.envs.toy_text import FrozenLakeEnv

# Project modules (adjust if your paths differ)
from customised_toy_text_envs.customised_frozenlake import (
    CustomisedFrozenLakeEnv,               # used to get native MDP from env
    plot_frozenlake_transition_overlays,
    plot_frozenlake_occupancy_overlays,   # occupancy overlay renderer
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
SOFTMAX_TEMPERATURE: float = 0.1              # <== adjustable float: Q* -> policy via softmax(T)

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


def process_one_mdp_bundle(
    label: str,
    env: FrozenLakeEnv,
    mdp: MDPNetwork,
    out_dir: Path,
    native_mdp: Optional[MDPNetwork] = None,   # NEW: for cross-evaluation on the native environment
):
    """
    For a single MDP:
      - Plot per-action transition overlays.
      - Compute occupancy for random policy and optimal softmax policy (on this MDP).
      - Plot both occupancy overlays.
      - If native_mdp is provided: cross-evaluate the learned optimal softmax policy on the native MDP and plot.
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

    # 2b) Optimal policy via DP (value iteration → Q*; then softmax with configurable temperature)
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
        temperature=SOFTMAX_TEMPERATURE,
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

    plot_frozenlake_occupancy_overlays(
        env=env,
        occupancy_matrix=occ_opt,
        output_dir=str(out_dir),
        filename_prefix=f"{label}_occupancy_optimal_softmaxT{SOFTMAX_TEMPERATURE:g}",
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

    # 4) NEW: Cross-evaluate the learned optimal softmax policy on the NATIVE MDP (if provided)
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
            filename_prefix=f"{label}_occupancy_optPolicy_on_NATIVE_softmaxT{SOFTMAX_TEMPERATURE:g}",
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
    try:
        native_mdp = env.get_mdp_network()  # provided by CustomisedFrozenLakeEnv
        native_out = output_dir / NATIVE_SUBDIR_NAME
        process_one_mdp_bundle(
            label=NATIVE_PREFIX,
            env=env,
            mdp=native_mdp,
            out_dir=native_out,
            native_mdp=None,          # Do not cross-eval the native policy on native (would be redundant)
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

            process_one_mdp_bundle(
                label=stem,
                env=env,
                mdp=mdp,
                out_dir=out_subdir,
                native_mdp=native_mdp,   # NEW: pass native for cross-evaluation
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
