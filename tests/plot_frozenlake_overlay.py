import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

from gymnasium.envs.toy_text import FrozenLakeEnv

# Your project modules
from customised_toy_text_envs.customised_frozenlake import plot_frozenlake_transition_overlays
from mdp_network import MDPNetwork


# -----------------------------
# Hard-coded constants
# -----------------------------
JSON_DIR: Path = Path("./outputs/ga_test")      # Directory containing MDP JSON files
OUTPUT_DIR: Path = Path("./outputs/ga_vis")      # Output directory for images
MAP_NAME: str = "8x8"                     # "4x4" or "8x8"
IS_SLIPPERY: bool = True                  # Background env dynamics flag
RECURSIVE: bool = True                    # Whether to search subdirectories for JSONs

# Overlay/plot style
MIN_PROB: float = 0.05                    # Min probability threshold to draw arrows
ALPHA: float = 0.65                       # Transparency for arrows and labels
SHOW_SELF_LOOPS: bool = False             # Draw self-loop arcs for s->s
DPI: int = 200                            # Figure DPI


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


def ensure_env(map_name: str, is_slippery: bool) -> FrozenLakeEnv:
    """
    Build a FrozenLakeEnv with rgb_array capability for background rendering.
    """
    # Force rgb_array so we can capture the board image.
    env = FrozenLakeEnv(render_mode="rgb_array", map_name=map_name, is_slippery=is_slippery)
    # Reset to initialize internal state, then place the token consistently if possible.
    try:
        env.reset()
        # Prefer the most likely start (often 0) for a clean board.
        if hasattr(env, "initial_state_distrib"):
            env.s = int(env.initial_state_distrib.argmax())
    except Exception:
        pass
    return env


def states_aligned(env: FrozenLakeEnv, mdp: MDPNetwork) -> bool:
    """
    Check whether mdp.states equals {0..nS-1}.
    """
    nS = env.nrow * env.ncol
    return set(mdp.states) == set(range(nS))


def main():
    # Resolve and prepare paths
    json_dir = JSON_DIR.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = find_json_files(json_dir, RECURSIVE)
    if not json_files:
        print(f"[WARN] No JSON files found under: {json_dir}")
        sys.exit(0)

    # Create the env once; all MDPs must match this grid size.
    env = ensure_env(MAP_NAME, IS_SLIPPERY)
    nS = env.nrow * env.ncol

    print(f"[INFO] Found {len(json_files)} JSON file(s). Env grid: {env.nrow}x{env.ncol} ({nS} states).")

    for jf in json_files:
        print(f"[INFO] Processing: {jf}")
        cfg = load_json(jf)
        mdp = MDPNetwork(config_data=cfg)

        # Sanity: 4 actions expected
        if getattr(mdp, "num_actions", None) != 4:
            print(f"[WARN] Skip (num_actions != 4): {jf}")
            continue

        # Sanity: states alignment
        if not states_aligned(env, mdp):
            print(f"[WARN] Skip (states not aligned to 0..{nS-1}): {jf}")
            continue

        # Dedicated output folder per JSON
        stem = jf.stem
        out_subdir = output_dir / stem
        out_subdir.mkdir(parents=True, exist_ok=True)

        # Call the overlay function
        plot_frozenlake_transition_overlays(
            env=env,
            mdp=mdp,
            output_dir=str(out_subdir),
            filename_prefix=stem,
            min_prob=MIN_PROB,
            alpha=ALPHA,
            annotate=True,
            show_self_loops=SHOW_SELF_LOOPS,
            dpi=DPI,
        )

        print(f"[OK] Saved overlays for '{stem}' -> {out_subdir}")


    print(f"[DONE] All finished. Outputs in: {output_dir}")

    # Optional: close env
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
