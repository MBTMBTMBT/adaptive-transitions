from __future__ import annotations

import csv
from typing import List, Dict, Optional

import numpy as np
from matplotlib import pyplot as plt


def plot_pairwise(
    out_png_path: str,
    checkpoints: np.ndarray,
    phase_boundaries: List[int],
    title_prefix: str,
    baseline: Dict[str, np.ndarray],
    curves_target: Dict[str, np.ndarray],
    curves_source: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """
    Plot greedy and train-policy curves (mean±std).
    Also draw vertical phase boundary lines at the provided absolute timesteps.

    Args:
        out_png_path: where to save the PNG.
        checkpoints: x-axis points (global timesteps) used for plotting the curves.
        phase_boundaries: absolute global timesteps marking phase ends (excluding final end).
        title_prefix: figure title prefix.
        baseline/curves_target/curves_source: dicts with keys:
            - 'greedy_mean', 'greedy_std', 'train_mean', 'train_std'
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Titles
    ax1.set_title(f"{title_prefix} (Greedy)")
    ax2.set_title(f"{title_prefix} (Training-policy)")

    # --- Greedy curves ---
    ax1.plot(checkpoints, baseline["greedy_mean"], label="Target-only baseline", linewidth=1.8)
    ax1.fill_between(checkpoints,
                     baseline["greedy_mean"] - baseline["greedy_std"],
                     baseline["greedy_mean"] + baseline["greedy_std"],
                     alpha=0.2)
    ax1.plot(checkpoints, curves_target["greedy_mean"], label="Curriculum → Target (primary)", linewidth=2.2)
    ax1.fill_between(checkpoints,
                     curves_target["greedy_mean"] - curves_target["greedy_std"],
                     curves_target["greedy_mean"] + curves_target["greedy_std"],
                     alpha=0.15)
    if curves_source is not None:
        ax1.plot(checkpoints, curves_source["greedy_mean"], label="Curriculum (eval on Source)", linewidth=1.6)
        ax1.fill_between(checkpoints,
                         curves_source["greedy_mean"] - curves_source["greedy_std"],
                         curves_source["greedy_mean"] + curves_source["greedy_std"],
                         alpha=0.12)

    ax1.set_xlabel("Timesteps"); ax1.set_ylabel("Mean return"); ax1.grid(True, alpha=0.3); ax1.legend()

    # --- Train-policy curves ---
    ax2.plot(checkpoints, baseline["train_mean"], label="Target-only baseline", linewidth=1.8)
    ax2.fill_between(checkpoints,
                     baseline["train_mean"] - baseline["train_std"],
                     baseline["train_mean"] + baseline["train_std"],
                     alpha=0.2)
    ax2.plot(checkpoints, curves_target["train_mean"], label="Curriculum → Target (primary)", linewidth=2.2)
    ax2.fill_between(checkpoints,
                     curves_target["train_mean"] - curves_target["train_std"],
                     curves_target["train_mean"] + curves_target["train_std"],
                     alpha=0.15)
    if curves_source is not None:
        ax2.plot(checkpoints, curves_source["train_mean"], label="Curriculum (eval on Source)", linewidth=1.6)
        ax2.fill_between(checkpoints,
                         curves_source["train_mean"] - curves_source["train_std"],
                         curves_source["train_mean"] + curves_source["train_std"],
                         alpha=0.12)

    ax2.set_xlabel("Timesteps"); ax2.set_ylabel("Mean return"); ax2.grid(True, alpha=0.3); ax2.legend()

    # --- Vertical phase boundary markers (clamped to x-limits) ---
    x_min, x_max = int(np.min(checkpoints)), int(np.max(checkpoints))
    b_valid = [int(b) for b in (phase_boundaries or []) if x_min <= int(b) <= x_max]

    for ax in (ax1, ax2):
        for b in b_valid:
            ax.axvline(b, linestyle="--", alpha=0.7)

        # Optional zone labels (similar to the old version)
        if b_valid:
            ymin, ymax = ax.get_ylim()
            ytxt = ymin + 0.06 * (ymax - ymin)
            left_mid = b_valid[0] * 0.5
            right_mid = b_valid[-1] + (x_max - b_valid[-1]) * 0.5
            ax.text(left_mid, ytxt, "Phase 1 (Source)", ha="center", va="bottom", fontsize=9, alpha=0.8)
            ax.text(right_mid, ytxt, "Later Phases (Target/others)", ha="center", va="bottom", fontsize=9, alpha=0.8)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)


def save_csv(path: str, steps: np.ndarray, mean: np.ndarray, std: np.ndarray, header: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([header])
        w.writerow(["step", "mean", "std"])
        for s, m, sd in zip(steps.tolist(), mean.tolist(), std.tolist()):
            w.writerow([int(s), float(m), float(sd)])