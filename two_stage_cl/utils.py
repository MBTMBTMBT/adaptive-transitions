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
):
    """
    Plot greedy and train-policy curves (mean±std) with vertical phase boundary markers.
    baseline: {"greedy_mean","greedy_std","train_mean","train_std"}
    curves_target / curves_source: same keys as baseline
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Titles
    ax1.set_title(f"{title_prefix} (Greedy)")
    ax2.set_title(f"{title_prefix} (Training-policy)")

    # Greedy
    ax1.plot(checkpoints, baseline["greedy_mean"], label="Target-only baseline", linewidth=1.8)
    ax1.fill_between(checkpoints, baseline["greedy_mean"] - baseline["greedy_std"],
                     baseline["greedy_mean"] + baseline["greedy_std"], alpha=0.2)
    ax1.plot(checkpoints, curves_target["greedy_mean"], label="Curriculum → Target (primary)", linewidth=2.2)
    ax1.fill_between(checkpoints, curves_target["greedy_mean"] - curves_target["greedy_std"],
                     curves_target["greedy_mean"] + curves_target["greedy_std"], alpha=0.15)
    # if curves_source is not None:
    #     ax1.plot(checkpoints, curves_source["greedy_mean"], label="Curriculum (eval on Source)", linewidth=1.6)
    #     ax1.fill_between(checkpoints, curves_source["greedy_mean"] - curves_source["greedy_std"],
    #                      curves_source["greedy_mean"] + curves_source["greedy_std"], alpha=0.15)
    ax1.set_xlabel("Timesteps"); ax1.set_ylabel("Mean return"); ax1.grid(True, alpha=0.3); ax1.legend()

    # Train-policy
    ax2.plot(checkpoints, baseline["train_mean"], label="Target-only baseline", linewidth=1.8)
    ax2.fill_between(checkpoints, baseline["train_mean"] - baseline["train_std"],
                     baseline["train_mean"] + baseline["train_std"], alpha=0.2)
    ax2.plot(checkpoints, curves_target["train_mean"], label="Curriculum → Target (primary)", linewidth=2.2)
    ax2.fill_between(checkpoints, curves_target["train_mean"] - curves_target["train_std"],
                     curves_target["train_mean"] + curves_target["train_std"], alpha=0.15)
    # if curves_source is not None:
    #     ax2.plot(checkpoints, curves_source["train_mean"], label="Curriculum (eval on Source)", linewidth=1.6)
    #     ax2.fill_between(checkpoints, curves_source["train_mean"] - curves_source["train_std"],
    #                      curves_source["train_mean"] + curves_source["train_std"], alpha=0.15)
    ax2.set_xlabel("Timesteps"); ax2.set_ylabel("Mean return"); ax2.grid(True, alpha=0.3); ax2.legend()

    # Phase boundary markers
    for ax in (ax1, ax2):
        for b in phase_boundaries:
            ax.axvline(b, linestyle="--", alpha=0.7)
        ymin, ymax = ax.get_ylim()
        ytxt = ymin + 0.06 * (ymax - ymin)
        # Optional labels: show first-half vs second-half regions
        if len(phase_boundaries) >= 1:
            left_mid = phase_boundaries[0] * 0.5
            right_mid = phase_boundaries[-1] + (checkpoints[-1] - phase_boundaries[-1]) * 0.5
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