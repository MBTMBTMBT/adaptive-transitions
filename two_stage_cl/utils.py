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
    Plot greedy and train-policy curves (mean ± std) and draw vertical phase
    boundary lines at the provided absolute timesteps.

    Notes on robustness:
    - Do NOT cast x-range or boundaries to int; keep float to avoid truncation
      (interpolated checkpoints may be non-integers).
    - Use a tiny epsilon tolerance when filtering boundaries by x-limits.
    - Draw boundary lines with high zorder so they are not covered by fill_between.
    """

    # --- Coerce to numpy arrays (defensive) ---
    checkpoints = np.asarray(checkpoints, dtype=float)
    x_min = float(np.min(checkpoints))
    x_max = float(np.max(checkpoints))
    # Tiny tolerance to account for float round-off during interpolation/union
    eps = 1e-9 * max(1.0, (x_max - x_min))

    # --- Figure & axes ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Titles
    ax1.set_title(f"{title_prefix} (Greedy)")
    ax2.set_title(f"{title_prefix} (Training-policy)")

    # ----------- Greedy curves -----------
    ax1.plot(
        checkpoints,
        baseline["greedy_mean"],
        label="Target-only baseline",
        linewidth=1.8,
    )
    ax1.fill_between(
        checkpoints,
        np.asarray(baseline["greedy_mean"]) - np.asarray(baseline["greedy_std"]),
        np.asarray(baseline["greedy_mean"]) + np.asarray(baseline["greedy_std"]),
        alpha=0.2,
    )
    ax1.plot(
        checkpoints,
        curves_target["greedy_mean"],
        label="Curriculum → Target (primary)",
        linewidth=2.2,
    )
    ax1.fill_between(
        checkpoints,
        np.asarray(curves_target["greedy_mean"])
        - np.asarray(curves_target["greedy_std"]),
        np.asarray(curves_target["greedy_mean"])
        + np.asarray(curves_target["greedy_std"]),
        alpha=0.15,
    )
    if curves_source is not None:
        ax1.plot(
            checkpoints,
            curves_source["greedy_mean"],
            label="Curriculum (eval on Source)",
            linewidth=1.6,
        )
        ax1.fill_between(
            checkpoints,
            np.asarray(curves_source["greedy_mean"])
            - np.asarray(curves_source["greedy_std"]),
            np.asarray(curves_source["greedy_mean"])
            + np.asarray(curves_source["greedy_std"]),
            alpha=0.12,
        )
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Mean return")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ----------- Train-policy curves -----------
    ax2.plot(
        checkpoints, baseline["train_mean"], label="Target-only baseline", linewidth=1.8
    )
    ax2.fill_between(
        checkpoints,
        np.asarray(baseline["train_mean"]) - np.asarray(baseline["train_std"]),
        np.asarray(baseline["train_mean"]) + np.asarray(baseline["train_std"]),
        alpha=0.2,
    )
    ax2.plot(
        checkpoints,
        curves_target["train_mean"],
        label="Curriculum → Target (primary)",
        linewidth=2.2,
    )
    ax2.fill_between(
        checkpoints,
        np.asarray(curves_target["train_mean"])
        - np.asarray(curves_target["train_std"]),
        np.asarray(curves_target["train_mean"])
        + np.asarray(curves_target["train_std"]),
        alpha=0.15,
    )
    if curves_source is not None:
        ax2.plot(
            checkpoints,
            curves_source["train_mean"],
            label="Curriculum (eval on Source)",
            linewidth=1.6,
        )
        ax2.fill_between(
            checkpoints,
            np.asarray(curves_source["train_mean"])
            - np.asarray(curves_source["train_std"]),
            np.asarray(curves_source["train_mean"])
            + np.asarray(curves_source["train_std"]),
            alpha=0.12,
        )
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Mean return")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # ----------- Vertical phase boundary markers -----------
    # Lock x-limits first so later autoscale won't hide lines
    for ax in (ax1, ax2):
        ax.set_xlim(x_min, x_max)

    # Keep boundaries that fall within the visible x-range (with tolerance)
    b_valid = []
    if phase_boundaries:
        for b in phase_boundaries:
            try:
                bb = float(b)
            except Exception:
                continue
            if (x_min - eps) <= bb <= (x_max + eps):
                b_valid.append(bb)

    # Draw lines with solid color + high zorder to ensure visibility
    for ax in (ax1, ax2):
        for b in b_valid:
            ax.axvline(
                b, linestyle="--", color="k", linewidth=1.6, alpha=0.9, zorder=10
            )

        # Optional zone labels (kept consistent with the older version)
        if b_valid:
            ymin, ymax = ax.get_ylim()
            ytxt = ymin + 0.06 * (ymax - ymin)
            left_mid = (b_valid[0] + x_min) * 0.5
            right_mid = (b_valid[-1] + x_max) * 0.5
            ax.text(
                left_mid,
                ytxt,
                "Phase 1 (Source)",
                ha="center",
                va="bottom",
                fontsize=9,
                alpha=0.85,
            )
            ax.text(
                right_mid,
                ytxt,
                "Later Phases (Target/others)",
                ha="center",
                va="bottom",
                fontsize=9,
                alpha=0.85,
            )

    # ----------- Save & close -----------
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)


def save_csv(
    path: str, steps: np.ndarray, mean: np.ndarray, std: np.ndarray, header: str
) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([header])
        w.writerow(["step", "mean", "std"])
        for s, m, sd in zip(steps.tolist(), mean.tolist(), std.tolist()):
            w.writerow([int(s), float(m), float(sd)])
