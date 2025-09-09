from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

import numpy as np


def _cap_curve(steps: np.ndarray, vals: np.ndarray, max_step: Optional[int]):
    xs, ys = np.asarray(steps, int), np.asarray(vals, float)
    if max_step is None or xs[-1] <= max_step:
        return xs, ys
    mask = xs <= max_step
    xs2, ys2 = xs[mask], ys[mask]
    if xs2.size == 0:
        return np.array([max_step], int), np.array(
            [float(np.interp(max_step, xs, ys))], float
        )
    if xs2[-1] < max_step:
        y_at = float(np.interp(max_step, xs, ys))
        xs2 = np.concatenate([xs2, np.array([max_step], int)])
        ys2 = np.concatenate([ys2, np.array([y_at], float)])
    return xs2, ys2


def _cap(xs: np.ndarray, ys: np.ndarray, max_step: Optional[int]):
    if max_step is None:
        return xs, ys
    if xs.size == 0 or max_step < xs[0]:
        return xs[:0], ys[:0]  # empty -> undefined
    mask = xs <= max_step
    X = xs[mask]
    Y = ys[mask]
    if X.size == 0:
        return X, Y
    if X[-1] < max_step and xs[-1] >= max_step:
        y_at = float(np.interp(max_step, xs, ys))
        X = np.concatenate([X, np.array([max_step], int)])
        Y = np.concatenate([Y, np.array([y_at], float)])
    return X, Y


def _mean_over(xs, ys, lo: Optional[float] = None, hi: Optional[float] = None,
               clamp_hi: Optional[float] = None):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if xs.size < 1 or ys.size != xs.size:
        return None

    x0, x1 = float(xs[0]), float(xs[-1])

    # Default interval = full support
    lo = x0 if lo is None else float(lo)
    hi = x1 if hi is None else float(hi)

    # Optional global clamp on the right (used by mean_total with use_max_step)
    if clamp_hi is not None:
        hi = min(float(clamp_hi), hi)

    # Clip to support
    lo = max(lo, x0)
    hi = min(hi, x1)
    if not (hi > lo):
        return None

    # Interpolate endpoints
    y_lo = float(np.interp(lo, xs, ys))
    y_hi = float(np.interp(hi, xs, ys))

    # Interior samples strictly inside (lo, hi)
    mask = (xs > lo) & (xs < hi)
    vals = [y_lo, *ys[mask].tolist(), y_hi]

    return float(np.mean(vals)) if len(vals) > 0 else None


def _ap_last_k(xs, ys, k, max_step):
    X, Y = _cap(xs, ys, max_step)
    if Y.size < 1:
        return None
    k = max(1, min(int(k), int(Y.size)))
    return float(np.mean(Y[-k:]))


def _ttt_frac(xs, ys, frac, max_step):
    X, Y = _cap(xs, ys, max_step)
    if Y.size < 1:
        return None
    thr = float(np.max(Y)) * float(frac)
    idx = np.where(Y >= thr)[0]
    return int(X[int(idx[0])]) if idx.size > 0 else None


def _interp_at(xs, ys, s):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if xs.size < 1 or ys.size != xs.size:
        return None
    if s < xs[0] or s > xs[-1]:
        return None
    return float(np.interp(float(s), xs, ys))


def _ensure_curve(block: Dict[str, Any], curve_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) for curve_name; raise on structural errors."""
    assert "steps" in block and curve_name in block, f"missing '{curve_name}' or steps"
    xs = np.asarray(block["steps"], int)
    ys = np.asarray(block[curve_name], float)
    if xs.ndim != 1 or ys.ndim != 1 or xs.size != ys.size:
        raise ValueError(f"shape mismatch in '{curve_name}' series")
    return xs, ys

def _auc_over(xs, ys, lo: Optional[float] = None, hi: Optional[float] = None,
              clamp_hi: Optional[float] = None):
    """
    Trapezoidal area over [lo, hi], with endpoint interpolation.
    If clamp_hi is not None, hi = min(hi, clamp_hi) before clipping to support.
    Returns None if interval is empty or insufficient.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if xs.size < 1 or ys.size != xs.size:
        return None

    x0, x1 = float(xs[0]), float(xs[-1])
    lo = x0 if lo is None else float(lo)
    hi = x1 if hi is None else float(hi)
    if clamp_hi is not None:
        hi = min(float(clamp_hi), hi)

    lo = max(lo, x0)
    hi = min(hi, x1)
    if not (hi > lo):
        return None

    y_lo = float(np.interp(lo, xs, ys))
    y_hi = float(np.interp(hi, xs, ys))
    mask = (xs > lo) & (xs < hi)
    X = np.concatenate(([lo], xs[mask], [hi]))
    Y = np.concatenate(([y_lo], ys[mask], [y_hi]))
    if X.size < 2:
        return None
    return float(np.trapz(Y, X))

def _jumpstart_fields(
    tgt_block: Dict[str, Any],
    baseline_target: Optional[Dict[str, Any]],
    B: Optional[int],
    chan: str,
    first_n: int,
) -> Dict[str, Optional[float]]:
    """
    Compute absolute jumpstart fields for a given channel.
    Returns:
      {
        "target_start": float|None,   # first value on target curve
        "p2_head": float|None,        # mean of first_n samples in phase-2 including boundary @B
        "baseline_B": float|None,     # baseline Target interpolated at B (if baseline is provided)
      }
    Notes:
      - If B is out of [min_step, max_step] of the target curve, p2_head is None.
      - first_n must be >= 1.
    """
    if tgt_block is None or B is None:
        return {"target_start": None, "p2_head": None, "baseline_B": None}

    xs, ys = _ensure_curve(tgt_block, f"{chan}_mean")
    if xs.size < 1:
        return {"target_start": None, "p2_head": None, "baseline_B": None}

    target_start = float(ys[0])

    # p2_head: include boundary B and subsequent samples (> B), then average first_n
    if xs[0] <= B <= xs[-1]:
        yB = float(np.interp(B, xs, ys))
        mask_p2 = xs > B
        Yp2 = [yB] + ys[mask_p2].tolist()
        n = min(int(first_n), len(Yp2))
        p2_head = float(np.mean(Yp2[:n])) if n > 0 else None
    else:
        p2_head = None

    # baseline_B: interpolate on baseline target if provided
    if baseline_target is not None:
        bx, by = _ensure_curve(baseline_target, f"{chan}_mean")
        baseline_B = _interp_at(bx, by, B)
    else:
        baseline_B = None

    return {"target_start": target_start, "p2_head": p2_head, "baseline_B": baseline_B}

