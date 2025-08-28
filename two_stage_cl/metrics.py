from __future__ import annotations

from typing import Optional

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


def _auc(xs: np.ndarray, ys: np.ndarray, max_step: Optional[int] = None) -> float:
    xs2, ys2 = _cap_curve(xs, ys, max_step)
    return float(np.trapz(ys2, xs2)) if len(xs2) >= 2 else 0.0


def _ap(
    xs: np.ndarray, ys: np.ndarray, last_k: int = 10, max_step: Optional[int] = None
) -> float:
    xs2, ys2 = _cap_curve(xs, ys, max_step)
    k = max(1, min(int(last_k), len(ys2)))
    return float(np.mean(ys2[-k:]))


def _ttt(
    xs: np.ndarray, ys: np.ndarray, frac: float = 0.9, max_step: Optional[int] = None
) -> Optional[int]:
    xs2, ys2 = _cap_curve(xs, ys, max_step)
    if len(xs2) == 0:
        return None
    thr = float(np.max(ys2)) * float(frac)
    idx = np.where(ys2 >= thr)[0]
    return int(xs2[int(idx[0])]) if idx.size > 0 else None


def _value_at(xs: np.ndarray, ys: np.ndarray, at_step: int) -> float:
    xs, ys = np.asarray(xs, int), np.asarray(ys, float)
    if at_step <= xs[0]:
        return float(ys[0])
    if at_step >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(at_step, xs, ys))
