# customised_simplegrid.py

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Arc

from PIL import Image  # optional upscale
_HAS_PIL = True

from gym_simplegrid.envs import SimpleGridEnv
from apis.customisable import CustomisableEnvAbs
from mdp_network.mdp_network import MDPNetwork


MAPS = {
    "4x4": ["0000", "0101", "0001", "1000"],
    "8x8": [
        "00000000",
        "00000000",
        "00010000",
        "00000100",
        "00010000",
        "01100010",
        "01001010",
        "00010000",
    ],
}

# Global action names (match SimpleGridEnv.MOVES order 0..3)
ACTION_NAMES: Tuple[str, str, str, str] = ("UP", "DOWN", "LEFT", "RIGHT")


class CustomisedSimpleGridEnv(SimpleGridEnv, CustomisableEnvAbs):
    """
    SimpleGridEnv with:
      - Optional delegation to an external NetworkX/MDP backend.
      - Encode/decode integer state: s = row * ncol + col.
      - MDP export to MDPNetwork.
    """

    def __init__(
            self,
            obstacle_map: str | list[str],
            render_mode: str | None = None,
            networkx_env: Any = None,
            use_original_rewards: bool = False,
            start_candidates: Optional[list[int | tuple]] = None,
            goal_candidates: Optional[list[int | tuple]] = None,
    ):
        resolved_map = MAPS.get(obstacle_map, obstacle_map) if isinstance(obstacle_map, str) else obstacle_map

        # Call base ctor with the resolved map
        super().__init__(obstacle_map=resolved_map, render_mode=render_mode)

        # Existing fields unchanged
        self.networkx_env = networkx_env
        self.use_original_rewards = bool(use_original_rewards)
        self.start_candidates_raw = list(start_candidates or [])
        self.goal_candidates_raw = list(goal_candidates or [])
        self.initial_state_distrib: Optional[np.ndarray] = None

    def parse_state_option(self, state_name: str, options: dict) -> tuple:
        """
        Priority: explicit options[state_name] -> candidate list -> default sampler.
        Accept int (state index) or (row,col) tuples.
        """
        # 1) explicit option
        if isinstance(options, dict) and state_name in options:
            v = options[state_name]
            if isinstance(v, int):
                return self.to_xy(int(v))
            if isinstance(v, tuple) and len(v) == 2:
                return (int(v[0]), int(v[1]))
            raise TypeError(f"Allowed types for `{state_name}` are int or tuple.")

        # 2) candidate lists (uniform over valid)
        raw = self.start_candidates_raw if state_name == "start_loc" else (
            self.goal_candidates_raw if state_name == "goal_loc" else []
        )
        if raw:
            valids: list[tuple[int, int]] = []
            for v in raw:
                if isinstance(v, int):
                    r, c = self.to_xy(int(v))
                elif isinstance(v, tuple) and len(v) == 2:
                    r, c = int(v[0]), int(v[1])
                else:
                    continue
                if self.is_in_bounds(r, c) and self.is_free(r, c):
                    valids.append((r, c))
            if valids:
                idx = int(self.np_random.integers(0, len(valids)))
                return valids[idx]

        # 3) fallback: original sampler
        state = self.sample_valid_state_xy()
        print(f"Key `{state_name}` not provided; sampled: {state}")
        return state

    # -------------------------
    # Core overrides
    # -------------------------
    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None,
    ):
        """
        Support options['start_loc'/'goal_loc'] and candidate lists.
        initial_state_distrib:
          - If options has start_loc: one-hot at that start.
          - Else if start_candidates provided: uniform over valid candidates.
          - Else: one-hot at sampled start.
        """
        options = options or {}

        # Backend path (unchanged): one-hot at backend start.
        if self.networkx_env is not None:
            sp, backend_info = self.networkx_env.reset(seed=seed)
            sp = int(sp)
            try:
                self.networkx_env.current_state = sp
            except Exception:
                pass
            obs, decode_info = self.decode_state(sp)
            self.initial_state_distrib = np.zeros(self.nrow * self.ncol, dtype=float)
            self.initial_state_distrib[int(obs)] = 1.0
            info: Dict[str, Any] = {}
            if isinstance(backend_info, dict):
                info.update(backend_info)
            if isinstance(decode_info, dict):
                info.update(decode_info)
            if self.render_mode == "human":
                self.render()
            return obs, info

        # Native path (keep original flow)
        super().reset(seed=seed)  # seeds self.np_random

        # pick start/goal using: options -> candidates -> default
        self.start_xy = self.parse_state_option("start_loc", options)
        self.goal_xy = self.parse_state_option("goal_loc", options)

        # init internals
        self.agent_xy = self.start_xy
        self.reward = self.get_reward(*self.agent_xy)
        self.done = self.on_goal()
        self.agent_action = None
        self.n_iter = 0

        # integrity
        self.integrity_checks()

        # build initial_state_distrib
        nS = self.nrow * self.ncol
        self.initial_state_distrib = np.zeros(nS, dtype=float)
        if "start_loc" in options:
            s = self.to_s(*self.start_xy)
            self.initial_state_distrib[int(s)] = 1.0
        elif self.start_candidates_raw:
            # uniform over valid candidate start states
            valids_idx: list[int] = []
            for v in self.start_candidates_raw:
                if isinstance(v, int):
                    r, c = self.to_xy(int(v))
                elif isinstance(v, tuple) and len(v) == 2:
                    r, c = int(v[0]), int(v[1])
                else:
                    continue
                if self.is_in_bounds(r, c) and self.is_free(r, c):
                    valids_idx.append(self.to_s(r, c))
            if valids_idx:
                for s in valids_idx:
                    self.initial_state_distrib[int(s)] = 1.0
                self.initial_state_distrib /= float(self.initial_state_distrib.sum())
            else:
                s = self.to_s(*self.start_xy)
                self.initial_state_distrib[int(s)] = 1.0
        else:
            s = self.to_s(*self.start_xy)
            self.initial_state_distrib[int(s)] = 1.0

        # optional render
        self.render()
        return self.get_obs(), self.get_info()

    def step(self, action: int):
        """
        If networkx_env provided, use backend reward.
        Else:
          - use_original_rewards=True: call base step (original rewards).
          - use_original_rewards=False: shaped rewards
              * every step: -1
              * invalid move: extra -1 (total -2)
              * reaching goal: 0 and done
        """
        action = int(action)

        # External backend (unchanged)
        if self.networkx_env is not None:
            s = int(self.encode_state())
            try:
                self.networkx_env.current_state = s
            except Exception:
                pass
            sp, r, terminated, truncated, info = self.networkx_env.step(action)
            obs, decode_info = self.decode_state(int(sp))
            info = info.copy() if isinstance(info, dict) else {}
            info.update(decode_info)
            if self.render_mode == "human":
                self.render()
            return int(obs), float(r), bool(terminated), bool(truncated), info

        # Original rewards
        if self.use_original_rewards:
            return super().step(action)

        # Shaped rewards, original dynamics preserved
        self.agent_action = action
        row, col = self.agent_xy
        dx, dy = self.MOVES[action]
        tr, tc = row + dx, col + dy
        base_step_cost = -1.0
        truncated = False

        # invalid move -> stay, -2
        if (not self.is_in_bounds(tr, tc)) or (not self.is_free(tr, tc)):
            self.reward = base_step_cost - 1.0
            self.done = False
        else:
            # valid move
            self.agent_xy = (tr, tc)
            if (tr, tc) == getattr(self, "goal_xy", (-1, -1)):
                self.reward = 0.0
                self.done = True
            else:
                self.reward = base_step_cost
                self.done = False

        self.n_iter += 1
        if self.render_mode == "human":
            self.render()
        return self.get_obs(), float(self.reward), bool(self.done), bool(truncated), self.get_info()

    # -------------------------
    # Encode / Decode
    # -------------------------
    def encode_state(self) -> int:
        """Return integer state index (row * ncol + col)."""
        return int(self.to_s(*self.agent_xy))

    def decode_state(self, state: int) -> Tuple[int, Dict[str, Any]]:
        """
        Force the environment to the given integer state.
        Update lightweight local fields for consistent rendering/info.
        """
        s = int(state)
        if not self.is_valid_state(s):
            raise ValueError(f"Invalid state {s} for grid {self.nrow}x{self.ncol}.")
        r, c = self._rc_from_state(s)
        self.agent_xy = (r, c)
        self.agent_action = None
        self.reward = self.get_reward(*self.agent_xy)
        self.done = self.on_goal()

        info = {
            "row": r,
            "col": c,
            "is_free": bool(self.is_free(r, c)),
            "is_goal": bool((r, c) == getattr(self, "goal_xy", (-1, -1))),
            "action_names": ACTION_NAMES,
        }
        if self.render_mode == "human":
            self.render()
        return int(s), info

    # -------------------------
    # Start states
    # -------------------------
    def get_start_states(self) -> List[int]:
        """
        Return a single start state index (current start_xy) if available,
        else fallback to [0].
        """
        if hasattr(self, "start_xy"):
            return [self.to_s(*self.start_xy)]
        return [0]

    # -------------------------
    # MDP export
    # -------------------------
    def get_mdp_network(self) -> MDPNetwork:
        """
        Build an MDPNetwork reflecting SimpleGridEnv step semantics:
          - Move into wall/out-of-bounds: stay (self-loop), reward -1, not terminal.
          - Move into goal: reward +1, terminal.
          - Move into free cell: reward 0, not terminal.
        Terminal state(s): goal state only.
        """
        nS = self.nrow * self.ncol
        nA = 4

        def to_s(row: int, col: int) -> int:
            return int(row) * self.ncol + int(col)

        # Determine start/goal if present
        start_states = self.get_start_states()
        terminal_states: List[int] = []
        if hasattr(self, "goal_xy"):
            terminal_states = [to_s(*self.goal_xy)]

        # Build transition map
        transitions: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
        for s in range(nS):
            r, c = self._rc_from_state(s)

            # If terminal (goal), add absorbing self-loops with r=0
            if int(s) in terminal_states:
                for a in range(nA):
                    s_key = str(s)
                    a_key = str(a)
                    transitions.setdefault(s_key, {})
                    transitions[s_key].setdefault(a_key, {})
                    transitions[s_key][a_key][str(s)] = {"p": 1.0, "r": 0.0}
                continue

            for a in range(nA):
                s_key = str(s)
                a_key = str(a)
                transitions.setdefault(s_key, {})
                a_bucket = transitions[s_key].setdefault(a_key, {})

                # Compute intended move (dx,dy in SimpleGridEnv.MOVES)
                dx, dy = self.MOVES[a]
                tr, tc = r + dx, c + dy

                # Out-of-bounds or wall: self-loop with -1
                if (not self.is_in_bounds(tr, tc)) or (not self.is_free(tr, tc)):
                    sp = s
                    p, rr = 1.0, -1.0
                    a_bucket[str(sp)] = {"p": float(p), "r": float(rr)}
                    continue

                # Free cell
                sp = to_s(tr, tc)
                is_goal = hasattr(self, "goal_xy") and (tr, tc) == self.goal_xy
                rr = 1.0 if is_goal else 0.0
                a_bucket[str(sp)] = {"p": 1.0, "r": float(rr)}

        # Tags
        free_states: List[int] = []
        wall_states: List[int] = []
        goal_state_list: List[int] = terminal_states.copy()
        start_state_list: List[int] = start_states.copy()

        for s in range(nS):
            r, c = self._rc_from_state(s)
            if hasattr(self, "goal_xy") and (r, c) == self.goal_xy:
                continue
            if self.is_free(r, c):
                free_states.append(s)
            else:
                wall_states.append(s)

        tags = {
            "start": sorted(list(set(start_state_list))),
            "goal": sorted(list(set(goal_state_list))),
            "free": sorted(free_states),
            "wall": sorted(wall_states),
        }

        config = {
            "num_actions": int(nA),
            "states": list(range(nS)),
            "start_states": start_states if start_states else [0],
            "terminal_states": sorted(list(set(terminal_states))),
            "default_reward": 0.0,
            "transitions": transitions,
            "tags": tags,
        }
        return MDPNetwork(config_data=config)

    # -------------------------
    # Extras (debug utils)
    # -------------------------
    def get_state_info(self) -> Dict[str, Any]:
        """Readable snapshot of current state."""
        s = int(self.encode_state())
        r, c = self._rc_from_state(s)
        return {
            "encoded_state": s,
            "row": r,
            "col": c,
            "is_free": bool(self.is_free(r, c)),
            "is_goal": bool((r, c) == getattr(self, "goal_xy", (-1, -1))),
            "action_names": ACTION_NAMES,
        }

    def is_valid_state(self, state: int) -> bool:
        return 0 <= int(state) < self.nrow * self.ncol

    # -------------------------
    # Utilities
    # -------------------------
    def _to_state(self, row: int, col: int) -> int:
        return int(row) * self.ncol + int(col)

    def _rc_from_state(self, s: int) -> Tuple[int, int]:
        return int(s) // self.ncol, int(s) % self.ncol

    # -------------------------
    # Safer close (avoid sys.exit)
    # -------------------------
    def close(self, *args, **kwargs):
        """Close figure safely without sys.exit."""
        try:
            if getattr(self, "fig", None) is not None:
                plt.close(self.fig)
        except Exception:
            pass
        # Do not call sys.exit()


# ---------------------------------------------------------------------
# Plotting utilities (Scheme B: use env.render('rgb_array') as background)
# ---------------------------------------------------------------------

def _get_bg_image_via_render(env, target_cell_px: int, dpi: int) -> Tuple[np.ndarray, float, float]:
    """
    Render env to RGB array and upscale if needed. Return (bg_img, cell_w, cell_h).
    """
    prev_mode = getattr(env, "render_mode", None)
    env.render_mode = "rgb_array"
    try:
        # Ensure a frame exists (SimpleGridEnv.reset() triggers a render)
        env.reset()
    except Exception:
        # As a fallback, try calling render_frame directly
        try:
            env.render_frame()
        except Exception as e:
            raise RuntimeError("Failed to reset/render background.") from e

    bg_img = env.render()
    if bg_img is None:
        raise RuntimeError("env.render('rgb_array') returned None.")

    if bg_img.shape[-1] == 4:
        # Convert RGBA -> RGB for safety
        bg_img = bg_img[..., :3]

    nrow, ncol = int(env.nrow), int(env.ncol)
    H, W = bg_img.shape[:2]
    cell_w, cell_h = W / ncol, H / nrow

    # Auto-upscale to improve readability
    upscale = int(np.ceil(target_cell_px / min(cell_w, cell_h)))
    upscale = max(1, min(upscale, 4))
    if upscale > 1:
        if _HAS_PIL:
            bg_img = np.array(
                Image.fromarray(bg_img).resize(
                    (int(W * upscale), int(H * upscale)),
                    resample=Image.BICUBIC,
                )
            )
        else:
            bg_img = np.kron(bg_img, np.ones((upscale, upscale, 1), dtype=bg_img.dtype))
        H, W = bg_img.shape[:2]
        cell_w *= upscale
        cell_h *= upscale

    env.render_mode = prev_mode
    return bg_img, float(cell_w), float(cell_h)


def _state_center_xy(s: int, ncol: int, cell_w: float, cell_h: float) -> Tuple[float, float]:
    r, c = divmod(int(s), ncol)
    return (c + 0.5) * cell_w, (r + 0.5) * cell_h


def _px_to_pt(px: float, dpi: int) -> float:
    return float(px) * 72.0 / float(dpi)


def _prob_to_color_fn(cmap_name: str, gamma: float):
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)
    def f(p: float):
        return cmap(norm(np.clip(float(p), 0.0, 1.0)))
    return f, cmap, norm


def _fetch_probs(mdp: MDPNetwork, s: int, a: int) -> Dict[int, float]:
    """
    Try mdp.get_transition_probabilities(s, a); fallback to transitions dict.
    """
    # Preferred API
    if hasattr(mdp, "get_transition_probabilities"):
        try:
            probs = mdp.get_transition_probabilities(int(s), int(a))
            # Expected to be {sp: p}
            return {int(k): float(v) for k, v in probs.items()}
        except Exception:
            pass

    # Fallback to config/transitions (string keys)
    trans = None
    if hasattr(mdp, "transitions"):
        trans = getattr(mdp, "transitions")
    elif hasattr(mdp, "config"):
        try:
            trans = mdp.config.get("transitions", None)
        except Exception:
            trans = None

    if not isinstance(trans, dict):
        return {}

    s_str, a_str = str(int(s)), str(int(a))
    a_bucket = trans.get(s_str, {}).get(a_str, {})
    out: Dict[int, float] = {}
    for sp_str, d in a_bucket.items():
        try:
            out[int(sp_str)] = float(d.get("p", 0.0))
        except Exception:
            continue
    return out


def plot_simplegrid_transition_overlays(
    env: Union[SimpleGridEnv, CustomisedSimpleGridEnv],
    mdp: MDPNetwork,
    output_dir: str,
    filename_prefix: str = "simplegrid_transitions",
    min_prob: float = 0.05,
    alpha: float = 0.90,
    annotate: bool = True,
    show_self_loops: bool = False,
    dpi: int = 200,
    target_cell_px: int = 240,
    arrow_scale: float = 0.04,
    font_scale: float = 0.16,
    cmap_name: str = "viridis",
    gamma: float = 1.0,
):
    """
    Draw per-action overlays of MDP transition probabilities on a SimpleGrid board.
    """
    assert hasattr(env, "nrow") and hasattr(env, "ncol"), "Env must have nrow/ncol."
    nrow, ncol = int(env.nrow), int(env.ncol)
    nS = nrow * ncol
    os.makedirs(output_dir, exist_ok=True)

    # Background via render (Scheme B)
    bg_img, cell_w, cell_h = _get_bg_image_via_render(env, target_cell_px, dpi)
    H, W = bg_img.shape[:2]
    cell_min = min(cell_w, cell_h)

    # Sizes
    ARROW_LW_PT = _px_to_pt(max(1.0, arrow_scale * cell_min), dpi)
    mutation_scale = _px_to_pt(0.45 * cell_min, dpi)
    shrink_pt = _px_to_pt(0.18 * cell_min, dpi)
    font_pt = max(6.0, min(12.0, _px_to_pt(font_scale * cell_min, dpi)))
    title_pt = max(9.0, min(14.0, _px_to_pt(0.18 * cell_min, dpi)))

    # Text styles
    text_bbox = dict(facecolor="white", alpha=0.50, edgecolor="none", boxstyle="round,pad=0.15")
    text_effects = [pe.withStroke(linewidth=_px_to_pt(1.0, dpi), foreground="black", alpha=0.35)]

    # Color mapping
    prob_to_color, cmap, norm = _prob_to_color_fn(cmap_name, gamma)

    def draw_self_loop(ax, x, y, p):
        color = prob_to_color(p)
        radius = 0.28 * cell_min
        arc = Arc(
            (x + 0.4 * radius, y - 0.4 * radius),
            width=radius,
            height=radius,
            angle=0,
            theta1=30,
            theta2=320,
            linewidth=ARROW_LW_PT,
            color=color,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(arc)
        arr = FancyArrowPatch(
            (x + 0.78 * radius, y - 0.55 * radius),
            (x + 0.63 * radius, y - 0.45 * radius),
            arrowstyle="->",
            mutation_scale=mutation_scale,
            linewidth=ARROW_LW_PT,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            zorder=4,
            shrinkA=0.0,
            shrinkB=0.0,
        )
        ax.add_patch(arr)

    # One figure per action
    for a in range(4):
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax = plt.gca()
        ax.imshow(bg_img, origin="upper", extent=[0, W, H, 0], zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        title = f"Action: {ACTION_NAMES[a] if 0 <= a < len(ACTION_NAMES) else str(a)}"
        ax.set_title(title, fontsize=title_pt)

        # Colorbar legend
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Transition probability", fontsize=max(8, int(title_pt * 0.7)))
        cbar.ax.tick_params(labelsize=max(6, int(font_pt * 0.9)))

        for s in range(nS):
            probs = _fetch_probs(mdp, s, a)
            if not probs:
                continue
            x0, y0 = _state_center_xy(s, ncol, cell_w, cell_h)
            for sp, p in probs.items():
                if p < float(min_prob):
                    continue
                x1, y1 = _state_center_xy(sp, ncol, cell_w, cell_h)
                color = prob_to_color(p)

                if int(sp) == int(s):
                    if show_self_loops:
                        draw_self_loop(ax, x0, y0, p)
                        if annotate:
                            ax.text(
                                x0,
                                y0 - 0.33 * cell_h,
                                f"{p:.2f}",
                                ha="center",
                                va="center",
                                fontsize=font_pt,
                                bbox=text_bbox,
                                alpha=alpha,
                                zorder=5,
                                path_effects=text_effects,
                            )
                    continue

                arrow = FancyArrowPatch(
                    (x0, y0),
                    (x1, y1),
                    arrowstyle="->",
                    mutation_scale=mutation_scale,
                    linewidth=ARROW_LW_PT,
                    facecolor=color,
                    edgecolor=color,
                    alpha=alpha,
                    zorder=3,
                    shrinkA=shrink_pt,
                    shrinkB=shrink_pt,
                )
                ax.add_patch(arrow)

                if annotate:
                    mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
                    ax.text(
                        mx,
                        my,
                        f"{p:.2f}",
                        ha="center",
                        va="center",
                        fontsize=font_pt,
                        bbox=text_bbox,
                        alpha=alpha,
                        zorder=4,
                        path_effects=text_effects,
                    )

        out_name = f"{filename_prefix}_a{a}_{(ACTION_NAMES[a] if 0 <= a < len(ACTION_NAMES) else str(a)).lower()}.png"
        plt.savefig(os.path.join(output_dir, out_name), bbox_inches="tight", pad_inches=0.05, dpi=dpi)
        plt.close(fig)

    print(f"[OK] Saved overlays to: {os.path.abspath(output_dir)}")


def plot_simplegrid_scalar_overlay(
    env: Union[SimpleGridEnv, CustomisedSimpleGridEnv],
    value_map: Any,  # supports .get_value(s) or dict-like
    output_dir: str,
    filename_prefix: str = "simplegrid_scalar",
    alpha: float = 0.65,
    annotate: bool = True,
    dpi: int = 200,
    target_cell_px: int = 240,
    font_scale: float = 0.18,
    cmap_name: str = "magma",
    gamma: float = 1.0,
    min_abs_label: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "State Value",
    cbar_label: str = "Value",
    value_format: str | None = None,
) -> str:
    """
    Overlay a per-state scalar (e.g., V(s)) as a semi-transparent heat layer.
    """
    assert hasattr(env, "nrow") and hasattr(env, "ncol"), "Env must have nrow/ncol."
    nrow, ncol = int(env.nrow), int(env.ncol)
    nS = nrow * ncol
    os.makedirs(output_dir, exist_ok=True)

    bg_img, cell_w, cell_h = _get_bg_image_via_render(env, target_cell_px, dpi)
    H, W = bg_img.shape[:2]
    cell_min = min(cell_w, cell_h)

    def px_to_pt(px: float) -> float:
        return _px_to_pt(px, dpi)

    def fmt_val(v: float) -> str:
        if value_format is not None:
            return format(v, value_format)
        return f"{v:.2e}" if (v != 0.0 and abs(v) < 0.01) else f"{v:.2f}"

    font_pt = max(6.0, min(14.0, px_to_pt(font_scale * cell_min)))
    title_pt = max(10.0, min(16.0, px_to_pt(0.20 * cell_min)))

    text_bbox = dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.15")
    text_effects = [pe.withStroke(linewidth=px_to_pt(1.0), foreground="black", alpha=0.35)]

    def val_get(s: int) -> float:
        if hasattr(value_map, "get_value"):
            return float(value_map.get_value(int(s)))
        try:
            return float(value_map.get(int(s), 0.0))
        except Exception:
            return 0.0

    val_grid = np.zeros((nrow, ncol), dtype=float)
    for s in range(nS):
        r, c = divmod(s, ncol)
        val_grid[r, c] = val_get(s)

    data_min = float(np.nanmin(val_grid)) if np.isfinite(val_grid).any() else 0.0
    data_max = float(np.nanmax(val_grid)) if np.isfinite(val_grid).any() else 1.0
    vmin = data_min if vmin is None else float(vmin)
    vmax = data_max if vmax is None else float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-9

    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = plt.gca()

    ax.imshow(bg_img, origin="upper", extent=[0, W, H, 0], zorder=0)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=title_pt)

    ax.imshow(
        val_grid,
        origin="upper",
        cmap=cmap,
        norm=norm,
        extent=[0, W, H, 0],
        alpha=alpha,
        zorder=1,
        interpolation="nearest",
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=max(8, int(title_pt * 0.7)))
    cbar.ax.tick_params(labelsize=max(6, int(font_pt * 0.9)))

    if annotate:
        for s in range(nS):
            r, c = divmod(s, ncol)
            v = val_grid[r, c]
            if abs(v) < float(min_abs_label):
                continue
            x, y = _state_center_xy(s, ncol, cell_w, cell_h)
            ax.text(
                x,
                y,
                fmt_val(v),
                ha="center",
                va="center",
                fontsize=font_pt,
                bbox=text_bbox,
                alpha=0.95,
                zorder=2,
                path_effects=text_effects,
            )

    out_path = os.path.join(output_dir, f"{filename_prefix}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved scalar overlay to: {os.path.abspath(out_path)}")
    return os.path.abspath(out_path)


def plot_simplegrid_scalar_diff_overlay(
    env: Union[SimpleGridEnv, CustomisedSimpleGridEnv],
    values_a: Any,  # supports .get_value(s) or dict-like
    values_b: Any,  # supports .get_value(s) or dict-like
    output_dir: str,
    filename_prefix: str = "simplegrid_scalar_diff",
    alpha: float = 0.65,
    annotate: bool = True,
    dpi: int = 200,
    target_cell_px: int = 240,
    font_scale: float = 0.18,
    cmap_name: str = "coolwarm",
    min_abs_label: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "Δ State Value (A − B)",
    cbar_label: str = "Δ value (A − B)",
    value_format: str | None = "+.2f",
) -> str:
    """
    Overlay the difference between two per-state scalars (A − B) with a diverging colormap.
    """
    assert hasattr(env, "nrow") and hasattr(env, "ncol"), "Env must have nrow/ncol."
    nrow, ncol = int(env.nrow), int(env.ncol)
    nS = nrow * ncol
    os.makedirs(output_dir, exist_ok=True)

    bg_img, cell_w, cell_h = _get_bg_image_via_render(env, target_cell_px, dpi)
    H, W = bg_img.shape[:2]
    cell_min = min(cell_w, cell_h)

    def px_to_pt(px: float) -> float:
        return _px_to_pt(px, dpi)

    def fmt_val(v: float) -> str:
        if value_format is not None:
            return format(v, value_format)
        return f"{v:+.2e}" if (v != 0.0 and abs(v) < 0.01) else f"{v:+.2f}"

    font_pt = max(6.0, min(14.0, px_to_pt(font_scale * cell_min)))
    title_pt = max(10.0, min(16.0, px_to_pt(0.20 * cell_min)))

    text_bbox = dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.15")
    text_effects = [pe.withStroke(linewidth=px_to_pt(1.0), foreground="black", alpha=0.35)]

    def val_get(tbl, s: int) -> float:
        if hasattr(tbl, "get_value"):
            return float(tbl.get_value(int(s)))
        try:
            return float(tbl.get(int(s), 0.0))
        except Exception:
            return 0.0

    grid_a = np.zeros((nrow, ncol), dtype=float)
    grid_b = np.zeros((nrow, ncol), dtype=float)
    for s in range(nS):
        r, c = divmod(s, ncol)
        grid_a[r, c] = val_get(values_a, s)
        grid_b[r, c] = val_get(values_b, s)
    diff_grid = grid_a - grid_b

    finite_mask = np.isfinite(diff_grid)
    max_abs = float(np.nanmax(np.abs(diff_grid[finite_mask]))) if finite_mask.any() else 1.0
    if vmin is None or vmax is None:
        vmin, vmax = -max_abs, max_abs
    if not (vmin < 0.0 < vmax):
        if vmax <= 0.0 and vmin < 0.0:
            vmax = abs(vmin)
        elif vmin >= 0.0 and vmax > 0.0:
            vmin = -vmax
        else:
            vmin, vmax = -1e-9, 1e-9

    cmap = cm.get_cmap(cmap_name)
    try:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    except Exception:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = plt.gca()

    ax.imshow(bg_img, origin="upper", extent=[0, W, H, 0], zorder=0)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=title_pt)

    ax.imshow(
        diff_grid,
        origin="upper",
        cmap=cmap,
        norm=norm,
        extent=[0, W, H, 0],
        alpha=alpha,
        zorder=1,
        interpolation="nearest",
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=max(8, int(title_pt * 0.7)))
    cbar.ax.tick_params(labelsize=max(6, int(font_pt * 0.9)))

    if annotate:
        for s in range(nS):
            r, c = divmod(s, ncol)
            v = diff_grid[r, c]
            if not np.isfinite(v) or abs(v) < float(min_abs_label):
                continue
            x, y = _state_center_xy(s, ncol, cell_w, cell_h)
            ax.text(
                x,
                y,
                fmt_val(v),
                ha="center",
                va="center",
                fontsize=font_pt,
                bbox=text_bbox,
                alpha=0.95,
                zorder=2,
                path_effects=text_effects,
            )

    out_path = os.path.join(output_dir, f"{filename_prefix}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved scalar diff overlay to: {os.path.abspath(out_path)}")
    return os.path.abspath(out_path)
