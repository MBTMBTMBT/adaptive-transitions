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

from PIL import Image
_HAS_PIL = True

from gym_simplegrid.envs import SimpleGridEnv
from apis.customisable import CustomisableEnvAbs
from mdp_network.mdp_network import MDPNetwork


MAPS = {
    "4x4": ["s000", "0101", "0001", "100g"],
    "8x8": [
        "s0010000",
        "00000000",
        "00010000",
        "11000100",
        "00010110",
        "01100010",
        "01001010",
        "0001g000",
    ],
}

# Match SimpleGridEnv.MOVES order (0:UP,1:DOWN,2:LEFT,3:RIGHT)
ACTION_NAMES: Tuple[str, str, str, str] = ("UP", "DOWN", "LEFT", "RIGHT")


class CustomisedSimpleGridEnv(SimpleGridEnv, CustomisableEnvAbs):
    """
    SimpleGridEnv with:
      - Optional delegation to an external NetworkX/MDP backend.
      - Encode/decode integer state: s = row * ncol + col.
      - Start/goal can be marked in map via s/S and g/G (case-insensitive).
      - MDP export to MDPNetwork (not shown here).
    """

    def __init__(
        self,
        obstacle_map: str | list[str],
        render_mode: str | None = None,
        networkx_env: Any = None,
        use_original_rewards: bool = False,
    ):
        # Resolve named map; allow custom list[str] with s/S/g/G markers.
        raw_map = MAPS.get(obstacle_map, obstacle_map) if isinstance(obstacle_map, str) else obstacle_map

        # Extract markers (uniform later) and build a clean 0/1 map for the base env.
        self._start_markers_xy: list[tuple[int, int]] = []
        self._goal_markers_xy: list[tuple[int, int]] = []
        if isinstance(raw_map, list):
            cleaned: list[str] = []
            for r, line in enumerate(raw_map):
                row_chars = []
                for c, ch in enumerate(line):
                    if ch in ("s", "S"):
                        self._start_markers_xy.append((r, c))
                        row_chars.append("0")
                    elif ch in ("g", "G"):
                        self._goal_markers_xy.append((r, c))
                        row_chars.append("0")
                    elif ch == "1":
                        row_chars.append("1")
                    else:
                        # treat any non-'1' as free ('0')
                        row_chars.append("0")
                cleaned.append("".join(row_chars))
            resolved_map = cleaned
        else:
            # String name not in our MAPS: let base class handle it.
            resolved_map = raw_map

        super().__init__(obstacle_map=resolved_map, render_mode=render_mode)

        self.networkx_env = networkx_env
        self.use_original_rewards = bool(use_original_rewards)
        self.initial_state_distrib: Optional[np.ndarray] = None

    def parse_obstacle_map(self, obstacle_map) -> np.ndarray:
        """
        Convert map to 0/1 numpy array without using dtype='c'.
        - If string: look up in our MAPS.
        - If list[str]: convert directly.
        - Treat '1' as wall; anything else as free ('0'), so 's/S' and 'g/G' are fine.
        """

        def to_grid(map_list: list[str]) -> np.ndarray:
            rows = []
            for line in map_list:
                rows.append([1 if ch == '1' else 0 for ch in line])
            return np.asarray(rows, dtype=np.int8)

        if isinstance(obstacle_map, str):
            if obstacle_map not in MAPS:
                raise ValueError(
                    f"Unknown map name '{obstacle_map}'. Available: {', '.join(MAPS.keys())} "
                    "or pass a custom list[str] map."
                )
            return to_grid(MAPS[obstacle_map])

        if isinstance(obstacle_map, list):
            # Accept custom maps possibly containing 's/S' and 'g/G'
            return to_grid(obstacle_map)

        raise ValueError(
            f"You must provide a map name (str) or a custom map (list[str]). "
            f"Available names: {', '.join(MAPS.keys())}."
        )

    def parse_state_option(self, state_name: str, options: dict) -> tuple:
        """
        Priority: explicit options[state_name] -> map markers -> default sampler.
        Accept int (state index) or (row,col) tuples in options.
        """
        # 1) explicit override
        if isinstance(options, dict) and state_name in options:
            v = options[state_name]
            if isinstance(v, int):
                return self.to_xy(int(v))
            if isinstance(v, tuple) and len(v) == 2:
                return (int(v[0]), int(v[1]))
            raise TypeError(f"Allowed types for `{state_name}` are int or tuple.")

        # 2) markers (uniform)
        if state_name == "start_loc" and self._start_markers_xy:
            i = int(self.np_random.integers(0, len(self._start_markers_xy)))
            return self._start_markers_xy[i]
        if state_name == "goal_loc" and self._goal_markers_xy:
            i = int(self.np_random.integers(0, len(self._goal_markers_xy)))
            return self._goal_markers_xy[i]

        # 3) fallback
        state = self.sample_valid_state_xy()
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
        Start/goal selection:
          - options['start_loc'/'goal_loc'] if provided;
          - else uniform over s/S and g/G markers (if any);
          - else sample valid states (default method).
        initial_state_distrib:
          - one-hot at options start;
          - else uniform over all start markers;
          - else one-hot at sampled start.
        """
        options = options or {}

        # External backend path: unchanged
        if self.networkx_env is not None:
            # External backend path
            sp, backend_info = self.networkx_env.reset(seed=seed)
            sp = int(sp)
            try:
                self.networkx_env.current_state = sp
            except Exception:
                pass

            obs, decode_info = self.decode_state(sp)

            # one-hot initial distribution at backend start
            self.initial_state_distrib = np.zeros(self.nrow * self.ncol, dtype=float)
            self.initial_state_distrib[int(obs)] = 1.0

            info: Dict[str, Any] = {}
            if isinstance(backend_info, dict):
                info.update(backend_info)
            if isinstance(decode_info, dict):
                info.update(decode_info)

            # Render only if we actually have start/goal (native drawing depends on them)
            if self.render_mode == "human" and hasattr(self, "start_xy") and hasattr(self, "goal_xy"):
                self.render()

            return obs, info

        # Seed RNG (via parent Env.reset); do not keep parent's start/goal.
        super().reset(seed=seed)

        # Choose start/goal using our policy
        self.start_xy = self.parse_state_option("start_loc", options)
        self.goal_xy = self.parse_state_option("goal_loc", options)

        # Init internals
        self.agent_xy = self.start_xy
        self.reward = self.get_reward(*self.agent_xy)
        self.done = self.on_goal()
        self.agent_action = None
        self.n_iter = 0

        # Sanity checks
        self.integrity_checks()

        # Build initial_state_distrib (over starts)
        nS = self.nrow * self.ncol
        self.initial_state_distrib = np.zeros(nS, dtype=float)
        if "start_loc" in options:
            s = self.to_s(*self.start_xy)
            self.initial_state_distrib[int(s)] = 1.0
        elif self._start_markers_xy:
            for (r, c) in self._start_markers_xy:
                self.initial_state_distrib[self.to_s(r, c)] = 1.0
            self.initial_state_distrib /= float(self.initial_state_distrib.sum())
        else:
            s = self.to_s(*self.start_xy)
            self.initial_state_distrib[int(s)] = 1.0

        # Optional render
        self.render()
        return self.get_obs(), self.get_info()

    def step(self, action: int):
        """
        Backend provided -> use backend reward.
        Else (custom shaping when use_original_rewards=False):
          - every valid move: -1.0
          - collision (wall/out-of-bounds): -1.1 (extra -0.1)
          - reaching goal: 0.0 and done
        """
        action = int(action)

        # External backend
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
            if self.render_mode == "human" and hasattr(self, "start_xy") and hasattr(self, "goal_xy"):
                self.render()
            return int(obs), float(r), bool(terminated), bool(truncated), info

        # Original rewards path
        if self.use_original_rewards:
            return super().step(action)

        # Shaped rewards
        self.agent_action = action
        row, col = self.agent_xy
        dx, dy = self.MOVES[action]
        tr, tc = row + dx, col + dy
        base_step_cost = -1.0
        collision_extra = -0.1
        truncated = False

        # Collision: stay put, total -1.1
        if (not self.is_in_bounds(tr, tc)) or (not self.is_free(tr, tc)):
            self.reward = base_step_cost + collision_extra  # -1.1
            self.done = False

        else:
            # Valid move
            self.agent_xy = (tr, tc)
            if (tr, tc) == getattr(self, "goal_xy", (-1, -1)):
                self.reward = 0.0
                self.done = True
            else:
                self.reward = base_step_cost  # -1.0
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
        In backend mode, do not compute reward/done locally.
        """
        s = int(state)
        if not self.is_valid_state(s):
            raise ValueError(f"Invalid state {s} for grid {self.nrow}x{self.ncol}.")
        r, c = self._rc_from_state(s)
        self.agent_xy = (r, c)
        self.agent_action = None

        # Safe reward/done: only use local semantics if goal_xy exists (native mode).
        if hasattr(self, "goal_xy"):
            self.reward = self.get_reward(*self.agent_xy)
            self.done = self.on_goal()
        else:
            # Backend mode: reward/done come from backend .step(); keep neutral here.
            self.reward = 0.0
            self.done = False

        info = {
            "row": r,
            "col": c,
            "is_free": bool(self.is_free(r, c)),
            "is_goal": bool(hasattr(self, "goal_xy") and (r, c) == self.goal_xy),
            "action_names": ACTION_NAMES,
        }

        # Only render if start/goal exist; render() needs them to draw start/goal markers.
        if self.render_mode == "human" and hasattr(self, "start_xy") and hasattr(self, "goal_xy"):
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
        Export MDP matching the shaped rewards:
          - collision (wall/OOB): self-loop, reward -1.1
          - valid non-goal move: reward -1.0
          - reaching goal: reward 0.0, terminal
        """
        nS = self.nrow * self.ncol
        nA = 4

        start_states = self.get_start_states()
        terminal_states: List[int] = []
        if hasattr(self, "goal_xy"):
            terminal_states = [self.to_s(*self.goal_xy)]

        transitions: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
        for s in range(nS):
            r, c = self._rc_from_state(s)

            # Absorbing goal with 0 reward
            if int(s) in terminal_states:
                for a in range(nA):
                    s_key = str(s);
                    a_key = str(a)
                    transitions.setdefault(s_key, {})
                    transitions[s_key].setdefault(a_key, {})
                    transitions[s_key][a_key][str(s)] = {"p": 1.0, "r": 0.0}
                continue

            for a in range(nA):
                s_key = str(s);
                a_key = str(a)
                transitions.setdefault(s_key, {})
                a_bucket = transitions[s_key].setdefault(a_key, {})

                dx, dy = self.MOVES[a]
                tr, tc = r + dx, c + dy

                # Collision -> self-loop with -1.1
                if (not self.is_in_bounds(tr, tc)) or (not self.is_free(tr, tc)):
                    a_bucket[str(s)] = {"p": 1.0, "r": -1.1}
                    continue

                sp = self.to_s(tr, tc)
                is_goal = hasattr(self, "goal_xy") and (tr, tc) == self.goal_xy
                rr = 0.0 if is_goal else -1.0
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
            (free_states if self.is_free(r, c) else wall_states).append(s)

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
    Render and crop to the Matplotlib Axes area so our overlay uses
    the exact pixel geometry of the grid (no rescaling).
    Returns: (cropped_bg_img, cell_w_px, cell_h_px).
    """
    prev_mode = getattr(env, "render_mode", None)
    env.render_mode = "rgb_array"

    # Ensure a frame and a valid (fig, ax)
    try:
        env.reset()
    except Exception:
        try:
            env.render_frame()
        except Exception as e:
            raise RuntimeError("Failed to reset/render background.") from e

    fig = getattr(env, "fig", None)
    ax = getattr(env, "ax", None)
    if fig is None or ax is None:
        raise RuntimeError("Env figure/axes not available after render().")

    # Draw and grab full canvas RGBA
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    full = np.asarray(fig.canvas.renderer.buffer_rgba())
    Hc, Wc = full.shape[:2]

    # Axes bbox in display pixels; convert to array indices (top-left origin)
    bbox = ax.get_window_extent(renderer=renderer)
    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    left   = max(0, int(np.floor(x0)))
    right  = min(Wc, int(np.ceil(x1)))
    top    = max(0, int(np.floor(Hc - y1)))  # invert Y
    bottom = min(Hc, int(np.ceil(Hc - y0)))

    # Crop to Axes area (this is exactly where the grid is drawn)
    bg_img = full[top:bottom, left:right, :]

    # Drop alpha if present
    if bg_img.shape[-1] == 4:
        bg_img = bg_img[..., :3]

    # Cell size in *cropped* pixels
    nrow, ncol = int(env.nrow), int(env.ncol)
    H, W = bg_img.shape[:2]
    cell_w, cell_h = W / ncol, H / nrow

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
    arrow_scale: float = 0.04,  # arrow LW ~ arrow_scale * cell_min (in px)
    font_scale: float = 0.16,   # text size ~ font_scale * cell_min (in px)
    cmap_name: str = "viridis",
    gamma: float = 1.0,
):
    """
    Per-action transition overlays on top of env.render() background.
    Font/arrow sizes scale purely with `font_scale`, `arrow_scale`, and `dpi`.
    """
    assert hasattr(env, "nrow") and hasattr(env, "ncol"), "Env must have nrow/ncol."
    nrow, ncol = int(env.nrow), int(env.ncol)
    nS = nrow * ncol
    os.makedirs(output_dir, exist_ok=True)

    # Background via render (exact axes crop)
    bg_img, cell_w, cell_h = _get_bg_image_via_render(env, target_cell_px, dpi)
    H, W = bg_img.shape[:2]
    cell_min = min(cell_w, cell_h)

    # Sizes (no hard clamps; tiny safety floors only)
    ARROW_LW_PT   = max(0.25, _px_to_pt(arrow_scale * cell_min, dpi))   # line width
    mutation_scale = _px_to_pt(0.45 * cell_min, dpi)                    # arrow head size
    shrink_pt     = _px_to_pt(0.18 * cell_min, dpi)                     # arrow shrink
    font_pt       = max(0.5, _px_to_pt(font_scale * cell_min, dpi))     # label font
    title_pt      = max(0.5, _px_to_pt(0.18 * cell_min, dpi))           # title font

    # Text styles
    text_bbox = dict(facecolor="white", alpha=0.50, edgecolor="none", boxstyle="round,pad=0.15")
    text_effects = [pe.withStroke(linewidth=_px_to_pt(1.0, dpi), foreground="black", alpha=0.35)]

    # Color mapping
    prob_to_color, cmap, norm = _prob_to_color_fn(cmap_name, gamma)

    def draw_self_loop(ax, x, y, p):
        """Small arc + arrow for s->s."""
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
        cbar.set_label("Transition probability", fontsize=max(1, int(title_pt * 0.7)))
        cbar.ax.tick_params(labelsize=max(1, int(font_pt * 0.9)))

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
    Heat overlay on top of env.render() background.
    Scales fonts purely by `font_scale` and `dpi` (no hard clamps).
    """
    assert hasattr(env, "nrow") and hasattr(env, "ncol"), "Env must have nrow/ncol."
    nrow, ncol = int(env.nrow), int(env.ncol)
    nS = nrow * ncol
    os.makedirs(output_dir, exist_ok=True)

    # Background is the exact rendered grid area
    bg_img, cell_w, cell_h = _get_bg_image_via_render(env, target_cell_px, dpi)
    H, W = bg_img.shape[:2]
    cell_min = min(cell_w, cell_h)

    def px_to_pt(px: float) -> float:
        return _px_to_pt(px, dpi)

    def fmt_val(v: float) -> str:
        if value_format is not None:
            return format(v, value_format)
        return f"{v:.2e}" if (v != 0.0 and abs(v) < 0.01) else f"{v:.2f}"

    # Font sizes: no upper clamps; tiny lower safety only
    font_pt  = max(0.5, px_to_pt(font_scale * cell_min))
    title_pt = max(0.5, px_to_pt(0.20 * cell_min))

    text_bbox = dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.15")
    text_effects = [pe.withStroke(linewidth=px_to_pt(1.0), foreground="black", alpha=0.35)]

    # Build value grid
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
    cbar.set_label(cbar_label, fontsize=max(1, int(title_pt * 0.7)))
    cbar.ax.tick_params(labelsize=max(1, int(font_pt * 0.9)))

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
    Diverging heat overlay for A−B. Fonts scale by `font_scale` and `dpi` only.
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

    # Font sizes: no hard clamps
    font_pt  = max(0.5, px_to_pt(font_scale * cell_min))
    title_pt = max(0.5, px_to_pt(0.20 * cell_min))

    text_bbox = dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.15")
    text_effects = [pe.withStroke(linewidth=px_to_pt(1.0), foreground="black", alpha=0.35)]

    # Build diff grid
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
    cbar.set_label(cbar_label, fontsize=max(1, int(title_pt * 0.7)))
    cbar.ax.tick_params(labelsize=max(1, int(font_pt * 0.9)))

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
