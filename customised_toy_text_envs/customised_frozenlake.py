import os
from typing import Tuple, Dict, Any, List, Union

from gymnasium import spaces
from gymnasium.core import ObsType
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib import patheffects as pe
import matplotlib.colors as mcolors

from mdp_network.mdp_network import MDPNetwork
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map, LEFT, DOWN, RIGHT, UP
from customisable_env_abs import CustomisableEnvAbs
from mdp_network.mdp_tables import ValueTable


MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
    "env0": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFGFF",
        "FFFFFFFF",
        "FFFFFFFF",
    ],
    "env1": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFHFFF",
        "FFFFHGFF",
        "FFFFFFFF",
        "FFFFFFFF",
    ],
    "env2": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFHFFF",
        "FFFFHGFF",
        "FFFFHFFF",
        "FFFFHFFF",
    ],
    "env3": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFHFFFFF",
        "FFFFFFFF",
        "FFFFHHFF",
        "FFFFHGFF",
        "FFHFFFFF",
        "FFFFHFFF",
    ],
    "env4": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFHFFHHF",
        "FFFFFFFF",
        "FFFFHHFF",
        "FFFFHGFF",
        "FFHFFFFF",
        "FFFFHFFF",
    ],
}

ACTION_NAMES = ["Left", "Down", "Right", "Up"]


class CustomisedFrozenLakeEnv(FrozenLakeEnv, CustomisableEnvAbs):
    """
    A customised FrozenLake environment that implements state encoding/decoding
    and can be driven by a NetworkX-backed MDP environment.

    State encoding is the native FrozenLake index:
        s = row * ncol + col
    """

    def __init__(
        self,
        render_mode: str | None = None,
        desc=None,
        map_name: str = "4x4",
        is_slippery: bool = True,
        networkx_env=None,
    ):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            new_row, new_col = inc(row, col, action)
            new_state = to_s(new_row, new_col)
            new_letter = desc[new_row, new_col]
            terminated = bytes(new_letter) in b"GH"
            reward = float(new_letter == b"G")
            return new_state, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None
        CustomisableEnvAbs.__init__(self, networkx_env=networkx_env)

    # -------------------------
    # Core overrides
    # -------------------------
    def step(self, action):
        """Optionally delegate transition to an external NetworkX MDP env."""
        if self.networkx_env is not None:
            # Current encoded state
            s = self.encode_state()

            # Sync external env
            self.networkx_env.current_state = s

            # One step in NetworkX env
            sp, r, terminated, truncated, info = self.networkx_env.step(action)

            # Decode back into this env
            obs, decode_info = self.decode_state(int(sp))
            info.update(decode_info)

            # For visual parity with base env
            self.lastaction = int(action)
            if self.render_mode == "human":
                self.render()

            # TimeLimit truncation is handled by wrappers in gym.make(); we keep False here
            return obs, float(r), bool(terminated), bool(truncated), info

        # Fallback to native FrozenLake dynamics
        return super().step(action)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None,
    ):
        """
        If a NetworkX-backed env is provided, delegate start-state sampling to it,
        then decode that state into this env's internal representation.
        Otherwise, fall back to the native FrozenLake reset.
        """
        if self.networkx_env is not None:
            # Reset external backend RNG and episode
            sp, backend_info = self.networkx_env.reset(seed=seed)
            sp = int(sp)

            # Sync current state for clarity (reset already set it, but keep explicit)
            self.networkx_env.current_state = sp

            # Decode into this env's internal state (and set internal fields)
            obs, decode_info = self.decode_state(sp)

            # Bookkeeping consistent with base env
            self.lastaction = None

            # Optional rendering
            if self.render_mode == "human":
                self.render()

            # Merge info dicts (backend info first, then decode info)
            info = {}
            if isinstance(backend_info, dict):
                info.update(backend_info)
            if isinstance(decode_info, dict):
                info.update(decode_info)
            return obs, info

        # Fallback: native dynamics
        return super().reset(seed=seed, options=options)

    # -------------------------
    # Encode / Decode
    # -------------------------
    def encode_state(self) -> int:
        """Return the current state as integer index (row * ncol + col)."""
        return int(self.s)

    def decode_state(self, state: int) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Force the environment to the given integer state.
        Resets lastaction to None for a clean render.
        """
        if not (0 <= int(state) < self.nrow * self.ncol):
            raise ValueError(f"Invalid state {state}, must be in [0, {self.nrow * self.ncol - 1}]")

        self.s = int(state)
        self.lastaction = None

        # Compose info payload
        r, c = self._rc_from_state(self.s)
        ch = bytes(self.desc[r, c]).decode("utf-8")
        info = {
            "prob": 1.0,
            "row": r,
            "col": c,
            "tile": ch,  # one of 'S','F','H','G'
            "action_names": ACTION_NAMES,
        }

        if self.render_mode == "human":
            self.render()

        return int(self.s), info

    # -------------------------
    # Start states
    # -------------------------
    def get_start_states(self) -> List[int]:
        """
        All indices where tile == 'S'. Matches the environment's initial_state_distrib support.
        """
        starts: List[int] = []
        for r in range(self.nrow):
            for c in range(self.ncol):
                if self.desc[r, c] == b"S":
                    starts.append(self._to_state(r, c))
        # Fallback: if map is malformed (no 'S'), default to 0
        return starts if starts else [0]

    # -------------------------
    # MDP export
    # -------------------------
    def get_mdp_network(self) -> MDPNetwork:
        """
        Build an MDPNetwork from FrozenLakeEnv.P.
        Rewards are taken directly from the underlying env:
          - stepping onto Goal: +1
          - otherwise: 0
        Terminal states are those that transition with done=True (G/H).
        """
        num_states: int = self.observation_space.n
        num_actions: int = self.action_space.n

        # States & starts
        states = list(range(num_states))
        start_states = [s for s, w in enumerate(self.initial_state_distrib) if w > 0.0]

        # Terminal set (any successor marked done)
        terminal_states_set = set()
        for s in range(num_states):
            for a in range(num_actions):
                for p, sp, r, done in self.P[s][a]:
                    if done:
                        terminal_states_set.add(int(sp))
        terminal_states = sorted(terminal_states_set)

        # Transitions: transitions["s"]["a"]["sp"] = {"p": prob, "r": reward}
        transitions: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
        for s in range(num_states):
            s_key = str(s)
            for a in range(num_actions):
                entries = self.P[s][a]
                if not entries:
                    continue
                a_key = str(a)
                # Aggregate (s,a,s') duplicates by summing probs and prob-weighted reward
                accum: Dict[int, Dict[str, float]] = {}
                for p, sp, r, _done in entries:
                    sp = int(sp)
                    acc = accum.setdefault(sp, {"p": 0.0, "r": 0.0})
                    new_p = acc["p"] + float(p)
                    acc["r"] = (acc["r"] * acc["p"] + float(r) * float(p)) / new_p if new_p > 0.0 else float(r)
                    acc["p"] = new_p

                if accum:
                    transitions.setdefault(s_key, {})
                    a_bucket = transitions[s_key].setdefault(a_key, {})
                    for sp, v in accum.items():
                        a_bucket[str(sp)] = {"p": float(v["p"]), "r": float(v["r"])}

        # Tags: label states by agent's tile type
        agent_on_start: List[int] = []
        agent_on_goal: List[int] = []
        agent_on_hole: List[int] = []
        agent_on_frozen: List[int] = []

        for s in range(num_states):
            r, c = self._rc_from_state(s)
            tile = self.desc[r, c]
            if tile == b"S":
                agent_on_start.append(s)
            elif tile == b"G":
                agent_on_goal.append(s)
            elif tile == b"H":
                agent_on_hole.append(s)
            elif tile == b"F":
                agent_on_frozen.append(s)

        tags = {
            "agent_on_start": sorted(agent_on_start),
            "agent_on_goal": sorted(agent_on_goal),
            "agent_on_hole": sorted(agent_on_hole),
            "agent_on_frozen": sorted(agent_on_frozen),
        }

        config = {
            "num_actions": int(num_actions),
            "states": states,
            "start_states": start_states if start_states else [0],
            "terminal_states": terminal_states,
            "default_reward": 0.0,  # non-terminal steps give 0 in FrozenLake
            "transitions": transitions,
            "tags": tags,
        }
        return MDPNetwork(config_data=config)

    # -------------------------
    # Extras (debug helpers)
    # -------------------------
    def get_state_info(self) -> Dict[str, Any]:
        """Return a readable summary of the current state."""
        s = int(self.s)
        r, c = self._rc_from_state(s)
        tile = bytes(self.desc[r, c]).decode("utf-8")
        return {
            "encoded_state": s,
            "row": r,
            "col": c,
            "tile": tile,  # 'S','F','H','G'
            "is_terminal": tile in ("G", "H"),
            "action_names": ACTION_NAMES,
        }

    def is_valid_state(self, state: int) -> bool:
        """Check if state index is in range (grid bounds)."""
        return 0 <= int(state) < self.nrow * self.ncol

    # -------------------------
    # Utilities
    # -------------------------
    def _to_state(self, row: int, col: int) -> int:
        return int(row) * self.ncol + int(col)

    def _rc_from_state(self, s: int) -> Tuple[int, int]:
        return int(s) // self.ncol, int(s) % self.ncol


# Compatible action-name resolver (list/tuple/dict)
def _get_action_name(a: int) -> str:
    try:
        if isinstance(ACTION_NAMES, dict):
            return str(ACTION_NAMES.get(a, a))
        elif isinstance(ACTION_NAMES, (list, tuple)) and 0 <= a < len(ACTION_NAMES):
            return str(ACTION_NAMES[a])
    except Exception:
        pass
    return str(a)


def plot_frozenlake_transition_overlays(
    env: Union[FrozenLakeEnv, CustomisedFrozenLakeEnv],
    mdp: MDPNetwork,
    output_dir: str,
    filename_prefix: str = "frozenlake_transitions",
    min_prob: float = 0.05,
    alpha: float = 0.90,              # constant transparency
    annotate: bool = True,            # draw probability labels
    show_self_loops: bool = False,    # draw s->s arcs
    dpi: int = 200,
    target_cell_px: int = 240,        # target cell size in pixels for readability
    arrow_scale: float = 0.04,        # arrow linewidth as fraction of cell size
    font_scale: float = 0.16,         # label font size as fraction of cell size
    cmap_name: str = "viridis",       # colormap for probability -> color
    gamma: float = 1.0                # gamma correction for probability mapping
):
    """
    Draw per-action overlays of MDP transition probabilities on a FrozenLake board.
    Visual encoding:
      - Arrow color comes from `cmap_name` based on probability p.
      - Arrow thickness is fixed and thin for clarity.
      - Board image is upscaled to make cells large enough for arrows and labels.
    """

    # -------- Basic checks --------
    assert hasattr(env, "nrow") and hasattr(env, "ncol"), "Env must have nrow/ncol."
    nrow, ncol = env.nrow, env.ncol
    nS = nrow * ncol

    if set(mdp.states) != set(range(nS)):
        print("[WARN] mdp.states does not match 0..nS-1")

    if getattr(mdp, "num_actions", 4) != 4:
        raise ValueError("This visualizer assumes exactly 4 actions (LEFT/DOWN/RIGHT/UP).")

    os.makedirs(output_dir, exist_ok=True)

    # -------- Grab board background --------
    prev_mode = getattr(env, "render_mode", None)
    env.render_mode = "rgb_array"
    try:
        env.reset()
        if hasattr(env, "initial_state_distrib"):
            env.s = int(np.argmax(env.initial_state_distrib))
    except Exception:
        pass

    bg_img = env.render()
    if bg_img is None:
        try:
            bg_img = env._render_gui("rgb_array")  # type: ignore
        except Exception as e:
            raise RuntimeError("Failed to render background.") from e
    env.render_mode = prev_mode

    H, W = bg_img.shape[:2]
    cell_w, cell_h = W / ncol, H / nrow

    # -------- Auto upscale board for readability --------
    upscale = int(np.ceil(target_cell_px / min(cell_w, cell_h)))
    upscale = max(1, min(upscale, 4))
    if upscale > 1:
        try:
            from PIL import Image
            bg_img = np.array(
                Image.fromarray(bg_img).resize((int(W * upscale), int(H * upscale)), resample=Image.BICUBIC)
            )
        except Exception:
            # fallback: nearest-neighbor upscale
            bg_img = np.kron(bg_img, np.ones((upscale, upscale, 1), dtype=bg_img.dtype))
        H, W = bg_img.shape[:2]
        cell_w *= upscale
        cell_h *= upscale

    # -------- Coordinate and style helpers --------
    def state_to_center_xy(s: int):
        """Convert state index to pixel coordinates of cell center."""
        r, c = divmod(s, ncol)
        return (c + 0.5) * cell_w, (r + 0.5) * cell_h

    def px_to_pt(px: float):
        """Convert pixels to points given figure DPI."""
        return float(px) * 72.0 / float(dpi)

    cell_min = min(cell_w, cell_h)

    # arrow and label style parameters
    ARROW_LW_PT = px_to_pt(max(1.0, arrow_scale * cell_min))
    mutation_scale = px_to_pt(0.45 * cell_min)
    shrink_pt = px_to_pt(0.18 * cell_min)
    font_pt = max(6.0, min(12.0, px_to_pt(font_scale * cell_min)))
    title_pt = max(9.0, min(14.0, px_to_pt(0.18 * cell_min)))

    # text style for labels
    text_bbox = dict(facecolor="white", alpha=0.50, edgecolor="none", boxstyle="round,pad=0.15")
    text_effects = [pe.withStroke(linewidth=px_to_pt(1.0), foreground="black", alpha=0.35)]

    # probability -> RGBA color using colormap
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)  # NEW: gamma-consistent normalization
    def prob_to_color(p: float):
        """Map probability to RGBA using cmap and gamma-corrected norm."""
        return cmap(norm(np.clip(p, 0, 1)))

    def draw_self_loop(ax, x, y, p):
        """Draw a small self-loop arc with color from probability."""
        color = prob_to_color(p)
        radius = 0.28 * cell_min
        arc = Arc((x + 0.4 * radius, y - 0.4 * radius),
                  width=radius, height=radius,
                  angle=0, theta1=30, theta2=320,
                  linewidth=ARROW_LW_PT, color=color, alpha=alpha, zorder=3)
        ax.add_patch(arc)
        arr = FancyArrowPatch(
            (x + 0.78 * radius, y - 0.55 * radius),
            (x + 0.63 * radius, y - 0.45 * radius),
            arrowstyle="->",
            mutation_scale=mutation_scale,
            linewidth=ARROW_LW_PT,
            facecolor=color, edgecolor=color,
            alpha=alpha, zorder=4, shrinkA=0.0, shrinkB=0.0
        )
        ax.add_patch(arr)

    # -------- Draw one figure per action --------
    for a in range(4):
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax = plt.gca()
        ax.imshow(bg_img, origin="upper", extent=[0, W, H, 0], zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Action: {_get_action_name(a)}", fontsize=title_pt)

        # NEW: add a colorbar legend for probability
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # required by older Matplotlib versions
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Transition probability", fontsize=max(8, int(title_pt * 0.7)))
        cbar.ax.tick_params(labelsize=max(6, int(font_pt * 0.9)))

        for s in range(nS):
            probs = mdp.get_transition_probabilities(s, a)
            if not probs:
                continue
            x0, y0 = state_to_center_xy(s)

            for sp, p in probs.items():
                try:
                    sp_int = int(sp)
                except Exception:
                    continue
                if p < min_prob:
                    continue

                x1, y1 = state_to_center_xy(sp_int)
                color = prob_to_color(p)

                if sp_int == s:
                    if show_self_loops:
                        draw_self_loop(ax, x0, y0, p)
                        if annotate:
                            ax.text(x0, y0 - 0.33 * cell_h, f"{p:.2f}",
                                    ha="center", va="center", fontsize=font_pt,
                                    bbox=text_bbox, alpha=alpha, zorder=5,
                                    path_effects=text_effects)
                    continue

                arrow = FancyArrowPatch(
                    (x0, y0), (x1, y1),
                    arrowstyle="->", mutation_scale=mutation_scale,
                    linewidth=ARROW_LW_PT,
                    facecolor=color, edgecolor=color,
                    alpha=alpha, zorder=3,
                    shrinkA=shrink_pt, shrinkB=shrink_pt
                )
                ax.add_patch(arrow)

                if annotate:
                    mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
                    ax.text(mx, my, f"{p:.2f}",
                            ha="center", va="center", fontsize=font_pt,
                            bbox=text_bbox, alpha=alpha, zorder=4,
                            path_effects=text_effects)

        out_name = f"{filename_prefix}_a{a}_{_get_action_name(a).lower()}.png"
        plt.savefig(os.path.join(output_dir, out_name), bbox_inches="tight", pad_inches=0.05, dpi=dpi)
        plt.close(fig)

    print(f"[OK] Saved overlays to: {os.path.abspath(output_dir)}")


def plot_frozenlake_scalar_overlay(
    env: Union[FrozenLakeEnv, CustomisedFrozenLakeEnv],
    value_map: "ValueTable",                # supports .get_value(s); dict-like also works
    output_dir: str,
    filename_prefix: str = "frozenlake_scalar",
    alpha: float = 0.65,                    # overlay transparency for the heat layer
    annotate: bool = True,                  # draw per-cell numeric labels
    dpi: int = 200,
    target_cell_px: int = 240,              # target cell size in pixels for readability
    font_scale: float = 0.18,               # label font size as fraction of cell size
    cmap_name: str = "magma",               # colormap for scalar values
    gamma: float = 1.0,                     # gamma for color normalization (PowerNorm)
    min_abs_label: float = 0.0,             # do not draw labels if |value| < threshold
    vmin: float | None = None,              # color scale min; defaults to data min
    vmax: float | None = None,              # color scale max; defaults to data max
    title: str = "State Value",             # title text
    cbar_label: str = "Value",              # colorbar label
    value_format: str | None = None,        # e.g. ".2f" or ".2e"; None -> auto (2f / 2e)
) -> str:
    """
    Overlay an arbitrary per-state scalar (e.g., V(s) or occupancy) as a semi-transparent heat layer
    on top of the FrozenLake board, with optional numeric labels and a colorbar legend.

    Assumptions:
      - `value_map` exposes `get_value(state)` -> float, or is dict-like {state: value}.
      - State indices match the environment encoding: s = row * ncol + col.

    Returns:
      The absolute path to the saved PNG file.
    """
    # -------- Basic checks --------
    assert hasattr(env, "nrow") and hasattr(env, "ncol"), "Env must have nrow/ncol."
    nrow, ncol = env.nrow, env.ncol
    nS = nrow * ncol
    os.makedirs(output_dir, exist_ok=True)

    # -------- Board background --------
    prev_mode = getattr(env, "render_mode", None)
    env.render_mode = "rgb_array"
    try:
        env.reset()
        if hasattr(env, "initial_state_distrib"):
            env.s = int(np.argmax(env.initial_state_distrib))
    except Exception:
        pass

    bg_img = env.render()
    if bg_img is None:
        try:
            bg_img = env._render_gui("rgb_array")  # type: ignore
        except Exception as e:
            raise RuntimeError("Failed to render background.") from e
    env.render_mode = prev_mode

    H, W = bg_img.shape[:2]
    cell_w, cell_h = W / ncol, H / nrow

    # -------- Auto upscale board for readability --------
    upscale = int(np.ceil(target_cell_px / min(cell_w, cell_h)))
    upscale = max(1, min(upscale, 4))
    if upscale > 1:
        try:
            from PIL import Image
            bg_img = np.array(
                Image.fromarray(bg_img).resize((int(W * upscale), int(H * upscale)), resample=Image.BICUBIC)
            )
        except Exception:
            bg_img = np.kron(bg_img, np.ones((upscale, upscale, 1), dtype=bg_img.dtype))
        H, W = bg_img.shape[:2]
        cell_w *= upscale
        cell_h *= upscale

    # -------- Helpers (coords, sizing) --------
    def state_to_center_xy(s: int) -> tuple[float, float]:
        r, c = divmod(int(s), ncol)
        return (c + 0.5) * cell_w, (r + 0.5) * cell_h

    def px_to_pt(px: float) -> float:
        return float(px) * 72.0 / float(dpi)

    def fmt_val(v: float) -> str:
        if value_format is not None:
            return format(v, value_format)
        # auto: tiny -> scientific; else fixed 2 decimals
        return (f"{v:.2e}" if abs(v) < 0.01 and v != 0.0 else f"{v:.2f}")

    cell_min = min(cell_w, cell_h)
    font_pt = max(6.0, min(14.0, px_to_pt(font_scale * cell_min)))
    title_pt = max(10.0, min(16.0, px_to_pt(0.20 * cell_min)))

    # Text style
    text_bbox = dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.15")
    text_effects = [pe.withStroke(linewidth=px_to_pt(1.0), foreground="black", alpha=0.35)]

    # -------- Build value grid --------
    def val_get(s: int) -> float:
        if hasattr(value_map, "get_value"):
            return float(value_map.get_value(int(s)))
        try:
            return float(value_map.get(int(s), 0.0))  # type: ignore[attr-defined]
        except Exception:
            return 0.0

    val_grid = np.zeros((nrow, ncol), dtype=float)
    for s in range(nS):
        r, c = divmod(s, ncol)
        val_grid[r, c] = val_get(s)

    # Defaults for vmin/vmax: data-driven (generic for value / occupancy)
    data_min = float(np.nanmin(val_grid)) if np.isfinite(val_grid).any() else 0.0
    data_max = float(np.nanmax(val_grid)) if np.isfinite(val_grid).any() else 1.0
    vmin = data_min if vmin is None else float(vmin)
    vmax = data_max if vmax is None else float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-9

    # Colormap and normalization (gamma-aware)
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    # -------- Plot --------
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
            val = val_grid[r, c]
            if abs(val) < min_abs_label:
                continue
            x, y = state_to_center_xy(s)
            ax.text(
                x, y, fmt_val(val),
                ha="center", va="center", fontsize=font_pt,
                bbox=text_bbox, alpha=0.95, zorder=2, path_effects=text_effects,
            )

    out_name = f"{filename_prefix}.png"
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    plt.close(fig)

    print(f"[OK] Saved scalar overlay to: {os.path.abspath(out_path)}")
    return os.path.abspath(out_path)


def plot_frozenlake_scalar_diff_overlay(
    env: Union[FrozenLakeEnv, CustomisedFrozenLakeEnv],
    values_a: "ValueTable",                 # supports .get_value(s); dict-like also works
    values_b: "ValueTable",                 # supports .get_value(s); dict-like also works
    output_dir: str,
    filename_prefix: str = "frozenlake_scalar_diff",
    alpha: float = 0.65,                    # overlay transparency for the heat layer
    annotate: bool = True,                  # draw per-cell numeric labels
    dpi: int = 200,
    target_cell_px: int = 240,              # target cell size in pixels for readability
    font_scale: float = 0.18,               # label font size as fraction of cell size
    cmap_name: str = "coolwarm",            # diverging colormap for signed differences
    min_abs_label: float = 0.0,             # do not draw labels if |Δ| < threshold
    vmin: float | None = None,              # color scale min; defaults to symmetric about 0
    vmax: float | None = None,              # color scale max; defaults to symmetric about 0
    title: str = "Δ State Value (A − B)",   # title text
    cbar_label: str = "Δ value (A − B)",    # colorbar label
    value_format: str | None = "+.2f",      # default signed fixed 2 decimals; None -> auto with sign
) -> str:
    """
    Overlay the difference between two per-state scalars (A − B) as a semi-transparent
    diverging heat layer on the FrozenLake board, with optional numeric labels and a colorbar.

    Assumptions:
      - `values_a` and `values_b` expose `get_value(state)` -> float, or are dict-like {state: value}.
      - State indices match the environment encoding: s = row * ncol + col.

    Returns:
      The absolute path to the saved PNG file.
    """
    # -------- Basic checks --------
    assert hasattr(env, "nrow") and hasattr(env, "ncol"), "Env must have nrow/ncol."
    nrow, ncol = env.nrow, env.ncol
    nS = nrow * ncol
    os.makedirs(output_dir, exist_ok=True)

    # -------- Board background --------
    prev_mode = getattr(env, "render_mode", None)
    env.render_mode = "rgb_array"
    try:
        env.reset()
        if hasattr(env, "initial_state_distrib"):
            env.s = int(np.argmax(env.initial_state_distrib))
    except Exception:
        pass

    bg_img = env.render()
    if bg_img is None:
        try:
            bg_img = env._render_gui("rgb_array")  # type: ignore
        except Exception as e:
            raise RuntimeError("Failed to render background.") from e
    env.render_mode = prev_mode

    H, W = bg_img.shape[:2]
    cell_w, cell_h = W / ncol, H / nrow

    # -------- Auto upscale board for readability --------
    upscale = int(np.ceil(target_cell_px / min(cell_w, cell_h)))
    upscale = max(1, min(upscale, 4))
    if upscale > 1:
        try:
            from PIL import Image
            bg_img = np.array(
                Image.fromarray(bg_img).resize((int(W * upscale), int(H * upscale)),
                                               resample=Image.BICUBIC)
            )
        except Exception:
            bg_img = np.kron(bg_img, np.ones((upscale, upscale, 1), dtype=bg_img.dtype))
        H, W = bg_img.shape[:2]
        cell_w *= upscale
        cell_h *= upscale

    # -------- Helpers (coords, sizing) --------
    def state_to_center_xy(s: int) -> tuple[float, float]:
        r, c = divmod(int(s), ncol)
        return (c + 0.5) * cell_w, (r + 0.5) * cell_h

    def px_to_pt(px: float) -> float:
        return float(px) * 72.0 / float(dpi)

    def fmt_val(v: float) -> str:
        if value_format is not None:
            return format(v, value_format)
        # auto with explicit sign
        return (f"{v:+.2e}" if abs(v) < 0.01 and v != 0.0 else f"{v:+.2f}")

    cell_min = min(cell_w, cell_h)
    font_pt = max(6.0, min(14.0, px_to_pt(font_scale * cell_min)))
    title_pt = max(10.0, min(16.0, px_to_pt(0.20 * cell_min)))

    # Text style
    text_bbox = dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.15")
    text_effects = [pe.withStroke(linewidth=px_to_pt(1.0), foreground="black", alpha=0.35)]

    # -------- Build grids and difference --------
    def val_get(tbl, s: int) -> float:
        if hasattr(tbl, "get_value"):
            return float(tbl.get_value(int(s)))
        try:
            return float(tbl.get(int(s), 0.0))  # type: ignore[attr-defined]
        except Exception:
            return 0.0

    grid_a = np.zeros((nrow, ncol), dtype=float)
    grid_b = np.zeros((nrow, ncol), dtype=float)
    for s in range(nS):
        r, c = divmod(s, ncol)
        grid_a[r, c] = val_get(values_a, s)
        grid_b[r, c] = val_get(values_b, s)

    diff_grid = grid_a - grid_b

    # Default symmetric vmin/vmax around zero
    finite_mask = np.isfinite(diff_grid)
    max_abs = float(np.nanmax(np.abs(diff_grid[finite_mask]))) if finite_mask.any() else 1.0
    if vmin is None or vmax is None:
        vmin = -max_abs
        vmax = max_abs
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

    # -------- Plot --------
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
            val = diff_grid[r, c]
            if not np.isfinite(val) or abs(val) < min_abs_label:
                continue
            x, y = state_to_center_xy(s)
            ax.text(
                x, y, fmt_val(val),
                ha="center", va="center", fontsize=font_pt,
                bbox=text_bbox, alpha=0.95, zorder=2, path_effects=text_effects,
            )

    out_name = f"{filename_prefix}.png"
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    plt.close(fig)

    print(f"[OK] Saved scalar diff overlay to: {os.path.abspath(out_path)}")
    return os.path.abspath(out_path)
