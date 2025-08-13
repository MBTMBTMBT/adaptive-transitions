import os
from typing import Tuple, Dict, Any, List, Union
from gymnasium.core import ObsType
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib import patheffects as pe
import matplotlib.colors as mcolors

from mdp_network.mdp_network import MDPNetwork
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from customisable_env_abs import CustomisableEnvAbs


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
        FrozenLakeEnv.__init__(self, render_mode=render_mode, desc=desc, map_name=map_name, is_slippery=is_slippery)
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
    TARGET_CELL_PX: int = 240,        # target cell size in pixels for readability
    ARROW_SCALE: float = 0.04,        # arrow linewidth as fraction of cell size
    FONT_SCALE: float = 0.16,         # label font size as fraction of cell size
    cmap_name: str = "viridis",       # colormap for probability -> color
    GAMMA: float = 1.0                # gamma correction for probability mapping
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
    upscale = int(np.ceil(TARGET_CELL_PX / min(cell_w, cell_h)))
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
    ARROW_LW_PT = px_to_pt(max(1.0, ARROW_SCALE * cell_min))
    mutation_scale = px_to_pt(0.45 * cell_min)
    shrink_pt = px_to_pt(0.18 * cell_min)
    font_pt = max(6.0, min(12.0, px_to_pt(FONT_SCALE * cell_min)))
    title_pt = max(9.0, min(14.0, px_to_pt(0.18 * cell_min)))

    # text style for labels
    text_bbox = dict(facecolor="white", alpha=0.50, edgecolor="none", boxstyle="round,pad=0.15")
    text_effects = [pe.withStroke(linewidth=px_to_pt(1.0), foreground="black", alpha=0.35)]

    # probability -> RGBA color using colormap
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.PowerNorm(gamma=GAMMA, vmin=0.0, vmax=1.0)  # NEW: gamma-consistent normalization
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
