import os

from gymnasium.envs.toy_text.taxi import TaxiEnv
from apis.customisable import CustomisableEnvAbs
from typing import Tuple, Dict, Any, List, Union
from gymnasium import spaces
from gymnasium.core import ObsType
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib import patheffects as pe
import matplotlib.colors as mcolors
import numpy as np

from mdp_network import MDPNetwork
from mdp_network.mdp_tables import ValueTable


class CustomisedTaxiEnv(TaxiEnv, CustomisableEnvAbs):
    """
    A customised Taxi environment that implements state encoding and decoding functionality.

    This class extends the standard Taxi environment with the ability to encode the current
    state into a compact representation and decode it back to restore the environment state.

    The Taxi environment state consists of:
    - Taxi position (row, col): 5x5 grid = 25 positions
    - Passenger location: 5 possible values (4 pickup locations + in taxi)
    - Destination: 4 possible locations (R, G, Y, B)

    State encoding formula: ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
    """

    def __init__(
        self,
        render_mode: str = None,
        is_rainy: bool = False,
        fickle_passenger: bool = False,
        networkx_env=None,
    ):
        """
        Initialize the customised Taxi environment.

        Args:
            render_mode: Rendering mode ('human', 'ansi', 'rgb_array', or None)
            is_rainy: If True, movement actions have stochastic effects
            fickle_passenger: If True, passenger may change destination during trip
            networkx_env: Optional NetworkXMDPEnvironment for external state control
        """
        TaxiEnv.__init__(
            self,
            render_mode=render_mode,
            is_rainy=is_rainy,
            fickle_passenger=fickle_passenger,
        )
        CustomisableEnvAbs.__init__(self, networkx_env=networkx_env)

    def step(self, action):
        """Override step method to optionally use NetworkX environment."""
        if self.networkx_env is not None:
            # Get current encoded state
            current_encoded_state = self.encode_state()

            # Map to NetworkX state space (assuming direct mapping for now)
            # You might need to implement a mapping function here
            networkx_state = current_encoded_state

            # Set NetworkX environment to current state
            self.networkx_env.current_state = networkx_state

            # Execute step in NetworkX environment
            next_networkx_state, reward, terminated, truncated, info = (
                self.networkx_env.step(action)
            )

            # Map back to our state space and decode
            next_encoded_state = next_networkx_state
            obs, decode_info = self.decode_state(next_encoded_state)

            # Merge info dictionaries
            info.update(decode_info)

            return obs, reward, terminated, truncated, info
        else:
            # Use original Taxi environment step
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
        Otherwise, fall back to the native Taxi reset.
        """
        if self.networkx_env is not None:
            sp, backend_info = self.networkx_env.reset(seed=seed)
            sp = int(sp)

            # Keep current_state in sync explicitly
            self.networkx_env.current_state = sp

            # Decode into Taxi internal state (sets self.s, etc.)
            obs, decode_info = self.decode_state(sp)

            # Match base env bookkeeping
            self.lastaction = None

            if self.render_mode == "human":
                self.render()

            info = {}
            if isinstance(backend_info, dict):
                info.update(backend_info)
            if isinstance(decode_info, dict):
                info.update(decode_info)
            return obs, info

        # Fallback: native dynamics
        return super().reset(seed=seed, options=options)

    def encode_state(self) -> int:
        """
        Encode the current environment state into a compact integer representation.

        The state is encoded using the same formula as the original Taxi environment:
        state = ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination

        Returns:
            int: Encoded state representing current environment configuration
        """
        return int(self.s)

    def decode_state(self, state: int) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Decode an encoded state integer and set the environment to that state.

        This method forcibly sets the environment to the specified state, allowing
        for state restoration and manipulation.

        Args:
            state (int): The encoded state integer to decode and set

        Returns:
            Tuple[ObsType, Dict[str, Any]]: Observation and info dict after state restoration

        Raises:
            ValueError: If the provided state is invalid (outside valid range)
        """
        # Validate state range
        if not (0 <= state < 500):
            raise ValueError(
                f"Invalid state: {state}. State must be between 0 and 499."
            )

        # Decode the state components
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)

        # Validate decoded components
        if not (0 <= taxi_row <= 4):
            raise ValueError(f"Invalid taxi row: {taxi_row}")
        if not (0 <= taxi_col <= 4):
            raise ValueError(f"Invalid taxi column: {taxi_col}")
        if not (0 <= pass_loc <= 4):
            raise ValueError(f"Invalid passenger location: {pass_loc}")
        if not (0 <= dest_idx <= 3):
            raise ValueError(f"Invalid destination index: {dest_idx}")

        # Additional validation: passenger and destination should not be the same
        # unless the passenger is in the taxi (pass_loc == 4)
        # if pass_loc < 4 and pass_loc == dest_idx:
        #     raise ValueError(
        #         f"Invalid state: passenger at location {pass_loc} cannot have "
        #         f"the same location as destination {dest_idx}"
        #     )

        # Set the environment state
        self.s = state

        for attr, default in (
            ("lastaction", None),
            ("fickle_step", False),
            ("taxi_orientation", 0),
            ("step_count", 0),
        ):
            if hasattr(self, attr):
                setattr(self, attr, default)

        # Generate observation
        observation = int(self.s)

        # Generate info dict with current state information
        info = {
            "prob": 1.0,  # Deterministic state setting
            "action_mask": self.action_mask(self.s),
            "taxi_row": taxi_row,
            "taxi_col": taxi_col,
            "passenger_location": pass_loc,
            "destination": dest_idx,
            "state_components": {
                "taxi_position": (taxi_row, taxi_col),
                "passenger_location": pass_loc,
                "destination": dest_idx,
                "passenger_in_taxi": pass_loc == 4,
                "at_destination": (
                    (taxi_row, taxi_col) == self.locs[dest_idx]
                    if pass_loc == 4
                    else False
                ),
            },
        }

        # Trigger rendering if in human mode
        if self.render_mode == "human":
            self.render()

        return observation, info

    def get_start_states(self) -> List[int]:
        """
        Get all possible starting states for the Taxi environment.

        This method dynamically determines valid starting states based on the current
        environment configuration (grid size and pickup locations).

        Valid starting states are those where:
        - The passenger is at one of the pickup locations (not in taxi)
        - The destination is different from the passenger's current location
        - The taxi can be at any valid position on the grid

        Returns:
            List[int]: List of all valid starting state integers
        """
        start_states = []

        # Get environment dimensions and pickup locations dynamically
        num_rows = self.max_row + 1
        num_cols = self.max_col + 1
        num_pickup_locs = len(self.locs)

        # Iterate through all possible combinations
        for taxi_row in range(num_rows):
            for taxi_col in range(num_cols):
                for pass_loc in range(
                    num_pickup_locs
                ):  # Passenger at pickup locations only
                    for dest_idx in range(num_pickup_locs):  # All possible destinations
                        # Only include states where passenger location != destination
                        # (passenger shouldn't start at their destination)
                        if pass_loc != dest_idx:
                            state = self.encode(taxi_row, taxi_col, pass_loc, dest_idx)
                            start_states.append(state)

        return start_states

    def get_state_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current state.

        Returns:
            Dict[str, Any]: Dictionary containing detailed state information
        """
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(self.s)

        return {
            "encoded_state": self.s,
            "taxi_position": (taxi_row, taxi_col),
            "passenger_location": pass_loc,
            "destination": dest_idx,
            "passenger_in_taxi": pass_loc == 4,
            "at_pickup_location": (
                (taxi_row, taxi_col) == self.locs[pass_loc] if pass_loc < 4 else False
            ),
            "at_destination": (taxi_row, taxi_col) == self.locs[dest_idx],
            "available_actions": np.where(self.action_mask(self.s) == 1)[0].tolist(),
            "location_names": ["Red", "Green", "Yellow", "Blue"],
            "action_names": ["South", "North", "East", "West", "Pickup", "Dropoff"],
        }

    def is_valid_state(self, state: int) -> bool:
        """
        Check if a given state is valid.

        Args:
            state (int): State to validate

        Returns:
            bool: True if state is valid, False otherwise
        """
        if not (0 <= state < 500):
            return False

        try:
            taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)

            # Check component ranges
            if not (0 <= taxi_row <= 4 and 0 <= taxi_col <= 4):
                return False
            if not (0 <= pass_loc <= 4):
                return False
            if not (0 <= dest_idx <= 3):
                return False

            # Check logical consistency: passenger and destination shouldn't be the same
            # unless passenger is in taxi
            if pass_loc < 4 and pass_loc == dest_idx:
                return False

            return True

        except Exception:
            return False

    def get_mdp_network(self) -> MDPNetwork:
        """
        Build an MDPNetwork from current TaxiEnv dynamics (self.P).
        Tags: 4 combos (in/out zone) x (with/without passenger).
        """
        num_states: int = self.observation_space.n
        num_actions: int = self.action_space.n

        # Core sets
        states = list(range(num_states))
        start_states = [s for s, w in enumerate(self.initial_state_distrib) if w > 0.0]

        terminal_states_set = set()
        for s in range(num_states):
            for a in range(num_actions):
                for p, sp, r, done in self.P[s][a]:
                    if done:
                        terminal_states_set.add(int(sp))
        terminal_states = sorted(terminal_states_set)

        # Transitions: transitions["s"]["a"]["sp"] = {"p": prob, "r": reward}
        from typing import Dict, List

        transitions: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
        for s in range(num_states):
            s_key = str(s)
            for a in range(num_actions):
                entries = self.P[s][a]
                if not entries:
                    continue
                a_key = str(a)
                # Aggregate duplicate (s,a,s')
                accum: Dict[int, Dict[str, float]] = {}
                for p, sp, r, done in entries:
                    sp = int(sp)
                    acc = accum.setdefault(sp, {"p": 0.0, "r": 0.0})
                    new_p = acc["p"] + float(p)
                    acc["r"] = (
                        (acc["r"] * acc["p"] + float(r) * float(p)) / new_p
                        if new_p > 0.0
                        else float(r)
                    )
                    acc["p"] = new_p
                if accum:
                    transitions.setdefault(s_key, {})
                    a_bucket = transitions[s_key].setdefault(a_key, {})
                    for sp, v in accum.items():
                        a_bucket[str(sp)] = {"p": float(v["p"]), "r": float(v["r"])}

        # ----- 4 tags only -----
        colour_cells = set(
            self.locs
        )  # taxi is "in zone" iff (row,col) in these 4 colored cells
        out_zone_no_passenger: List[int] = []
        in_zone_no_passenger: List[int] = []
        out_zone_with_passenger: List[int] = []
        in_zone_with_passenger: List[int] = []

        for s in range(num_states):
            taxi_row, taxi_col, pass_loc, dest_idx = self.decode(s)
            in_zone = (taxi_row, taxi_col) in colour_cells
            carrying = pass_loc == 4
            if in_zone and not carrying:
                in_zone_no_passenger.append(s)
            elif in_zone and carrying:
                in_zone_with_passenger.append(s)
            elif (not in_zone) and not carrying:
                out_zone_no_passenger.append(s)
            else:  # not in_zone and carrying
                out_zone_with_passenger.append(s)

        tags = {
            "out_zone_no_passenger": sorted(out_zone_no_passenger),
            "in_zone_no_passenger": sorted(in_zone_no_passenger),
            "out_zone_with_passenger": sorted(out_zone_with_passenger),
            "in_zone_with_passenger": sorted(in_zone_with_passenger),
        }

        config = {
            "num_actions": int(num_actions),
            "states": states,
            "start_states": start_states,
            "terminal_states": terminal_states,
            "default_reward": -1.0,  # Taxi default step reward
            "transitions": transitions,
            "tags": tags,
        }
        return MDPNetwork(config_data=config)


# ---------- Taxi plotting helpers ----------

_TAXI_MOVE_ACTIONS = [0, 1, 2, 3]  # South, North, East, West
_TAXI_ACTION_NAMES = ["South", "North", "East", "West"]


def _taxi_grid_shape(env) -> tuple[int, int]:
    """Infer (nrow, ncol) from Taxi env."""
    nrow = int(getattr(env, "max_row", 4)) + 1
    ncol = int(getattr(env, "max_col", 4)) + 1
    return nrow, ncol


def _make_blank_board(nrow: int, ncol: int, cell_px: int) -> np.ndarray:
    """
    Build a simple RGB background: white board with light grid lines.
    Returns HxWx3 uint8 image.
    """
    H, W = nrow * cell_px, ncol * cell_px
    img = np.ones((H, W, 3), dtype=np.uint8) * 255  # white
    # grid lines
    for r in range(1, nrow):
        y = int(r * cell_px)
        img[max(y - 1, 0) : y + 1, :, :] = 230  # light gray
    for c in range(1, ncol):
        x = int(c * cell_px)
        img[:, max(x - 1, 0) : x + 1, :] = 230
    return img


def _draw_landmarks_rgby(ax, env, cell_w: float, cell_h: float, alpha: float = 0.25):
    """
    Draw semi-transparent colored rectangles at the 4 landmark cells R,G,Y,B.
    This provides orientation similar to the gym Taxi map.
    """
    try:
        locs = list(getattr(env, "locs", [(0, 0), (0, 4), (4, 0), (4, 3)]))
        colors = list(
            getattr(
                env,
                "locs_colors",
                [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)],
            )
        )
    except Exception:
        locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]
    for (r, c), rgb in zip(locs, colors):
        x0, y0 = c * cell_w, r * cell_h
        rect = plt.Rectangle(
            (x0, y0),
            cell_w,
            cell_h,
            facecolor=np.array(rgb) / 255.0,
            edgecolor="none",
            alpha=alpha,
            zorder=0.5,
        )
        ax.add_patch(rect)


def _px_to_pt(px: float, dpi: int) -> float:
    return float(px) * 72.0 / float(dpi)


def _make_blank_board_char_ratio(nrow: int, ncol: int, char_px: int = 24) -> np.ndarray:
    """
    Build a white board that matches the Taxi render aspect ratio (ascii grid 11x7 for 5x5).
    Draw light grid lines aligned to Taxi's *character* grid (with outer margins).
    """
    ascii_rows = nrow + 2  # top + bottom margins
    ascii_cols = 2 * ncol + 1  # vertical bars layout
    H, W = ascii_rows * char_px, ascii_cols * char_px
    img = np.ones((H, W, 3), dtype=np.uint8) * 255  # white

    # Character cell sizes
    ch = H / ascii_rows
    cw = W / ascii_cols

    # Draw Taxi interior grid lines (only the playable 5x5 area)
    # Vertical lines at x = cw*(1 + 2*k), k=0..ncol
    # Horizontal lines at y = ch*(1 + r), r=0..nrow
    color = 230
    for k in range(ncol + 1):
        x = int(round(cw * (1 + 2 * k)))
        img[:, max(x - 1, 0) : x + 1, :] = color
    for r in range(nrow + 1):
        y = int(round(ch * (1 + r)))
        img[max(y - 1, 0) : y + 1, :, :] = color
    return img


def _draw_landmarks_rgby_char(ax, env, cw: float, ch: float, alpha: float = 0.22):
    """
    Draw semi-transparent RG/Y/B landmark rectangles aligned to the *character-grid*
    geometry used by the Taxi renderer (with outer margins).
    """
    try:
        locs = list(getattr(env, "locs", [(0, 0), (0, 4), (4, 0), (4, 3)]))
        colors = list(
            getattr(
                env,
                "locs_colors",
                [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)],
            )
        )
    except Exception:
        locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]

    for (r, c), rgb in zip(locs, colors):
        # Taxi cell spans x in [(2c+1)cw, (2c+3)cw], y in [(r+1)ch, (r+2)ch]
        x0 = (2 * c + 1) * cw
        y0 = (r + 1) * ch
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                2 * cw,
                ch,
                facecolor=np.array(rgb) / 255.0,
                edgecolor="none",
                alpha=alpha,
                zorder=0.5,
            )
        )


def _render_taxi_bg_image(
    env: Union[TaxiEnv, "CustomisedTaxiEnv"],
    dest_idx: int,
    pass_loc: int,
    cell_px: int = 120,
) -> tuple[np.ndarray, bool]:
    """
    Render an RGB background from the live Taxi environment for a representative state
    consistent with (pass_loc, dest_idx). If real rendering fails (e.g., pygame missing),
    fall back to a synthetic board that preserves the Taxi render aspect ratio.

    Returns:
        (img, used_env): img is HxWx3 uint8, used_env=True if real env render succeeded.
    """
    nrow, ncol = _taxi_grid_shape(env)
    used_env = False
    prev_mode = getattr(env, "render_mode", None)

    try:
        # Pick a neutral taxi cell (center) – any row/col is fine for background
        r0, c0 = nrow // 2, ncol // 2
        state = int(env.encode(int(r0), int(c0), int(pass_loc), int(dest_idx)))

        # Switch to rgb_array render, set state, and render
        env.render_mode = "rgb_array"
        try:
            # Prefer the custom decode_state (resets bookkeeping cleanly)
            env.decode_state(state)  # type: ignore[attr-defined]
        except Exception:
            env.s = state
            if hasattr(env, "lastaction"):
                env.lastaction = None

        img = env.render()
        if img is None:
            img = env._render_gui("rgb_array")  # type: ignore[attr-defined]
        used_env = True

        # Keep the *original* render size to preserve the true aspect ratio.
        # (No forced resizing to square.)
    except Exception:
        # Fallback: synth board with Taxi's 11:7 character-grid aspect ratio
        img = _make_blank_board_char_ratio(
            nrow, ncol, char_px=max(16, int(cell_px * 0.5))
        )
        used_env = False
    finally:
        env.render_mode = prev_mode

    return img, used_env


def plot_taxi_transition_overlays(
    env: Union[TaxiEnv, "CustomisedTaxiEnv"],
    mdp: MDPNetwork,
    output_dir: str,
    filename_prefix: str = "taxi_transitions_mosaic",
    min_prob: float = 0.05,
    alpha: float = 0.90,
    annotate: bool = True,
    show_self_loops: bool = True,
    dpi: int = 200,
    target_cell_px: int = 120,  # only used if we must fallback to synthetic background
    arrow_scale: float = 0.035,  # slightly thinner
    font_scale: float = 0.14,  # slightly smaller
    cmap_name: str = "viridis",
    gamma: float = 1.0,
    # choose a slice for the background so sprites (passenger/dest) look consistent
    pass_loc_for_bg: int = 4,  # IN_TAXI
    dest_idx_for_bg: int = 0,  # R
) -> str:
    """
    2x2 mosaic of per-action transition overlays (South, North, East, West) for Taxi.
    Uses a live env-rendered background (native aspect ratio). Arrows are drawn only
    over the playable 5x5 inner area; pickup/dropoff are ignored.

    We query transitions from representative states (row, col, pass=pass_loc_for_bg, dest=dest_idx_for_bg).
    For Taxi, movement dynamics do not depend on passenger/destination, so this is safe.
    """
    assert hasattr(env, "encode") and hasattr(
        env, "decode"
    ), "Taxi env must provide encode/decode."
    nrow, ncol = _taxi_grid_shape(env)
    os.makedirs(output_dir, exist_ok=True)

    # Background image (native aspect); reuse for all four panes
    bg_img, used_env = _render_taxi_bg_image(
        env,
        dest_idx=int(dest_idx_for_bg),
        pass_loc=int(pass_loc_for_bg),
        cell_px=target_cell_px,
    )
    H_bg, W_bg = bg_img.shape[:2]

    # Character-grid geometry (matches Taxi ascii renderer)
    ascii_rows = nrow + 2
    ascii_cols = 2 * ncol + 1
    cw = W_bg / ascii_cols  # character cell width
    ch = H_bg / ascii_rows  # character cell height

    # Effective taxi cell size (each grid cell spans 2*cw x 1*ch)
    cell_w_eff = 2 * cw
    cell_h_eff = ch
    cell_min = min(cell_w_eff, cell_h_eff)

    # Styling
    ARROW_LW_PT = _px_to_pt(max(1.0, arrow_scale * cell_min), dpi)
    mutation_scale = _px_to_pt(0.42 * cell_min, dpi)
    shrink_pt = _px_to_pt(0.16 * cell_min, dpi)
    font_pt = max(5.0, min(12.0, _px_to_pt(font_scale * cell_min, dpi)))
    title_pt = max(8.5, min(13.0, _px_to_pt(0.15 * cell_min, dpi)))
    text_bbox = dict(
        facecolor="white", alpha=0.50, edgecolor="none", boxstyle="round,pad=0.12"
    )
    text_effects = [
        pe.withStroke(linewidth=_px_to_pt(0.8, dpi), foreground="black", alpha=0.30)
    ]
    cmap = cm.get_cmap(cmap_name)
    prob_norm = mcolors.PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)

    def prob_to_color(p: float):
        return cmap(prob_norm(np.clip(p, 0.0, 1.0)))

    def cell_center(col: int, row: int) -> tuple[float, float]:
        # Center of the taxi cell (character-grid coordinates)
        return ((2 * col + 2) * cw, (row + 1.5) * ch)

    def inner_extent() -> list[float]:
        # Arrow layer limits (playable inner rectangle)
        return [1 * cw, (2 * ncol + 1) * cw, 1 * ch, (nrow + 1) * ch]

    def draw_self_loop(ax, x, y, p):
        color = prob_to_color(p)
        radius = 0.40 * cell_min
        arc = Arc(
            (x + 0.30 * radius, y - 0.30 * radius),
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
            (x + 0.70 * radius, y - 0.48 * radius),
            (x + 0.55 * radius, y - 0.38 * radius),
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

    # Figure sized by the background's native aspect ratio
    fig = plt.figure(figsize=((W_bg / dpi) * 2, (H_bg / dpi) * 2), dpi=dpi)
    axes = [plt.subplot(2, 2, i) for i in (1, 2, 3, 4)]
    action_names = ["South", "North", "East", "West"]

    # Shared colorbar scaffold
    sm = cm.ScalarMappable(cmap=cmap, norm=prob_norm)
    sm.set_array([])

    # Draw four panes
    for a, ax in zip([0, 1, 2, 3], axes):
        # Background (native aspect)
        ax.imshow(bg_img, origin="upper", extent=[0, W_bg, H_bg, 0], zorder=0)
        ax.set_xlim(0, W_bg)
        ax.set_ylim(H_bg, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Action: {action_names[a]}", fontsize=title_pt)

        # If fallback, add RG/Y/B landmarks for orientation
        if not used_env:
            _draw_landmarks_rgby_char(ax, env, cw, ch, alpha=0.20)

        # Arrows: iterate over taxi positions; query transitions from a representative state
        for r in range(nrow):
            for c in range(ncol):
                s_rep = int(
                    env.encode(r, c, int(pass_loc_for_bg), int(dest_idx_for_bg))
                )
                probs = mdp.get_transition_probabilities(s_rep, a)
                if not probs:
                    continue
                x0, y0 = cell_center(c, r)
                for sp, p in probs.items():
                    try:
                        tr, tc, tpass, tdest = env.decode(int(sp))
                    except Exception:
                        continue
                    p = float(p)
                    if p < min_prob:
                        continue
                    x1, y1 = cell_center(int(tc), int(tr))
                    color = prob_to_color(p)

                    if (tr == r) and (tc == c):
                        if show_self_loops:
                            draw_self_loop(ax, x0, y0, p)
                            if annotate:
                                ax.text(
                                    x0,
                                    y0 - 0.33 * ch,
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

    # One shared colorbar
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label("Transition probability", fontsize=max(7, int(title_pt * 0.6)))
    cbar.ax.tick_params(labelsize=max(6, int(font_pt * 0.9)))

    out_path = os.path.join(output_dir, f"{filename_prefix}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved Taxi transitions mosaic to: {os.path.abspath(out_path)}")
    return os.path.abspath(out_path)


def plot_taxi_scalar_overlay(
    env: Union[TaxiEnv, "CustomisedTaxiEnv"],
    value_map: "ValueTable",
    output_dir: str,
    filename_prefix: str = "taxi_scalar_facets",
    dest_idx: int = 0,
    pass_locs: List[int] | None = None,
    skip_terminal_slice: bool = True,
    alpha: float = 0.65,
    annotate: bool = True,
    dpi: int = 200,
    target_cell_px: int = 120,
    font_scale: float = 0.15,
    cmap_name: str = "magma",
    gamma: float = 1.0,
    min_abs_label: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "State Scalar",
    cbar_label: str = "Value",
    value_format: str | None = None,
) -> str:
    """
    2x2 faceted heatmap of a per-state scalar at fixed destination.
    Now each facet uses an env-rendered background (kept at its native aspect ratio),
    and the heat layer only covers the playable inner area (no ascii margins).
    """
    assert hasattr(env, "encode") and hasattr(env, "decode")
    nrow, ncol = _taxi_grid_shape(env)
    os.makedirs(output_dir, exist_ok=True)

    # Facet order: IN_TAXI + the three pickups != dest
    if pass_locs is None:
        pickups = [0, 1, 2, 3]
        pickups.remove(int(dest_idx))
        facet_pass_locs = [4] + pickups
    else:
        facet_pass_locs = list(pass_locs)

    def get_val(tbl, s: int) -> float:
        if hasattr(tbl, "get_value"):
            return float(tbl.get_value(int(s)))
        try:
            return float(tbl.get(int(s), 0.0))
        except Exception:
            return 0.0

    # Build grids & min/max
    facet_grids: list[np.ndarray | None] = []
    data_min, data_max = +np.inf, -np.inf
    for pl in facet_pass_locs:
        if skip_terminal_slice and (pl == dest_idx) and (pl != 4):
            facet_grids.append(None)
            continue
        G = np.zeros((nrow, ncol), dtype=float)
        for r in range(nrow):
            for c in range(ncol):
                s = int(env.encode(r, c, int(pl), int(dest_idx)))
                G[r, c] = get_val(value_map, s)
        facet_grids.append(G)
        data_min = min(data_min, float(np.nanmin(G)))
        data_max = max(data_max, float(np.nanmax(G)))

    # Color scale
    if not np.isfinite(data_min):
        data_min = 0.0
    if not np.isfinite(data_max):
        data_max = 1.0
    vmin = data_min if vmin is None else float(vmin)
    vmax = data_max if vmax is None else float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-9
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    # Prepare figure size from one background (native aspect ratio)
    sample_bg, _ = _render_taxi_bg_image(
        env,
        dest_idx=int(dest_idx),
        pass_loc=int(facet_pass_locs[0]),
        cell_px=target_cell_px,
    )
    H_bg, W_bg = sample_bg.shape[:2]
    fig = plt.figure(figsize=((W_bg / dpi) * 2, (H_bg / dpi) * 2), dpi=dpi)
    axes = [plt.subplot(2, 2, i) for i in (1, 2, 3, 4)]

    # Character-grid geometry (matches Taxi renderer)
    ascii_rows = nrow + 2
    ascii_cols = 2 * ncol + 1
    cw = W_bg / ascii_cols
    ch = H_bg / ascii_rows

    # Typography
    cell_w_effective = 2 * cw  # taxi cell width in pixels
    cell_h_effective = ch  # taxi cell height in pixels
    cell_min = min(cell_w_effective, cell_h_effective)
    font_pt = max(5.0, min(12.0, _px_to_pt(font_scale * cell_min, dpi)))
    title_pt = max(9.0, min(14.0, _px_to_pt(0.16 * cell_min, dpi)))
    text_bbox = dict(
        facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.12"
    )
    text_effects = [
        pe.withStroke(linewidth=_px_to_pt(0.8, dpi), foreground="black", alpha=0.3)
    ]

    def fmt_val(v: float) -> str:
        if value_format is not None:
            return format(v, value_format)
        return f"{v:.2e}" if abs(v) < 0.01 and v != 0.0 else f"{v:.2f}"

    def cell_center(c: int, r: int) -> tuple[float, float]:
        # Center of taxi cell in pixel coords
        return ((2 * c + 2) * cw, (r + 1.5) * ch)

    def inner_extent() -> list[float]:
        # Heat layer covers only the playable inner rectangle
        return [1 * cw, (2 * ncol + 1) * cw, 1 * ch, (nrow + 1) * ch]

    for ax, pl, grid in zip(axes, facet_pass_locs, facet_grids):
        # Background for this slice (native aspect)
        bg_img, used_env = _render_taxi_bg_image(
            env, dest_idx=int(dest_idx), pass_loc=int(pl), cell_px=target_cell_px
        )
        H, W = bg_img.shape[:2]
        ax.imshow(bg_img, origin="upper", extent=[0, W, H, 0], zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{title} — {['P=R','P=G','P=Y','P=B','P=IN_TAXI'][pl] if 0 <= pl <=4 else f'P={pl}'}, "
            f"D={['R','G','Y','B'][dest_idx]}",
            fontsize=title_pt,
        )

        if not used_env:
            _draw_landmarks_rgby_char(ax, env, cw, ch, alpha=0.20)

        if grid is None:
            ax.imshow(
                np.zeros((nrow, ncol)),
                origin="upper",
                extent=inner_extent(),
                cmap="Greys",
                alpha=0.15,
                interpolation="nearest",
                zorder=1,
            )
            ax.text(
                W * 0.5,
                H * 0.5,
                "terminal (skipped)",
                ha="center",
                va="center",
                fontsize=font_pt,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
            )
            continue

        ax.imshow(
            grid,
            origin="upper",
            cmap=cmap,
            norm=norm,
            extent=inner_extent(),
            alpha=alpha,
            zorder=1,
            interpolation="nearest",
        )

        if annotate:
            for r in range(nrow):
                for c in range(ncol):
                    v = float(grid[r, c])
                    if abs(v) < min_abs_label:
                        continue
                    x, y = cell_center(c, r)
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

    # Shared colorbar (smaller labels)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=max(7, int(title_pt * 0.6)))
    cbar.ax.tick_params(labelsize=max(6, int(font_pt * 0.9)))

    out_path = os.path.join(output_dir, f"{filename_prefix}_D{dest_idx}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved Taxi scalar facets to: {os.path.abspath(out_path)}")
    return os.path.abspath(out_path)


def plot_taxi_scalar_diff_overlay(
    env: Union[TaxiEnv, "CustomisedTaxiEnv"],
    values_a: "ValueTable",
    values_b: "ValueTable",
    output_dir: str,
    filename_prefix: str = "taxi_scalar_diff_facets",
    dest_idx: int = 0,
    pass_locs: List[int] | None = None,
    skip_terminal_slice: bool = True,
    alpha: float = 0.65,
    annotate: bool = True,
    dpi: int = 200,
    target_cell_px: int = 120,
    font_scale: float = 0.15,
    cmap_name: str = "coolwarm",
    min_abs_label: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "Δ State Value (A − B)",
    cbar_label: str = "Δ value (A − B)",
    value_format: str | None = "+.2f",
) -> str:
    """
    2x2 faceted diverging heatmap of differences (A − B) at fixed destination.
    Backgrounds keep the native Taxi aspect; diff layer covers only the inner area.
    """
    assert hasattr(env, "encode") and hasattr(env, "decode")
    nrow, ncol = _taxi_grid_shape(env)
    os.makedirs(output_dir, exist_ok=True)

    if pass_locs is None:
        pickups = [0, 1, 2, 3]
        pickups.remove(int(dest_idx))
        facet_pass_locs = [4] + pickups
    else:
        facet_pass_locs = list(pass_locs)

    def get_val(tbl, s: int) -> float:
        if hasattr(tbl, "get_value"):
            return float(tbl.get_value(int(s)))
        try:
            return float(tbl.get(int(s), 0.0))
        except Exception:
            return 0.0

    facet_grids: list[np.ndarray | None] = []
    max_abs = 0.0
    for pl in facet_pass_locs:
        if skip_terminal_slice and (pl == dest_idx) and (pl != 4):
            facet_grids.append(None)
            continue
        G = np.zeros((nrow, ncol), dtype=float)
        for r in range(nrow):
            for c in range(ncol):
                s = int(env.encode(r, c, int(pl), int(dest_idx)))
                G[r, c] = get_val(values_a, s) - get_val(values_b, s)
        facet_grids.append(G)
        if np.isfinite(G).any():
            max_abs = max(max_abs, float(np.nanmax(np.abs(G))))

    # Symmetric diverging limits
    if (vmin is None) or (vmax is None):
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

    # Figure size from one background (native aspect)
    sample_bg, _ = _render_taxi_bg_image(
        env,
        dest_idx=int(dest_idx),
        pass_loc=int(facet_pass_locs[0]),
        cell_px=target_cell_px,
    )
    H_bg, W_bg = sample_bg.shape[:2]
    fig = plt.figure(figsize=((W_bg / dpi) * 2, (H_bg / dpi) * 2), dpi=dpi)
    axes = [plt.subplot(2, 2, i) for i in (1, 2, 3, 4)]

    # Character-grid geometry
    ascii_rows = nrow + 2
    ascii_cols = 2 * ncol + 1
    cw = W_bg / ascii_cols
    ch = H_bg / ascii_rows

    # Typography
    cell_w_effective = 2 * cw
    cell_h_effective = ch
    cell_min = min(cell_w_effective, cell_h_effective)
    font_pt = max(5.0, min(12.0, _px_to_pt(font_scale * cell_min, dpi)))
    title_pt = max(9.0, min(14.0, _px_to_pt(0.16 * cell_min, dpi)))
    text_bbox = dict(
        facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.12"
    )
    text_effects = [
        pe.withStroke(linewidth=_px_to_pt(0.8, dpi), foreground="black", alpha=0.3)
    ]

    def fmt_val(v: float) -> str:
        if value_format is not None:
            return format(v, value_format)
        return f"{v:+.2e}" if abs(v) < 0.01 and v != 0.0 else f"{v:+.2f}"

    def cell_center(c: int, r: int) -> tuple[float, float]:
        return ((2 * c + 2) * cw, (r + 1.5) * ch)

    def inner_extent() -> list[float]:
        return [1 * cw, (2 * ncol + 1) * cw, 1 * ch, (nrow + 1) * ch]

    for ax, pl, grid in zip(axes, facet_pass_locs, facet_grids):
        bg_img, used_env = _render_taxi_bg_image(
            env, dest_idx=int(dest_idx), pass_loc=int(pl), cell_px=target_cell_px
        )
        H, W = bg_img.shape[:2]
        ax.imshow(bg_img, origin="upper", extent=[0, W, H, 0], zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{title} — {['P=R','P=G','P=Y','P=B','P=IN_TAXI'][pl] if 0 <= pl <=4 else f'P={pl}'}, "
            f"D={['R','G','Y','B'][dest_idx]}",
            fontsize=title_pt,
        )

        if not used_env:
            _draw_landmarks_rgby_char(ax, env, cw, ch, alpha=0.20)

        if grid is None:
            ax.imshow(
                np.zeros((nrow, ncol)),
                origin="upper",
                extent=inner_extent(),
                cmap="Greys",
                alpha=0.15,
                interpolation="nearest",
                zorder=1,
            )
            ax.text(
                W * 0.5,
                H * 0.5,
                "terminal (skipped)",
                ha="center",
                va="center",
                fontsize=font_pt,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
            )
            continue

        ax.imshow(
            grid,
            origin="upper",
            cmap=cmap,
            norm=norm,
            extent=inner_extent(),
            alpha=alpha,
            zorder=1,
            interpolation="nearest",
        )

        if annotate:
            for r in range(nrow):
                for c in range(ncol):
                    v = float(grid[r, c])
                    if not np.isfinite(v) or abs(v) < min_abs_label:
                        continue
                    x, y = cell_center(c, r)
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

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=max(7, int(title_pt * 0.6)))
    cbar.ax.tick_params(labelsize=max(6, int(font_pt * 0.9)))

    out_path = os.path.join(output_dir, f"{filename_prefix}_D{dest_idx}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved Taxi scalar diff facets to: {os.path.abspath(out_path)}")
    return os.path.abspath(out_path)
