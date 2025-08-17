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

    def __init__(self, render_mode: str = None, is_rainy: bool = False, fickle_passenger: bool = False, networkx_env=None):
        """
        Initialize the customised Taxi environment.

        Args:
            render_mode: Rendering mode ('human', 'ansi', 'rgb_array', or None)
            is_rainy: If True, movement actions have stochastic effects
            fickle_passenger: If True, passenger may change destination during trip
            networkx_env: Optional NetworkXMDPEnvironment for external state control
        """
        TaxiEnv.__init__(self, render_mode=render_mode, is_rainy=is_rainy, fickle_passenger=fickle_passenger)
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
            next_networkx_state, reward, terminated, truncated, info = self.networkx_env.step(action)

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
            raise ValueError(f"Invalid state: {state}. State must be between 0 and 499.")

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
                "at_destination": (taxi_row, taxi_col) == self.locs[dest_idx] if pass_loc == 4 else False
            }
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
                for pass_loc in range(num_pickup_locs):  # Passenger at pickup locations only
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
            "at_pickup_location": (taxi_row, taxi_col) == self.locs[pass_loc] if pass_loc < 4 else False,
            "at_destination": (taxi_row, taxi_col) == self.locs[dest_idx],
            "available_actions": np.where(self.action_mask(self.s) == 1)[0].tolist(),
            "location_names": ["Red", "Green", "Yellow", "Blue"],
            "action_names": ["South", "North", "East", "West", "Pickup", "Dropoff"]
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
                    acc["r"] = (acc["r"] * acc["p"] + float(r) * float(p)) / new_p if new_p > 0.0 else float(r)
                    acc["p"] = new_p
                if accum:
                    transitions.setdefault(s_key, {})
                    a_bucket = transitions[s_key].setdefault(a_key, {})
                    for sp, v in accum.items():
                        a_bucket[str(sp)] = {"p": float(v["p"]), "r": float(v["r"])}

        # ----- 4 tags only -----
        colour_cells = set(self.locs)  # taxi is "in zone" iff (row,col) in these 4 colored cells
        out_zone_no_passenger: List[int] = []
        in_zone_no_passenger: List[int] = []
        out_zone_with_passenger: List[int] = []
        in_zone_with_passenger: List[int] = []

        for s in range(num_states):
            taxi_row, taxi_col, pass_loc, dest_idx = self.decode(s)
            in_zone = (taxi_row, taxi_col) in colour_cells
            carrying = (pass_loc == 4)
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
        img[max(y - 1, 0):y + 1, :, :] = 230  # light gray
    for c in range(1, ncol):
        x = int(c * cell_px)
        img[:, max(x - 1, 0):x + 1, :] = 230
    return img


def _draw_landmarks_rgby(ax, env, cell_w: float, cell_h: float, alpha: float = 0.25):
    """
    Draw semi-transparent colored rectangles at the 4 landmark cells R,G,Y,B.
    This provides orientation similar to the gym Taxi map.
    """
    try:
        locs = list(getattr(env, "locs", [(0, 0), (0, 4), (4, 0), (4, 3)]))
        colors = list(getattr(env, "locs_colors", [(255, 0, 0), (0, 255, 0),
                                                   (255, 255, 0), (0, 0, 255)]))
    except Exception:
        locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]
    for (r, c), rgb in zip(locs, colors):
        x0, y0 = c * cell_w, r * cell_h
        rect = plt.Rectangle((x0, y0), cell_w, cell_h,
                             facecolor=np.array(rgb) / 255.0, edgecolor="none", alpha=alpha, zorder=0.5)
        ax.add_patch(rect)


def _px_to_pt(px: float, dpi: int) -> float:
    return float(px) * 72.0 / float(dpi)


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
    target_cell_px: int = 120,          # a bit smaller than FrozenLake default (board is 5x5 but we have 4 panes)
    arrow_scale: float = 0.04,
    font_scale: float = 0.16,
    cmap_name: str = "viridis",
    gamma: float = 1.0,
) -> str:
    """
    Draw a 2x2 mosaic of per-action overlays (South, North, East, West) for the Taxi MDP.
    Only movement actions {0..3} are visualized. Pickup(4)/Dropoff(5) are intentionally ignored.

    For each taxi grid cell (row, col), we use a representative state with passenger=in taxi (4),
    destination=0 (arbitrary but fixed) to read movement transitions, since movement dynamics
    do not depend on passenger/destination in Taxi.
    """
    assert hasattr(env, "encode") and hasattr(env, "decode"), "Taxi env must provide encode/decode."
    nrow, ncol = _taxi_grid_shape(env)
    os.makedirs(output_dir, exist_ok=True)

    # --- Build a neutral board background (no sprites) ---
    bg_img = _make_blank_board(nrow, ncol, target_cell_px)
    H, W = bg_img.shape[:2]
    cell_w, cell_h = W / ncol, H / nrow

    # --- Visual style helpers ---
    cell_min = min(cell_w, cell_h)
    ARROW_LW_PT = _px_to_pt(max(1.0, arrow_scale * cell_min), dpi)
    mutation_scale = _px_to_pt(0.45 * cell_min, dpi)
    shrink_pt = _px_to_pt(0.18 * cell_min, dpi)
    font_pt = max(6.0, min(12.0, _px_to_pt(font_scale * cell_min, dpi)))
    title_pt = max(9.0, min(14.0, _px_to_pt(0.18 * cell_min, dpi)))
    text_bbox = dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.15")
    text_effects = [pe.withStroke(linewidth=_px_to_pt(1.0, dpi), foreground="black", alpha=0.35)]
    cmap = cm.get_cmap(cmap_name)
    prob_norm = mcolors.PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)

    def prob_to_color(p: float):
        return cmap(prob_norm(np.clip(p, 0.0, 1.0)))

    def state_center_xy(row: int, col: int) -> tuple[float, float]:
        return (col + 0.5) * cell_w, (row + 0.5) * cell_h

    def draw_self_loop(ax, x, y, p):
        color = prob_to_color(p)
        radius = 0.30 * cell_min
        arc = Arc((x + 0.35 * radius, y - 0.35 * radius),
                  width=radius, height=radius,
                  angle=0, theta1=30, theta2=320,
                  linewidth=ARROW_LW_PT, color=color, alpha=alpha, zorder=3)
        ax.add_patch(arc)
        arr = FancyArrowPatch(
            (x + 0.75 * radius, y - 0.52 * radius),
            (x + 0.60 * radius, y - 0.40 * radius),
            arrowstyle="->", mutation_scale=mutation_scale,
            linewidth=ARROW_LW_PT, facecolor=color, edgecolor=color,
            alpha=alpha, zorder=4, shrinkA=0.0, shrinkB=0.0
        )
        ax.add_patch(arr)

    # --- Build figure with 2x2 axes ---
    fig_w_in = (W / dpi) * 2
    fig_h_in = (H / dpi) * 2
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    axes = [
        plt.subplot(2, 2, 1),
        plt.subplot(2, 2, 2),
        plt.subplot(2, 2, 3),
        plt.subplot(2, 2, 4),
    ]

    # Shared colorbar scaffold
    sm = cm.ScalarMappable(cmap=cmap, norm=prob_norm)
    sm.set_array([])

    for a, ax in zip(_TAXI_MOVE_ACTIONS, axes):
        # Base board
        ax.imshow(bg_img, origin="upper", extent=[0, W, H, 0], zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Action: {_TAXI_ACTION_NAMES[a]}", fontsize=title_pt)

        # Landmarks for orientation
        _draw_landmarks_rgby(ax, env, cell_w, cell_h, alpha=0.20)

        # Draw arrows for each taxi grid cell (representative state has pass=4, dest=0)
        for r in range(nrow):
            for c in range(ncol):
                s_rep = int(env.encode(r, c, 4, 0))
                probs = mdp.get_transition_probabilities(s_rep, a)
                if not probs:
                    continue
                x0, y0 = state_center_xy(r, c)
                for sp, p in probs.items():
                    if p < min_prob:
                        continue
                    try:
                        tr, tc, tpass, tdest = env.decode(int(sp))
                    except Exception:
                        continue
                    x1, y1 = state_center_xy(int(tr), int(tc))
                    color = prob_to_color(float(p))
                    if (tr == r) and (tc == c):
                        if show_self_loops:
                            draw_self_loop(ax, x0, y0, float(p))
                            if annotate:
                                ax.text(x0, y0 - 0.33 * cell_h, f"{p:.2f}",
                                        ha="center", va="center", fontsize=font_pt,
                                        bbox=text_bbox, alpha=alpha, zorder=5,
                                        path_effects=text_effects)
                        continue
                    arrow = FancyArrowPatch(
                        (x0, y0), (x1, y1),
                        arrowstyle="->", mutation_scale=mutation_scale,
                        linewidth=ARROW_LW_PT, facecolor=color, edgecolor=color,
                        alpha=alpha, zorder=3, shrinkA=shrink_pt, shrinkB=shrink_pt
                    )
                    ax.add_patch(arrow)
                    if annotate:
                        mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
                        ax.text(mx, my, f"{p:.2f}", ha="center", va="center",
                                fontsize=font_pt, bbox=text_bbox, alpha=alpha,
                                zorder=4, path_effects=text_effects)

    # One shared colorbar on the right
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label("Transition probability", fontsize=max(8, int(title_pt * 0.7)))

    out_path = os.path.join(output_dir, f"{filename_prefix}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved Taxi transitions mosaic to: {os.path.abspath(out_path)}")
    return os.path.abspath(out_path)


def plot_taxi_scalar_overlay(
    env: Union[TaxiEnv, "CustomisedTaxiEnv"],
    value_map: "ValueTable",                 # supports .get_value(s) or dict-like
    output_dir: str,
    filename_prefix: str = "taxi_scalar_facets",
    # Faceting controls:
    dest_idx: int = 0,                       # fixed destination page (0=R,1=G,2=Y,3=B)
    pass_locs: List[int] | None = None,      # which passenger-locs to facet; default -> [IN_TAXI, others≠dest]
    skip_terminal_slice: bool = True,        # skip facet where pass_loc == dest_idx (terminal-like)
    # Style:
    alpha: float = 0.65,
    annotate: bool = True,
    dpi: int = 200,
    target_cell_px: int = 120,
    font_scale: float = 0.18,
    cmap_name: str = "magma",
    gamma: float = 1.0,
    min_abs_label: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "State Scalar",
    cbar_label: str = "Value",
    value_format: str | None = None,         # e.g. ".2f", ".2e"; None -> auto (2f/2e)
) -> str:
    """
    Draw a 2x2 faceted heatmap page for a per-state scalar on Taxi (e.g., V(s) or occupancy).
    Facets are four passenger-location slices at a fixed destination:
      - IN_TAXI (4)
      - the three pickup points excluding `dest_idx`
      - (optionally) the 'terminal' slice where pass_loc==dest_idx is skipped and replaced by a gray tile.

    Each facet shows a 5x5 grid over taxi positions; the scalar is read from the state
    encoded as (row, col, pass_loc, dest_idx).
    """
    assert hasattr(env, "encode") and hasattr(env, "decode"), "Taxi env must provide encode/decode."
    nrow, ncol = _taxi_grid_shape(env)
    os.makedirs(output_dir, exist_ok=True)

    # Determine the 4 facets: IN_TAXI + the three pickups != dest
    if pass_locs is None:
        pickups = [0, 1, 2, 3]
        pickups.remove(int(dest_idx))
        facet_pass_locs = [4] + pickups  # [IN_TAXI, the other three]
    else:
        facet_pass_locs = list(pass_locs)

    # Prepare grids and collect global min/max for shared colorbar
    def get_val(tbl, s: int) -> float:
        if hasattr(tbl, "get_value"):
            return float(tbl.get_value(int(s)))
        try:
            return float(tbl.get(int(s), 0.0))  # type: ignore[attr-defined]
        except Exception:
            return 0.0

    facet_grids: list[np.ndarray] = []
    data_min, data_max = +np.inf, -np.inf

    for pl in facet_pass_locs:
        # Skip the terminal-like slice if requested (pass at destination and not in taxi)
        if skip_terminal_slice and (pl == dest_idx) and (pl != 4):
            facet_grids.append(None)  # placeholder for "terminal (skipped)"
            continue
        grid = np.zeros((nrow, ncol), dtype=float)
        for r in range(nrow):
            for c in range(ncol):
                s = int(env.encode(r, c, int(pl), int(dest_idx)))
                grid[r, c] = get_val(value_map, s)
        facet_grids.append(grid)
        data_min = min(data_min, float(np.nanmin(grid)))
        data_max = max(data_max, float(np.nanmax(grid)))

    # Color scale (shared across facets)
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

    # Layout: 2x2 mosaic
    bg_img = _make_blank_board(nrow, ncol, target_cell_px)
    H, W = bg_img.shape[:2]
    cell_w, cell_h = W / ncol, H / nrow
    cell_min = min(cell_w, cell_h)
    font_pt = max(6.0, min(14.0, _px_to_pt(font_scale * cell_min, dpi)))
    title_pt = max(10.0, min(16.0, _px_to_pt(0.20 * cell_min, dpi)))
    text_bbox = dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.15")
    text_effects = [pe.withStroke(linewidth=_px_to_pt(1.0, dpi), foreground="black", alpha=0.35)]

    def fmt_val(v: float) -> str:
        if value_format is not None:
            return format(v, value_format)
        return (f"{v:.2e}" if abs(v) < 0.01 and v != 0.0 else f"{v:.2f}")

    fig = plt.figure(figsize=((W / dpi) * 2, (H / dpi) * 2), dpi=dpi)
    axes = [
        plt.subplot(2, 2, 1),
        plt.subplot(2, 2, 2),
        plt.subplot(2, 2, 3),
        plt.subplot(2, 2, 4),
    ]

    # Titles per facet
    def _facet_title(pl: int) -> str:
        names = ["P=R", "P=G", "P=Y", "P=B", "P=IN_TAXI"]
        return names[pl] if 0 <= pl <= 4 else f"P={pl}"

    for ax, pl, grid in zip(axes, facet_pass_locs, facet_grids):
        ax.imshow(bg_img, origin="upper", extent=[0, W, H, 0], zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        _draw_landmarks_rgby(ax, env, cell_w, cell_h, alpha=0.20)
        ax.set_title(f"{title} — {_facet_title(pl)}, D={['R','G','Y','B'][dest_idx]}", fontsize=title_pt)

        if grid is None:
            # Terminal-like slice placeholder
            ax.imshow(np.zeros((nrow, ncol)), origin="upper", extent=[0, W, H, 0],
                      cmap="Greys", alpha=0.15, interpolation="nearest", zorder=1)
            ax.text(W * 0.5, H * 0.5, "terminal (skipped)", ha="center", va="center",
                    fontsize=font_pt, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
            continue

        ax.imshow(grid, origin="upper", cmap=cmap, norm=norm, extent=[0, W, H, 0],
                  alpha=alpha, zorder=1, interpolation="nearest")

        if annotate:
            for r in range(nrow):
                for c in range(ncol):
                    v = float(grid[r, c])
                    if abs(v) < min_abs_label:
                        continue
                    x, y = (c + 0.5) * cell_w, (r + 0.5) * cell_h
                    ax.text(x, y, fmt_val(v), ha="center", va="center", fontsize=font_pt,
                            bbox=text_bbox, alpha=0.95, zorder=2, path_effects=text_effects)

    # Shared colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=max(8, int(title_pt * 0.7)))
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
    # Faceting
    dest_idx: int = 0,
    pass_locs: List[int] | None = None,
    skip_terminal_slice: bool = True,
    # Style
    alpha: float = 0.65,
    annotate: bool = True,
    dpi: int = 200,
    target_cell_px: int = 120,
    font_scale: float = 0.18,
    cmap_name: str = "coolwarm",
    min_abs_label: float = 0.0,
    vmin: float | None = None,              # if None -> symmetric by max |Δ|
    vmax: float | None = None,
    title: str = "Δ State Value (A − B)",
    cbar_label: str = "Δ value (A − B)",
    value_format: str | None = "+.2f",
) -> str:
    """
    Draw a 2x2 faceted, **diverging** heatmap page of differences (A − B) at a fixed destination.
    Facets follow the same convention as plot_taxi_scalar_overlay.

    A and B can be any per-state scalar tables (e.g., V_opt(loop) and V_opt(native)).
    """
    assert hasattr(env, "encode") and hasattr(env, "decode"), "Taxi env must provide encode/decode."
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
            return float(tbl.get(int(s), 0.0))  # type: ignore[attr-defined]
        except Exception:
            return 0.0

    facet_grids: list[np.ndarray] = []
    max_abs = 0.0

    for pl in facet_pass_locs:
        if skip_terminal_slice and (pl == dest_idx) and (pl != 4):
            facet_grids.append(None)
            continue
        grid = np.zeros((nrow, ncol), dtype=float)
        for r in range(nrow):
            for c in range(ncol):
                s = int(env.encode(r, c, int(pl), int(dest_idx)))
                grid[r, c] = get_val(values_a, s) - get_val(values_b, s)
        facet_grids.append(grid)
        if np.isfinite(grid).any():
            max_abs = max(max_abs, float(np.nanmax(np.abs(grid))))

    # Symmetric color limits
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

    bg_img = _make_blank_board(nrow, ncol, target_cell_px)
    H, W = bg_img.shape[:2]
    cell_w, cell_h = W / ncol, H / nrow
    cell_min = min(cell_w, cell_h)
    font_pt = max(6.0, min(14.0, _px_to_pt(font_scale * cell_min, dpi)))
    title_pt = max(10.0, min(16.0, _px_to_pt(0.20 * cell_min, dpi)))
    text_bbox = dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.15")
    text_effects = [pe.withStroke(linewidth=_px_to_pt(1.0, dpi), foreground="black", alpha=0.35)]

    def fmt_val(v: float) -> str:
        if value_format is not None:
            return format(v, value_format)
        return (f"{v:+.2e}" if abs(v) < 0.01 and v != 0.0 else f"{v:+.2f}")

    fig = plt.figure(figsize=((W / dpi) * 2, (H / dpi) * 2), dpi=dpi)
    axes = [
        plt.subplot(2, 2, 1),
        plt.subplot(2, 2, 2),
        plt.subplot(2, 2, 3),
        plt.subplot(2, 2, 4),
    ]

    def _facet_title(pl: int) -> str:
        names = ["P=R", "P=G", "P=Y", "P=B", "P=IN_TAXI"]
        return names[pl] if 0 <= pl <= 4 else f"P={pl}"

    for ax, pl, grid in zip(axes, facet_pass_locs, facet_grids):
        ax.imshow(bg_img, origin="upper", extent=[0, W, H, 0], zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        _draw_landmarks_rgby(ax, env, cell_w, cell_h, alpha=0.20)
        ax.set_title(f"{title} — {_facet_title(pl)}, D={['R','G','Y','B'][dest_idx]}", fontsize=title_pt)

        if grid is None:
            ax.imshow(np.zeros((nrow, ncol)), origin="upper", extent=[0, W, H, 0],
                      cmap="Greys", alpha=0.15, interpolation="nearest", zorder=1)
            ax.text(W * 0.5, H * 0.5, "terminal (skipped)", ha="center", va="center",
                    fontsize=font_pt, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
            continue

        ax.imshow(grid, origin="upper", cmap=cmap, norm=norm, extent=[0, W, H, 0],
                  alpha=alpha, zorder=1, interpolation="nearest")

        if annotate:
            for r in range(nrow):
                for c in range(ncol):
                    v = float(grid[r, c])
                    if not np.isfinite(v) or abs(v) < min_abs_label:
                        continue
                    x, y = (c + 0.5) * cell_w, (r + 0.5) * cell_h
                    ax.text(x, y, fmt_val(v), ha="center", va="center", fontsize=font_pt,
                            bbox=text_bbox, alpha=0.95, zorder=2, path_effects=text_effects)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=max(8, int(title_pt * 0.7)))
    cbar.ax.tick_params(labelsize=max(6, int(font_pt * 0.9)))

    out_path = os.path.join(output_dir, f"{filename_prefix}_D{dest_idx}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved Taxi scalar diff facets to: {os.path.abspath(out_path)}")
    return os.path.abspath(out_path)

