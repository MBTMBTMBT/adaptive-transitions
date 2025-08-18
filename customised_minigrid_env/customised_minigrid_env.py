import json
import os
import random
from typing import Optional, Dict, Any, SupportsFloat, List, Union
import itertools

from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import *
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, TILE_PIXELS
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib import patheffects as pe
import matplotlib.colors as mcolors

from .simple_actions import SimpleActions
from .simple_manual_control import SimpleManualControl
from . import load_map_config_by_name
from apis.customisable import CustomisableEnvAbs
from mdp_network.mdp_network import MDPNetwork


DEFAULT_REWARD_DICT = {
    "sparse": True,
    "step_penalty": -0.1,
    "goal_reward": 0.0,
    "lava_penalty": 0.0,
    "pickup_reward": 0.0,
    "door_open_reward": 0.0,
    "door_close_penalty": 0.0,
    "key_drop_penalty": 0.0,
    "item_drop_penalty": 0.0,
}


class CustomMiniGridEnv(MiniGridEnv, CustomisableEnvAbs):
    """
    MiniGrid variant driven by JSON layout. Supports integer encode/decode of full env state.
    Random generation logic is preserved; integer decode is only supported under fixed-layout mode.
    """

    # -----------------------------
    # Init & basic helpers
    # -----------------------------
    def __init__(
        self,
        map_name: Optional[str] = None,
        json_file_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        display_size: Optional[int] = None,
        display_mode: Optional[str] = "middle",
        random_rotate: bool = False,
        random_flip: bool = False,
        custom_mission: str = "Explore and interact with objects.",
        max_steps: int = 100000,
        render_carried_objs: bool = True,
        any_key_opens_the_door: bool = False,
        reward_config: Optional[Dict[str, Any]] = None,
        networkx_env=None,
        **kwargs,
    ) -> None:
        # Exactly one source of config must be provided.
        provided = sum(x is not None for x in (map_name, json_file_path, config))
        assert provided == 1, "Provide exactly one of: map_name / json_file_path / config."

        self.txt_file_path = json_file_path  # kept for compatibility/debug
        self.display_mode = display_mode
        self.random_rotate = random_rotate
        self.random_flip = random_flip
        self.render_carried_objs = render_carried_objs
        self.any_key_opens_the_door = any_key_opens_the_door
        self.mission = custom_mission
        self.tile_size = 32
        self.skip_reset = False

        self.reward_config = {**DEFAULT_REWARD_DICT, **(reward_config or {})}
        self.cumulative_reward = 0.0

        # ---- Load config from exactly one source ----
        if map_name is not None:
            self.config = load_map_config_by_name(map_name)
            self.txt_file_path = None  # no real path when using packaged resource
        elif json_file_path is not None:
            with open(json_file_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = config

        # ---- Sizes ----
        height, width = self.config["height_width"]
        layout_size = max(height, width)
        self.display_size = layout_size if display_size is None else display_size
        assert display_mode in ["middle", "random"], "Invalid display_mode"
        assert self.display_size >= layout_size, "display_size must be >= layout layout_size"

        # ---- Parents ----
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: custom_mission),
            grid_size=self.display_size,
            max_steps=max_steps,
            **kwargs,
        )
        CustomisableEnvAbs.__init__(self, networkx_env=networkx_env)

        self.actions = SimpleActions
        self.action_space = spaces.Discrete(len(self.actions))
        self.step_count = 0

        # Integer codec spec (prepared lazily)
        self._int_spec = None

    # -----------------------------
    # Rendering helpers
    # -----------------------------
    def get_frame(self, highlight: bool = True, tile_size: int = TILE_PIXELS, agent_pov: bool = False):
        frame = super().get_frame(highlight, tile_size, agent_pov)
        return frame if not self.render_carried_objs else self.render_with_carried_objects(frame)

    def render_with_carried_objects(self, full_image):
        """
        Add an extra row/column; render carried item at bottom-right if any.
        """
        tile_size = self.tile_size
        full_height, full_width, _ = full_image.shape
        grid_color = (100 / 255, 100 / 255, 100 / 255)

        def create_empty_tile():
            canvas = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            fill_coords(canvas, point_in_rect(0, 0.031, 0, 1), grid_color)
            fill_coords(canvas, point_in_rect(0, 1, 0, 0.031), grid_color)
            canvas = np.clip(canvas, 0.0, 1.0)
            return (canvas * 255).astype(np.uint8)

        num_height_tiles = full_height // tile_size
        right_column = np.zeros((full_height, tile_size, 3), dtype=np.uint8)
        for i in range(num_height_tiles):
            empty_tile = create_empty_tile()
            sy, ey = i * tile_size, (i + 1) * tile_size
            right_column[sy:ey, :, :] = empty_tile
        remaining_height = full_height % tile_size
        if remaining_height > 0:
            right_column[num_height_tiles * tile_size:, :, :] = create_empty_tile()[:remaining_height, :, :]
        image_with_column = np.hstack([full_image, right_column])

        new_width = image_with_column.shape[1]
        num_width_tiles = new_width // tile_size
        bottom_row = np.zeros((tile_size, new_width, 3), dtype=np.uint8)
        for i in range(num_width_tiles):
            empty_tile = create_empty_tile()
            sx, ex = i * tile_size, (i + 1) * tile_size
            bottom_row[:, sx:ex, :] = empty_tile
        remaining_width = new_width % tile_size
        if remaining_width > 0:
            bottom_row[:, num_width_tiles * tile_size:, :] = create_empty_tile()[:, :remaining_width, :]

        if self.carrying is not None:
            canvas = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            fill_coords(canvas, point_in_rect(0, 0.031, 0, 1), grid_color)
            fill_coords(canvas, point_in_rect(0, 1, 0, 0.031), grid_color)
            self.carrying.render(canvas)
            canvas = np.clip(canvas, 0.0, 1.0)
            item_tile = (canvas * 255).astype(np.uint8)
            bottom_row[:, -tile_size:, :] = item_tile

        return np.vstack([image_with_column, bottom_row])

    # -----------------------------
    # File helpers
    # -----------------------------
    def determine_layout_size(self) -> int:
        if self.txt_file_path:
            with open(self.txt_file_path, "r") as file:
                sections = file.read().split("\n\n")
                layout_lines = sections[0].strip().split("\n")
                h = len(layout_lines)
                w = max(len(line) for line in layout_lines)
                return max(w, h)
        else:
            return max(self.rand_gen_shape)

    def _read_file(self) -> Dict[str, Any]:
        with open(self.txt_file_path, "r") as f:
            config = json.load(f)
        assert "height_width" in config and "layers" in config, "Invalid layout file structure."
        return config

    # -----------------------------
    # Grid generation (random kept)
    # -----------------------------
    def _gen_grid(self, width: int, height: int) -> None:
        """
        Build grid by applying layers; only doors may overwrite.
        """
        self.grid = Grid(width, height)
        config = self.config
        H, W = config["height_width"]
        layers = config["layers"]

        free_width = self.display_size - W
        free_height = self.display_size - H
        if self.display_mode == "middle":
            anchor_x, anchor_y = free_width // 2, free_height // 2
        elif self.display_mode == "random":
            anchor_x = random.choice(range(max(free_width, 1))) if free_width > 0 else 0
            anchor_y = random.choice(range(max(free_height, 1))) if free_height > 0 else 0
        else:
            raise ValueError("Invalid display mode")

        image_direction = random.choice([0, 1, 2, 3]) if self.random_rotate else 0
        flip = random.choice([0, 1]) if self.random_flip else 0

        filled = set()
        key_positions = []
        goal_positions = []
        agent_positions = []
        orientation = "random"

        for layer in layers:
            obj_type = layer["obj"]
            colour = layer.get("colour", None)
            status = layer.get("status", None)
            orientation = layer.get("orientation", "random") if obj_type == "agent" else orientation
            dist = layer.get("distribution", "all")
            mat = layer["matrix"]

            raw_positions = [(x, y) for y in range(H) for x in range(W) if mat[y][x] == 1]
            candidates = []
            for x, y in raw_positions:
                xs, ys = anchor_x + x, anchor_y + y
                xr, yr = rotate_coordinate(xs, ys, image_direction, self.display_size)
                xf, yf = flip_coordinate(xr, yr, flip, self.display_size)
                candidates.append((xf, yf))

            available = candidates if obj_type == "door" else [p for p in candidates if p not in filled]

            if dist == "all":
                used = available
            elif dist == "one":
                if len(available) < 1:
                    raise ValueError(f"No available positions for layer '{layer.get('name','unknown')}'")
                used = [random.choice(available)]
            elif isinstance(dist, float):
                count = max(1, int(len(available) * dist))
                used = random.sample(available, min(count, len(available)))
            else:
                raise ValueError(f"Unknown distribution type: {dist}")

            for pos in used:
                x, y = pos
                obj = self.create_object(obj_type, colour, status)
                if obj_type == "agent":
                    agent_positions.append((x, y))
                elif obj_type == "key":
                    key_positions.append((x, y, colour))
                elif obj_type == "goal":
                    goal_positions.append((x, y))
                self.grid.set(x, y, obj)
                if obj_type != "door":
                    filled.add((x, y))

        if agent_positions:
            self.agent_pos = random.choice(agent_positions)
        else:
            raise ValueError("No available agent position")

        dir_map = {"right": 0, "down": 1, "left": 2, "up": 3, "random": random.randint(0, 3)}
        self.agent_dir = flip_direction(rotate_direction(dir_map.get(orientation, 0), image_direction), flip)
        self.carrying = None

    def create_object(self, obj_type: str, color: Optional[str], status: Optional[str]) -> Optional[WorldObj]:
        """
        Factory for world objects.
        """
        if obj_type == "wall":
            return Wall()
        elif obj_type == "floor":
            return Floor()
        elif obj_type == "key":
            return Key(color)
        elif obj_type == "ball":
            return Ball(color)
        elif obj_type == "box":
            return Box(color)
        elif obj_type == "lava":
            return Lava()
        elif obj_type == "goal":
            return Goal()
        elif obj_type == "door":
            is_open = status == "open"
            is_locked = status == "locked"
            return Door(color, is_open=is_open, is_locked=is_locked)
        elif obj_type == "agent":
            return None
        else:
            return None

    def _door_toggle_any_colour(self, door):
        # Toggle/open-locked with any key carried.
        if door.is_locked:
            if isinstance(self.carrying, Key):
                door.is_locked = False
                door.is_open = True
                return True
            return False
        door.is_open = not door.is_open
        return True

    # -----------------------------
    # Reset/Step
    # -----------------------------
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """
        If a NetworkX-backed env is provided, delegate start-state sampling to it,
        then decode that integer state into this env's internal representation.
        Otherwise, fall back to the existing (JSON-layout) native reset behavior.
        """
        # ---- NetworkX-backed branch ----
        if self.networkx_env is not None:
            # rebuild fixed layout grid to ensure decode_state applies on the right layout
            self.cumulative_reward = 0.0
            self._gen_grid(self.width, self.height)
            self.step_count = 0

            # reset backend and get starting integer state
            sp, backend_info = self.networkx_env.reset(seed=seed)
            sp = int(sp)
            self.networkx_env.current_state = sp  # keep explicit sync

            # decode that state into this env (sets internal fields) and get observation
            obs, decode_info = self.decode_state(sp)

            # optional render
            if self.render_mode == "human":
                self.render()

            # merge info
            info: Dict[str, Any] = {}
            if isinstance(backend_info, dict):
                info.update(backend_info)
            if isinstance(decode_info, dict):
                info.update(decode_info)
            return obs, info

        # ---- Original native reset branch (unchanged) ----
        self.cumulative_reward = 0.0
        self._gen_grid(self.width, self.height)

        if self.render_mode == "human":
            self.render()

        self.step_count = 0
        obs = self.gen_obs()

        # carrying/overlap fields (preserve your original behavior)
        obs["carrying"] = {"carrying": 1, "carrying_colour": 0}
        if self.carrying is not None:
            obs["carrying"] = {
                "carrying": OBJECT_TO_IDX[self.carrying.type],
                "carrying_colour": COLOR_TO_IDX[self.carrying.color],
            }
        overlap = self.grid.get(*self.agent_pos)
        obs["overlap"] = {
            "obj": OBJECT_TO_IDX[overlap.type] if overlap else 0,
            "colour": COLOR_TO_IDX[overlap.color] if overlap else 0,
        }
        return obs, {}

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        If networkx_env is provided, step via its transitions using integer codes.
        """
        if self.networkx_env is not None:
            current_encoded_state = self.encode_state()
            self.networkx_env.current_state = current_encoded_state
            next_state, reward, terminated, truncated, info = self.networkx_env.step(action)
            obs, decode_info = self.decode_state(next_state)

            self.step_count += 1
            if self.step_count >= self.max_steps:
                truncated = True

            if self.reward_config.get("sparse", False):
                self.cumulative_reward += reward
                returned_reward = self.cumulative_reward if (terminated or truncated) else 0.0
            else:
                returned_reward = reward

            if self.render_mode == "human":
                self.render()

            info.update(decode_info)
            return obs, returned_reward, terminated, truncated, info

        # native dynamics
        self.step_count += 1
        reward = self.reward_config["step_penalty"]
        terminated = False
        truncated = False

        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self.reward_config["goal_reward"]
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True
                reward = self.reward_config["lava_penalty"]
        elif action == self.actions.uni_toggle:
            if fwd_cell:
                if self.carrying is None and fwd_cell.can_pickup():
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
                    reward += self.reward_config["pickup_reward"]
                elif fwd_cell.type == "door":
                    was_open = fwd_cell.is_open
                    if self.any_key_opens_the_door:
                        self._door_toggle_any_colour(fwd_cell)
                    else:
                        fwd_cell.toggle(self, fwd_pos)
                    if fwd_cell.is_open and not was_open:
                        reward += self.reward_config["door_open_reward"]
                    elif not fwd_cell.is_open and was_open:
                        reward += self.reward_config["door_close_penalty"]
            else:
                if self.carrying:
                    self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                    self.carrying.cur_pos = fwd_pos
                    if self.carrying.type == "key":
                        reward += self.reward_config["key_drop_penalty"]
                    else:
                        reward += self.reward_config["item_drop_penalty"]
                    self.carrying = None
        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.reward_config["sparse"]:
            self.cumulative_reward += reward
            returned_reward = self.cumulative_reward if (terminated or truncated) else 0.0
        else:
            returned_reward = reward

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()
        obs["carrying"] = {"carrying": 1, "carrying_colour": 0}
        if self.carrying is not None and self.carrying != 0:
            obs["carrying"] = {"carrying": OBJECT_TO_IDX[self.carrying.type], "carrying_colour": COLOR_TO_IDX[self.carrying.color]}
        obs["overlap"] = {"obj": 0, "colour": 0}
        overlap = self.grid.get(*self.agent_pos)
        if overlap is not None:
            obs["overlap"] = {"obj": OBJECT_TO_IDX[overlap.type], "colour": COLOR_TO_IDX[overlap.color]}
        return obs, returned_reward, terminated, truncated, {}

    # -----------------------------
    # Integer encode/decode
    # -----------------------------
    def _linear_index(self, x: int, y: int, W: int) -> int:
        return int(y) * int(W) + int(x)

    def _xy_from_index(self, idx: int, W: int) -> Tuple[int, int]:
        return int(idx % W), int(idx // W)

    def _prepare_int_codec(self):
        """
        Build codec spec from config: lists of door colors, key colors, etc.
        """
        if self._int_spec is not None:
            return

        W, H = self.width, self.height
        C = W * H
        # collect by appearance order in config
        key_colors: List[str] = []
        door_colors: List[str] = []
        goal_expected = 0

        for layer in self.config["layers"]:
            obj_type = layer["obj"]
            if obj_type == "key":
                key_colors.append(layer.get("colour", "red"))
            elif obj_type == "door":
                door_colors.append(layer.get("colour", "red"))
            elif obj_type == "goal":
                goal_expected += 1

        # codec spec
        self._int_spec = {
            "W": W,
            "H": H,
            "C": C,
            "CARRIED": C,                 # sentinel index for keys
            "key_colors": key_colors,     # order fixed
            "door_colors": door_colors,   # order fixed
            "has_goal": goal_expected > 0
        }

    def _assert_fixed_layout_for_decode(self):
        """
        Integer decode requires fixed-layout (middle, no rotate/flip).
        """
        ok = (self.display_mode == "middle") and (not self.random_rotate) and (not self.random_flip)
        if not ok:
            raise RuntimeError(
                "decode_state(int) requires fixed-layout mode: display_mode='middle', random_rotate=False, random_flip=False."
            )

    def encode_state(self) -> int:
        """
        Mixed-radix integer encoding of the full dynamic state.
        Order (low -> high):
          agent_pos (C), agent_dir (4),
          for each key: key_pos (C+1) [C means carried],
          for each door: door_state (3),
          for each door: door_pos (C),
          goal_pos (C) if any.
        """
        self._prepare_int_codec()
        W, H, C = self._int_spec["W"], self._int_spec["H"], self._int_spec["C"]
        carried_idx = self._int_spec["CARRIED"]
        key_colors = self._int_spec["key_colors"]
        door_colors = self._int_spec["door_colors"]
        has_goal = self._int_spec["has_goal"]

        # agent
        ax, ay = self.agent_pos
        agent_pos = self._linear_index(ax, ay, W)
        agent_dir = int(self.agent_dir) % 4

        # keys: map color -> idx or CARRIED
        carried_color = None
        if isinstance(self.carrying, Key):
            carried_color = self.carrying.color

        def find_first(obj_cls, **kw):
            for y in range(H):
                for x in range(W):
                    obj = self.grid.get(x, y)
                    if isinstance(obj, obj_cls):
                        ok = True
                        for k, v in kw.items():
                            if getattr(obj, k) != v:
                                ok = False
                                break
                        if ok:
                            return x, y, obj
            return None

        key_positions: List[int] = []
        for col in key_colors:
            if carried_color == col:
                key_positions.append(carried_idx)
            else:
                found = find_first(Key, color=col)
                if found is None:
                    # if not on floor and not carried, treat as missing -> carried sentinel
                    key_positions.append(carried_idx)
                else:
                    x, y, _ = found
                    key_positions.append(self._linear_index(x, y, W))

        # doors: pos + state
        door_states: List[int] = []
        door_positions: List[int] = []
        for col in door_colors:
            found = find_first(Door, color=col)
            if found is None:
                # door not found -> fallback as closed at (0,0)
                door_states.append(0)
                door_positions.append(0)
            else:
                x, y, door = found
                state = 2 if door.is_locked else (1 if door.is_open else 0)
                door_states.append(state)
                door_positions.append(self._linear_index(x, y, W))

        # goal
        goal_pos = 0
        if has_goal:
            found = find_first(Goal)
            goal_pos = 0 if found is None else self._linear_index(found[0], found[1], W)

        # pack (low -> high)
        digits = []
        radices = []

        digits.append(agent_pos);  radices.append(C)
        digits.append(agent_dir);  radices.append(4)

        for kp in key_positions:
            digits.append(kp);      radices.append(C + 1)

        for ds in door_states:
            digits.append(ds);      radices.append(3)

        for dp in door_positions:
            digits.append(dp);      radices.append(C)

        if has_goal:
            digits.append(goal_pos); radices.append(C)

        # mixed radix -> int
        code = 0
        mul = 1
        for d, r in zip(digits, radices):
            if not (0 <= d < r):
                raise ValueError(f"Digit {d} out of range for radix {r}")
            code += d * mul
            mul *= r
        return int(code)

    def decode_state(self, encoded_state: int) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Decode integer state and overwrite env state.
        Requires fixed-layout mode to reconstruct static layers deterministically.
        """
        self._assert_fixed_layout_for_decode()
        self._prepare_int_codec()

        W, H, C = self._int_spec["W"], self._int_spec["H"], self._int_spec["C"]
        carried_idx = self._int_spec["CARRIED"]
        key_colors = self._int_spec["key_colors"]
        door_colors = self._int_spec["door_colors"]
        has_goal = self._int_spec["has_goal"]

        # unpack digits
        radices = []
        radices.append(C)   # agent_pos
        radices.append(4)   # agent_dir
        for _ in key_colors:
            radices.append(C + 1)  # key_pos
        for _ in door_colors:
            radices.append(3)      # door_state
        for _ in door_colors:
            radices.append(C)      # door_pos
        if has_goal:
            radices.append(C)      # goal_pos

        vals: List[int] = []
        rem = int(encoded_state)
        for r in radices:
            rem, d = divmod(rem, r)
            vals.append(d)
        # vals aligned with radices order

        idx = 0
        agent_pos = vals[idx]; idx += 1
        agent_dir = vals[idx] % 4; idx += 1

        key_positions: List[int] = []
        for _ in key_colors:
            key_positions.append(vals[idx]); idx += 1

        door_states: List[int] = []
        for _ in door_colors:
            door_states.append(vals[idx]); idx += 1

        door_positions: List[int] = []
        for _ in door_colors:
            door_positions.append(vals[idx]); idx += 1

        goal_pos = vals[idx] if has_goal else 0

        # rebuild static grid (fixed layout)
        base_grid = self._build_static_base_grid_fixed()
        self.grid = base_grid
        self.width, self.height = W, H

        # place doors
        for col, st, p in zip(door_colors, door_states, door_positions):
            x, y = self._xy_from_index(p % C, W)
            is_locked = (st == 2)
            is_open = (st == 1)
            self.grid.set(x, y, Door(col, is_open=is_open, is_locked=is_locked))

        # place goal
        if has_goal:
            xg, yg = self._xy_from_index(goal_pos % C, W)
            self.grid.set(xg, yg, Goal())

        # place keys / carrying
        carrying = None
        for col, kp in zip(key_colors, key_positions):
            if kp == carried_idx:
                carrying = Key(col) if carrying is None else carrying  # at most one carried
            else:
                xk, yk = self._xy_from_index(kp % C, W)
                self.grid.set(xk, yk, Key(col))

        # agent
        xa, ya = self._xy_from_index(agent_pos % C, W)
        self.agent_pos = (xa, ya)
        self.agent_dir = agent_dir
        self.carrying = carrying
        if self.carrying is not None:
            self.carrying.cur_pos = np.array([-1, -1])

        # counters/obs
        self.cumulative_reward = 0.0
        self.step_count = 0

        obs = self.gen_obs()
        obs["carrying"] = {"carrying": 1, "carrying_colour": 0}
        if self.carrying is not None:
            obs["carrying"] = {"carrying": OBJECT_TO_IDX[self.carrying.type], "carrying_colour": COLOR_TO_IDX[self.carrying.color]}
        obs["overlap"] = {"obj": 0, "colour": 0}
        overlap = self.grid.get(*self.agent_pos)
        if overlap is not None:
            obs["overlap"] = {"obj": OBJECT_TO_IDX[overlap.type], "colour": COLOR_TO_IDX[overlap.color]}

        return obs, {}

    def _build_static_base_grid_fixed(self) -> Grid:
        """
        Build static layers deterministically (middle anchor, no rotate/flip).
        Dynamic layers (agent/key/door/goal) are skipped here.
        """
        Wd, Hd = self.display_size, self.display_size
        g = Grid(Wd, Hd)

        config = self.config
        H, W = config["height_width"]
        layers = config["layers"]

        # fixed anchor, no transforms
        free_width = self.display_size - W
        free_height = self.display_size - H
        anchor_x, anchor_y = free_width // 2, free_height // 2

        filled = set()

        for layer in layers:
            obj_type = layer["obj"]
            dist = layer.get("distribution", "all")
            if obj_type in ["agent", "key", "door", "goal"]:
                continue  # dynamic skipped here

            mat = layer["matrix"]
            raw_positions = [(x, y) for y in range(H) for x in range(W) if mat[y][x] == 1]
            candidates = [(anchor_x + x, anchor_y + y) for (x, y) in raw_positions]
            if dist == "all":
                used = [p for p in candidates if p not in filled]
            elif dist == "one":
                if not candidates:
                    continue
                # still place one deterministically: choose first free
                cand = [p for p in candidates if p not in filled]
                used = [cand[0]] if cand else []
            elif isinstance(dist, float):
                count = max(1, int(len(candidates) * dist))
                used = candidates[:count]
            else:
                raise ValueError(f"Unknown distribution type: {dist}")

            for x, y in used:
                obj = self.create_object(obj_type, layer.get("colour", None), layer.get("status", None))
                g.set(x, y, obj)
                if obj_type != "door":
                    filled.add((x, y))

        return g

    # -----------------------------
    # Start states enumeration (kept; returns ints)
    # -----------------------------
    def get_start_states(self) -> List[int]:
        """
        Enumerate possible initial states by replaying placement logic on a temp grid.
        Works with random anchor/rot/flip; useful for analysis. Returns integer encodings.
        """
        start_states: List[int] = []

        config = self.config
        H, W = config['height_width']
        layers = config['layers']

        rotations = [0, 1, 2, 3] if self.random_rotate else [0]
        flips = [0, 1] if self.random_flip else [0]

        free_width = self.display_size - W
        free_height = self.display_size - H

        if self.display_mode == "middle":
            anchor_positions = [(free_width // 2, free_height // 2)]
        elif self.display_mode == "random":
            ax_opts = list(range(max(free_width, 1))) if free_width > 0 else [0]
            ay_opts = list(range(max(free_height, 1))) if free_height > 0 else [0]
            anchor_positions = list(itertools.product(ax_opts, ay_opts))
        else:
            anchor_positions = [(0, 0)]

        layer_possibilities = []
        agent_orientations = []

        for layer in layers:
            obj_type = layer["obj"]
            colour = layer.get("colour", None)
            status = layer.get("status", None)
            dist = layer.get("distribution", "all")
            mat = layer["matrix"]

            raw_positions = [(x, y) for y in range(H) for x in range(W) if mat[y][x] == 1]
            if not raw_positions:
                layer_possibilities.append([])
                continue

            if obj_type == "agent":
                orientation = layer.get("orientation", "random")
                if orientation == "random":
                    agent_orientations = [0, 1, 2, 3]
                else:
                    dir_map = {"right": 0, "down": 1, "left": 2, "up": 3}
                    agent_orientations = [dir_map.get(orientation, 0)]

            layer_position_sets = []
            if dist == "all":
                layer_position_sets = [raw_positions]
            elif dist == "one":
                layer_position_sets = [[pos] for pos in raw_positions]
            elif isinstance(dist, float):
                count = max(1, int(len(raw_positions) * dist))
                count = min(count, len(raw_positions))
                from itertools import combinations
                layer_position_sets = [list(combo) for combo in combinations(raw_positions, count)]

            layer_possibilities.append({
                'obj_type': obj_type,
                'colour': colour,
                'status': status,
                'position_sets': layer_position_sets
            })

        for rotation in rotations:
            for flip in flips:
                for anchor_x, anchor_y in anchor_positions:
                    combos = [layer['position_sets'] for layer in layer_possibilities if layer]
                    if not combos:
                        continue

                    for combination in itertools.product(*combos):
                        try:
                            temp_grid = Grid(self.display_size, self.display_size)
                            filled = set()
                            agent_positions = []

                            for layer_info, position_set in zip(layer_possibilities, combination):
                                if not layer_info or not position_set:
                                    continue

                                obj_type = layer_info['obj_type']
                                colour = layer_info['colour']
                                status = layer_info['status']

                                transformed = []
                                for x, y in position_set:
                                    xs, ys = anchor_x + x, anchor_y + y
                                    xr, yr = rotate_coordinate(xs, ys, rotation, self.display_size)
                                    xf, yf = flip_coordinate(xr, yr, flip, self.display_size)
                                    transformed.append((xf, yf))

                                if obj_type != "door":
                                    available = [p for p in transformed if p not in filled]
                                    if len(available) != len(transformed):
                                        break
                                    used = available
                                else:
                                    used = transformed

                                for x, y in used:
                                    if x < 0 or x >= self.display_size or y < 0 or y >= self.display_size:
                                        continue
                                    obj = self.create_object(obj_type, colour, status)
                                    if obj_type == "agent":
                                        agent_positions.append((x, y))
                                    else:
                                        temp_grid.set(x, y, obj)
                                        if obj_type != "door":
                                            filled.add((x, y))
                            else:
                                if not agent_positions:
                                    continue

                                for agent_pos in agent_positions:
                                    for agent_orientation in agent_orientations:
                                        old_grid = self.grid
                                        old_agent_pos = getattr(self, 'agent_pos', None)
                                        old_agent_dir = getattr(self, 'agent_dir', None)
                                        old_carrying = getattr(self, 'carrying', None)
                                        old_width = getattr(self, 'width', None)
                                        old_height = getattr(self, 'height', None)
                                        try:
                                            self.grid = temp_grid
                                            self.agent_pos = agent_pos
                                            self.agent_dir = flip_direction(rotate_direction(agent_orientation, rotation), flip)
                                            self.carrying = None
                                            self.width = self.display_size
                                            self.height = self.display_size
                                            state_encoding = self.encode_state()
                                            if state_encoding not in start_states:
                                                start_states.append(state_encoding)
                                        finally:
                                            self.grid = old_grid
                                            if old_agent_pos is not None:
                                                self.agent_pos = old_agent_pos
                                            if old_agent_dir is not None:
                                                self.agent_dir = old_agent_dir
                                            if old_carrying is not None:
                                                self.carrying = old_carrying
                                            if old_width is not None:
                                                self.width = old_width
                                            if old_height is not None:
                                                self.height = old_height
                        except Exception:
                            continue

        return start_states

    def get_mdp_network(self) -> MDPNetwork:
        """
        Build a deterministic MDPNetwork by enumerating reachable integer states via BFS.
        Requires fixed-layout mode to allow decode(int)->grid reconstruction.
        """
        # --- Preconditions ---
        self._assert_fixed_layout_for_decode()
        self._prepare_int_codec()

        # --- Save & patch env knobs (dense reward + native dynamics) ---
        saved_netx = self.networkx_env
        self.networkx_env = None
        saved_sparse = bool(self.reward_config.get("sparse", True))
        self.reward_config["sparse"] = False  # force immediate reward for R(s,a,s')
        saved_render_mode = getattr(self, "render_mode", None)

        try:
            from collections import deque

            print("[MDP] Building deterministic MDP via BFS...")
            num_actions = self.action_space.n
            print(f"[MDP] Actions: {num_actions}")

            # Start states (ints). If empty, fallback to one reset.
            start_states: List[int] = self.get_start_states()
            if not start_states:
                _ = self.reset()
                start_states = [self.encode_state()]
            print(f"[MDP] Start states: {len(start_states)} (show up to 8): {start_states[:8]}")

            # Helpers
            def is_terminal_current_grid() -> bool:
                ax, ay = self.agent_pos
                cell = self.grid.get(ax, ay)
                if cell is None:
                    return False
                if cell.type == "goal":
                    return True
                if cell.type == "lava":
                    return True
                return False

            # BFS
            visited: set[int] = set()
            queue: deque[int] = deque()
            for s0 in start_states:
                if s0 not in visited:
                    visited.add(s0)
                    queue.append(s0)

            transitions: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
            terminal_states: set[int] = set()

            expansions = 0
            triples_recorded = 0
            PROGRESS_EVERY = 5000  # print progress every N state expansions

            while queue:
                s = queue.popleft()
                expansions += 1

                # Decode s -> grid
                self.decode_state(s)
                if is_terminal_current_grid():
                    terminal_states.add(int(s))
                    # No outgoing transitions from terminal state
                    continue

                # For each action, take one step from s
                for a in range(num_actions):
                    # Re-decode to ensure same pre-state per action
                    self.decode_state(s)
                    _obs, r, term, trunc, _info = self.step(a)
                    sp = self.encode_state()

                    # Record transition (deterministic)
                    s_key = str(int(s))
                    a_key = str(int(a))
                    sp_key = str(int(sp))
                    transitions.setdefault(s_key, {})
                    transitions[s_key].setdefault(a_key, {})
                    if sp_key not in transitions[s_key][a_key]:
                        triples_recorded += 1
                    transitions[s_key][a_key][sp_key] = {"p": 1.0, "r": float(r)}

                    # Terminal bookkeeping on s'
                    if term:
                        terminal_states.add(int(sp))

                    # Explore s'
                    if sp not in visited:
                        visited.add(sp)
                        queue.append(sp)

                if expansions % PROGRESS_EVERY == 0:
                    print(
                        f"[MDP] Progress: expanded={expansions}, |S|={len(visited)}, queue={len(queue)}, |T|={len(terminal_states)}")

            print(f"[MDP] BFS done. |S|={len(visited)}, |T|={len(terminal_states)}, |(s,a,s')|={triples_recorded}")

            # --- Tags ---
            tags: Dict[str, List[int]] = {}

            def tag_add(name: str, sid: int):
                tags.setdefault(name, []).append(int(sid))

            key_colors = self._int_spec["key_colors"]
            door_colors = self._int_spec["door_colors"]
            has_goal = self._int_spec["has_goal"]
            print(f"[MDP] Tagging: keys={list(key_colors)}, doors={list(door_colors)}, goal={has_goal}")

            # Start/terminal tags
            for s in start_states:
                tag_add("start", s)
            for s in terminal_states:
                tag_add("terminal", s)

            # Scan every discovered state and label
            tag_scans = 0
            for s in visited:
                self.decode_state(s)
                ax, ay = self.agent_pos
                cell = self.grid.get(ax, ay)
                if cell is not None and cell.type == "goal":
                    tag_add("agent_on_goal", s)
                if cell is not None and cell.type == "lava":
                    tag_add("agent_on_lava", s)

                # keys by color
                carried_color = self.carrying.color if isinstance(self.carrying, Key) else None
                for c in key_colors:
                    if carried_color == c:
                        tag_add(f"key_{c}_carried", s)
                    on_floor = False
                    for y in range(self.height):
                        for x in range(self.width):
                            obj = self.grid.get(x, y)
                            if isinstance(obj, Key) and obj.color == c:
                                on_floor = True
                                break
                        if on_floor:
                            break
                    if on_floor:
                        tag_add(f"key_{c}_on_floor", s)

                # doors by color + state (assume <=1 per color)
                for c in door_colors:
                    door_obj = None
                    for y in range(self.height):
                        for x in range(self.width):
                            obj = self.grid.get(x, y)
                            if isinstance(obj, Door) and obj.color == c:
                                door_obj = obj
                                break
                        if door_obj is not None:
                            break
                    if door_obj is not None:
                        if door_obj.is_locked:
                            tag_add(f"door_{c}_locked", s)
                        elif door_obj.is_open:
                            tag_add(f"door_{c}_open", s)
                        else:
                            tag_add(f"door_{c}_closed", s)

                if has_goal:
                    tag_add("goal_present", s)

                tag_scans += 1
                if tag_scans % PROGRESS_EVERY == 0:
                    print(f"[MDP] Tagging progress: scanned={tag_scans}/{len(visited)}")

            print(f"[MDP] Tagging done. tag_types={len(tags)}")

            # --- Build MDP config ---
            config = {
                "num_actions": int(num_actions),
                "states": [int(x) for x in sorted(list(visited))],
                "start_states": [int(x) for x in sorted(set(start_states))],
                "terminal_states": [int(x) for x in sorted(list(terminal_states))],
                "default_reward": float(self.reward_config.get("step_penalty", -0.1)),
                "transitions": transitions,
                "tags": {k: sorted(v) for k, v in tags.items()},
            }

            print("[MDP] MDPNetwork config ready.")
            return MDPNetwork(config_data=config)

        finally:
            # restore env knobs
            self.networkx_env = saved_netx
            self.reward_config["sparse"] = saved_sparse
            if saved_render_mode is not None:
                self.render_mode = saved_render_mode
            print("[MDP] Env knobs restored.")


# -----------------------------
# Geometry helpers
# -----------------------------
def rotate_coordinate(x, y, rotation_mode, n):
    if rotation_mode == 0:
        return x, y
    elif rotation_mode == 1:
        return y, n - 1 - x
    elif rotation_mode == 2:
        return n - 1 - x, n - 1 - y
    elif rotation_mode == 3:
        return n - 1 - y, x
    else:
        raise ValueError("Invalid rotation mode. Please choose between 0, 1, 2, or 3.")


def flip_coordinate(x, y, flip_mode, n):
    if flip_mode == 0:
        return x, y
    elif flip_mode == 1:
        return n - 1 - x, y
    else:
        raise ValueError("Invalid flip mode. Please choose between 0 and 1.")


def rotate_direction(direction, rotation_mode):
    if rotation_mode in [0, 1, 2, 3]:
        return (direction + rotation_mode) % 4
    else:
        raise ValueError("Invalid rotation mode. Please choose between 0, 1, 2, or 3.")


def flip_direction(direction, flip_mode):
    if flip_mode == 0:
        return direction
    elif flip_mode == 1:
        flip_map = {0: 0, 1: 3, 2: 2, 3: 1}
        return flip_map[direction]
    else:
        raise ValueError("Invalid flip mode. Please choose between 0 and 1.")


# -----------------------------
# Small utility helpers
# -----------------------------

_DIR_GLYPHS = {0: "→", 1: "↓", 2: "←", 3: "↑"}

def _px_to_pt(px: float, dpi: int) -> float:
    return float(px) * 72.0 / float(dpi)


def _is_walkable_cell(g: Grid, x: int, y: int) -> bool:
    """
    A cell is 'walkable' if it's empty or the object allows overlap.
    """
    obj = g.get(x, y)
    return (obj is None) or bool(obj.can_overlap())


def _find_first(grid: Grid, cls, **kw) -> Optional[Tuple[int, int, WorldObj]]:
    """
    Find the first object of type `cls` (optionally matching attributes) in the grid.
    """
    for y in range(grid.height):
        for x in range(grid.width):
            obj = grid.get(x, y)
            if isinstance(obj, cls):
                ok = True
                for k, v in kw.items():
                    if getattr(obj, k) != v:
                        ok = False
                        break
                if ok:
                    return x, y, obj
    return None


def _collect_colors_from_env(env: MiniGridEnv) -> Tuple[List[str], List[str], bool]:
    """
    Collect ordered lists of key colors and door colors from the *current grid*.
    Also return whether a Goal exists.
    Order is stable by scanning order (y major then x).
    """
    key_colors: List[str] = []
    door_colors: List[str] = []
    has_goal = False
    seen_keys = set()
    seen_doors = set()
    for y in range(env.height):
        for x in range(env.width):
            obj = env.grid.get(x, y)
            if obj is None:
                continue
            if isinstance(obj, Key):
                if obj.color not in seen_keys:
                    seen_keys.add(obj.color)
                    key_colors.append(obj.color)
            elif isinstance(obj, Door):
                if obj.color not in seen_doors:
                    seen_doors.add(obj.color)
                    door_colors.append(obj.color)
            elif obj.type == "goal":
                has_goal = True
    return key_colors, door_colors, has_goal


def _layout_spec(env: MiniGridEnv) -> Dict[str, Union[int, List[str], Dict[str, int]]]:
    """
    Build a 'static' spec from the live env (positions of keys/doors; goal presence).
    This is used to assemble integer state codes without mutating env in-place.
    """
    W, H = env.width, env.height
    C = W * H
    key_colors, door_colors, has_goal = _collect_colors_from_env(env)

    # positions of keys and doors (by color) as linear index
    key_pos_idx: Dict[str, int] = {}
    door_pos_idx: Dict[str, int] = {}
    door_state_curr: Dict[str, int] = {}  # 0=closed,1=open,2=locked
    goal_idx = None

    for col in key_colors:
        found = _find_first(env.grid, Key, color=col)
        if found is not None:
            x, y, _ = found
            key_pos_idx[col] = y * W + x
        else:
            # if not on floor, treat as 'carried sentinel' in our encoding stage
            key_pos_idx[col] = C  # sentinel

    for col in door_colors:
        found = _find_first(env.grid, Door, color=col)
        if found is not None:
            x, y, door = found
            door_pos_idx[col] = y * W + x
            s = 2 if door.is_locked else (1 if door.is_open else 0)
            door_state_curr[col] = s
        else:
            door_pos_idx[col] = 0
            door_state_curr[col] = 0

    if has_goal:
        g = _find_first(env.grid, WorldObj)  # quick scan for 'goal' by type
        # re-scan to ensure correct type == 'goal'
        if g is None:
            for y in range(env.height):
                for x in range(env.width):
                    obj = env.grid.get(x, y)
                    if obj is not None and obj.type == "goal":
                        goal_idx = y * W + x
                        break
                if goal_idx is not None:
                    break
        else:
            # fallback: do a targeted scan
            goal_idx = None
            for y in range(env.height):
                for x in range(env.width):
                    obj = env.grid.get(x, y)
                    if obj is not None and obj.type == "goal":
                        goal_idx = y * W + x
                        break
                if goal_idx is not None:
                    break

    return {
        "W": W, "H": H, "C": C,
        "key_colors": key_colors,
        "door_colors": door_colors,
        "has_goal": has_goal,
        "key_pos_idx": key_pos_idx,
        "door_pos_idx": door_pos_idx,
        "door_state_curr": door_state_curr,
        "goal_idx": 0 if goal_idx is None else int(goal_idx),
        "CARRIED": C,
    }


def _encode_state_minigrid(
    spec: Dict[str, Union[int, List[str], Dict[str, int]]],
    agent_xy_idx: int,
    agent_dir: int,
    carrying_color: Optional[str],
    focus_door_color: str,
    focus_door_state: int,              # 0=closed,1=open,2=locked
) -> int:
    """
    Assemble the mixed-radix integer code consistent with CustomMiniGridEnv.encode_state.
    Uses positions from 'spec'; modifies the focused door's state and (optionally) carrying.
    """
    W = int(spec["W"])
    C = int(spec["C"])
    CARRIED = int(spec["CARRIED"])
    key_colors: List[str] = list(spec["key_colors"])  # ordered
    door_colors: List[str] = list(spec["door_colors"])
    has_goal: bool = bool(spec["has_goal"])
    key_pos_idx: Dict[str, int] = dict(spec["key_pos_idx"])      # may contain CARRIED
    door_pos_idx: Dict[str, int] = dict(spec["door_pos_idx"])
    door_state_curr: Dict[str, int] = dict(spec["door_state_curr"])
    goal_idx = int(spec["goal_idx"])

    # Carrying: push the selected key color to sentinel C if carried
    if carrying_color is not None:
        if carrying_color in key_pos_idx:
            key_pos_idx[carrying_color] = CARRIED  # carried sentinel

    # Door states: override the focus door color
    if focus_door_color in door_state_curr:
        door_state_curr[focus_door_color] = int(focus_door_state)

    digits: List[int] = []
    radices: List[int] = []

    # agent pos + dir
    digits.append(int(agent_xy_idx));  radices.append(C)
    digits.append(int(agent_dir) % 4); radices.append(4)

    # keys
    for col in key_colors:
        digits.append(int(key_pos_idx.get(col, CARRIED)))
        radices.append(C + 1)  # includes 'carried' sentinel

    # door states
    for col in door_colors:
        digits.append(int(door_state_curr.get(col, 0)))
        radices.append(3)

    # door positions
    for col in door_colors:
        digits.append(int(door_pos_idx.get(col, 0)))
        radices.append(C)

    # goal pos (if any)
    if has_goal:
        digits.append(int(goal_idx)); radices.append(C)

    # pack mixed radix
    code = 0
    mul = 1
    for d, r in zip(digits, radices):
        if not (0 <= d < r):
            raise ValueError(f"Digit {d} out of range for radix {r}")
        code += d * mul
        mul *= r
    return int(code)


def _pick_representative_xy(env: MiniGridEnv) -> Tuple[int, int]:
    """
    Choose a neutral 'walkable' tile to place the agent for facet background.
    Prefer center-ish; fall back to the first walkable tile.
    """
    cx, cy = env.width // 2, env.height // 2
    if _is_walkable_cell(env.grid, cx, cy):
        return cx, cy
    for y in range(env.height):
        for x in range(env.width):
            if _is_walkable_cell(env.grid, x, y):
                return x, y
    # If all else fails, return (0,0) and let the overlay skip non-walkables.
    return 0, 0


def _render_minigrid_bg_image(
    env: MiniGridEnv,
    door_color: str,
    key_color: str,
    agent_dir: int,
    carrying_flag: bool,
    door_state: int,   # 0=closed,1=open,2=locked
) -> np.ndarray:
    """
    Render one rgb background for a facet, consistent with the slice constraints.
    Keeps native aspect (no resizing). No fallback: raises if render fails.
    """
    spec = _layout_spec(env)
    W, H, C = int(spec["W"]), int(spec["H"]), int(spec["C"])

    ax, ay = _pick_representative_xy(env)
    s_code = _encode_state_minigrid(
        spec=spec,
        agent_xy_idx=int(ay) * W + int(ax),
        agent_dir=int(agent_dir),
        carrying_color=(key_color if carrying_flag else None),
        focus_door_color=str(door_color),
        focus_door_state=int(door_state),
    )

    # Temporarily switch to rgb_array to render
    prev_mode = getattr(env, "render_mode", None)
    env.render_mode = "rgb_array"
    try:
        env.decode_state(int(s_code))  # custom decode_state sets all internals
        img = env.render()
    finally:
        env.render_mode = prev_mode

    if img is None:
        raise RuntimeError("MiniGrid rgb_array render returned None (no fallback by design).")
    return img


def _geometry_from_bg(env: MiniGridEnv, bg_img: np.ndarray) -> Dict[str, float]:
    """
    Compute pixel geometry from background and env knobs.
    Returns inner extents and tile size.
    """
    tile = int(getattr(env, "tile_size", 32))
    W_tiles, H_tiles = env.width, env.height
    # background includes one extra right column and one bottom row in CustomMiniGridEnv.get_frame()
    W_bg = W_tiles * tile + tile
    H_bg = H_tiles * tile + tile
    H_img, W_img = bg_img.shape[:2]
    # Sanity align: if mismatch, trust image size to derive tile
    if (W_img != W_bg) or (H_img != H_bg):
        # Derive tile size from image (assumes the same "+1 tile" frame logic)
        tile_x = W_img // (W_tiles + 1)
        tile_y = H_img // (H_tiles + 1)
        tile = int(round((tile_x + tile_y) * 0.5))
        W_bg = W_img
        H_bg = H_img
    # Inner (playable) extents in pixel coords
    return {
        "tile": float(tile),
        "W_bg": float(W_bg),
        "H_bg": float(H_bg),
        "W_in": float(W_tiles * tile),
        "H_in": float(H_tiles * tile),
    }


def _cell_center_px(col: int, row: int, tile: float) -> Tuple[float, float]:
    """
    Pixel center for a grid cell (origin at top-left, y downward).
    """
    return ( (col + 0.5) * tile, (row + 0.5) * tile )


def _draw_compass_badge(ax, x: float, y: float, new_dir: int, font_pt: float, alpha: float = 0.85):
    """
    Small compass glyph next to a self-loop to denote an out-of-slice orientation change.
    """
    ax.text(
        x, y, _DIR_GLYPHS.get(int(new_dir), "?"),
        ha="center", va="center", fontsize=font_pt,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.12"),
        zorder=6,
    )


# -----------------------------
# Transition overlays (paged 2×3 facets)
# -----------------------------

def plot_minigrid_transition_overlays(
    env: MiniGridEnv,
    mdp: Optional[MDPNetwork],
    output_dir: str,
    filename_prefix: str = "minigrid_transitions",
    # paging controls
    door_colors: Optional[List[str]] = None,
    key_colors: Optional[List[str]] = None,
    agent_dirs: Optional[List[int]] = None,
    # viz controls
    min_prob: float = 0.05,
    alpha: float = 0.90,
    annotate: bool = True,
    dpi: int = 200,
    arrow_scale: float = 0.035,
    font_scale: float = 0.14,
    cmap_name: str = "viridis",
    gamma: float = 1.0,
) -> List[str]:
    """
    For each (door_color, key_color, agent_dir) page, draw a 2×3 mosaic of transitions
    for actions LEFT / RIGHT / FORWARD (one page per action, like Taxi), or keep the same
    mosaic layout but draw three figures (one per action). To mirror the Taxi API closely,
    we produce a *single* 2×2 mosaic per action in Taxi; here the facet grid is 2×3, so we
    will produce *one image per action* per page triple to keep titles short and legible.

    Returns list of saved file absolute paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # default paging from current env
    spec = _layout_spec(env)
    all_door_colors = door_colors if door_colors is not None else list(spec["door_colors"])
    all_key_colors  = key_colors  if key_colors  is not None else list(spec["key_colors"])
    if not all_key_colors:
        # Ensure at least one key color exists to define the 'carrying' row
        all_key_colors = ["red"]
    all_agent_dirs = agent_dirs if agent_dirs is not None else [0, 1, 2, 3]

    # action ids (ignore any uni_toggle)
    A_LEFT    = int(env.actions.left)
    A_RIGHT   = int(env.actions.right)
    A_FORWARD = int(env.actions.forward)
    actions = [(A_LEFT,  "LEFT"), (A_RIGHT, "RIGHT"), (A_FORWARD, "FORWARD")]

    out_files: List[str] = []

    # Shared colormap for probabilities
    cmap = cm.get_cmap(cmap_name)
    prob_norm = mcolors.PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)

    def prob_to_color(p: float):
        return cmap(prob_norm(np.clip(p, 0.0, 1.0)))

    # Arrow styles (computed later from tile px)
    def draw_self_loop(ax, x, y, tile, color, alpha, lw_pt, ms_pt):
        radius = 0.38 * tile
        arc = Arc((x + 0.28 * radius, y - 0.28 * radius),
                  width=radius, height=radius,
                  angle=0, theta1=30, theta2=320,
                  linewidth=lw_pt, color=color, alpha=alpha, zorder=4)
        ax.add_patch(arc)
        arr = FancyArrowPatch(
            (x + 0.68 * radius, y - 0.45 * radius),
            (x + 0.52 * radius, y - 0.35 * radius),
            arrowstyle="->", mutation_scale=ms_pt,
            linewidth=lw_pt, facecolor=color, edgecolor=color,
            alpha=alpha, zorder=5, shrinkA=0.0, shrinkB=0.0
        )
        ax.add_patch(arr)

    # Iterate pages
    for dcol in all_door_colors:
        for kcol in all_key_colors:
            for ddir in all_agent_dirs:

                # Pre-render all six facet backgrounds to size the figure
                # Facet rows: 0 => carrying=False, 1 => carrying=True
                # Facet cols: 0,1,2 => door_state = closed(0)/open(1)/locked(2)
                facet_bgs: Dict[Tuple[int,int], np.ndarray] = {}
                for r_carry in (0, 1):
                    for c_state in (0, 1, 2):
                        bg = _render_minigrid_bg_image(
                            env, door_color=dcol, key_color=kcol,
                            agent_dir=ddir, carrying_flag=bool(r_carry),
                            door_state=c_state
                        )
                        facet_bgs[(r_carry, c_state)] = bg

                # Geometry (use first bg to set figure size)
                sample_bg = facet_bgs[(0, 0)]
                geom = _geometry_from_bg(env, sample_bg)
                tile = float(geom["tile"])
                W_bg, H_bg = float(geom["W_bg"]), float(geom["H_bg"])
                W_in, H_in = float(geom["W_in"]), float(geom["H_in"])

                # Typography scaled from tile
                cell_min = tile
                ARROW_LW_PT = _px_to_pt(max(1.0, arrow_scale * cell_min), dpi)
                mutation_scale = _px_to_pt(0.42 * cell_min, dpi)
                shrink_pt = _px_to_pt(0.16 * cell_min, dpi)
                font_pt  = max(5.0, min(12.0, _px_to_pt(font_scale * cell_min, dpi)))
                title_pt = max(9.0,  min(14.0, _px_to_pt(0.16 * cell_min, dpi)))
                text_bbox = dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.12")
                text_effects = [pe.withStroke(linewidth=_px_to_pt(0.8, dpi), foreground="black", alpha=0.30)]

                # Inner extent helper
                def inner_extent():
                    return [0.0, W_in, H_in, 0.0]

                # Create one figure *per action* to keep clutter reasonable
                for a_id, a_name in actions:
                    # 2x3 = 6 subplots
                    fig = plt.figure(figsize=((W_bg / dpi) * 3, (H_bg / dpi) * 2), dpi=dpi)
                    axes = [plt.subplot(2, 3, i) for i in (1, 2, 3, 4, 5, 6)]

                    facet_list = [
                        (0, 0), (0, 1), (0, 2),  # row 0 carrying=False
                        (1, 0), (1, 1), (1, 2),  # row 1 carrying=True
                    ]

                    for ax, (r_carry, c_state) in zip(axes, facet_list):
                        bg = facet_bgs[(r_carry, c_state)]
                        ax.imshow(bg, origin="upper", extent=[0, W_bg, H_bg, 0], zorder=0)
                        ax.set_xlim(0, W_bg); ax.set_ylim(H_bg, 0)
                        ax.set_xticks([]); ax.set_yticks([])
                        ax.set_title(
                            f"{a_name} · Door={dcol}({['closed','open','locked'][c_state]}) · "
                            f"Carry={'None' if r_carry==0 else kcol} · Dir={_DIR_GLYPHS.get(ddir,'?')}",
                            fontsize=title_pt
                        )

                        # Overlay arrows inside the inner playable rectangle only
                        ax.imshow(np.zeros((env.height, env.width)), origin="upper",
                                  extent=inner_extent(), cmap="Greys",
                                  alpha=0.0, interpolation="nearest", zorder=1)

                        # Iterate all positions (skip non-walkables)
                        for ry in range(env.height):
                            for cx in range(env.width):
                                if not _is_walkable_cell(env.grid, cx, ry):
                                    continue

                                # Representative state for this cell under the facet/page slice
                                spec_local = _layout_spec(env)
                                s_rep = _encode_state_minigrid(
                                    spec=spec_local,
                                    agent_xy_idx=int(ry) * env.width + int(cx),
                                    agent_dir=int(ddir),
                                    carrying_color=(kcol if r_carry == 1 else None),
                                    focus_door_color=dcol,
                                    focus_door_state=c_state
                                )

                                # Query transition probabilities from mdp if provided; else simulate once
                                probs: Dict[int, float] = {}
                                if mdp is not None and hasattr(mdp, "get_transition_probabilities"):
                                    try:
                                        probs = mdp.get_transition_probabilities(int(s_rep), int(a_id))
                                    except Exception:
                                        probs = {}
                                if not probs:
                                    # Fallback sim (deterministic). This does not alter render mode.
                                    prev_mode = getattr(env, "render_mode", None)
                                    try:
                                        env.decode_state(int(s_rep))
                                        _obs, _r, _t, _tr, _info = env.step(int(a_id))
                                        sp = env.encode_state()
                                        probs = {int(sp): 1.0}
                                    finally:
                                        env.render_mode = prev_mode

                                x0, y0 = _cell_center_px(cx, ry, tile)

                                for sp_code, p in probs.items():
                                    p = float(p)
                                    if p < min_prob:
                                        continue
                                    color = prob_to_color(p)

                                    # Decode s' just enough to read (x,y,dir)
                                    try:
                                        env.decode_state(int(sp_code))
                                        nx, ny = env.agent_pos
                                        ndir = int(env.agent_dir)
                                    except Exception:
                                        continue

                                    # Out-of-slice handling:
                                    #   - For LEFT/RIGHT: direction changes, so target lies on another page.
                                    #     Draw a *dashed self-loop* and a small compass glyph for new_dir.
                                    #   - For FORWARD: draw a straight arrow (may be a self-loop if blocked).
                                    if a_id in (A_LEFT, A_RIGHT):
                                        # dashed self-loop
                                        draw_self_loop(ax, x0, y0, tile, color, alpha, ARROW_LW_PT, mutation_scale)
                                        # overwrite line style by adding a faint straight short dash (visual hint)
                                        ax.plot([x0 - 0.18 * tile, x0 + 0.18 * tile],
                                                [y0 + 0.18 * tile, y0 + 0.18 * tile],
                                                linestyle="--", linewidth=ARROW_LW_PT * 0.6,
                                                alpha=alpha * 0.7, color=color, zorder=4)
                                        # compass badge
                                        if annotate:
                                            _draw_compass_badge(ax, x0, y0 - 0.40 * tile, ndir, font_pt, alpha=0.9)
                                            ax.text(x0, y0 + 0.36 * tile, f"{p:.2f}",
                                                    ha="center", va="center", fontsize=font_pt,
                                                    bbox=text_bbox, alpha=alpha, zorder=6,
                                                    path_effects=text_effects)
                                        continue

                                    # FORWARD arrow (possibly to self if blocked)
                                    x1, y1 = _cell_center_px(int(nx), int(ny), tile)
                                    if (nx == cx) and (ny == ry):
                                        draw_self_loop(ax, x0, y0, tile, color, alpha, ARROW_LW_PT, mutation_scale)
                                        if annotate:
                                            ax.text(x0, y0 - 0.33 * tile, f"{p:.2f}",
                                                    ha="center", va="center", fontsize=font_pt,
                                                    bbox=text_bbox, alpha=alpha, zorder=6,
                                                    path_effects=text_effects)
                                        continue

                                    arr = FancyArrowPatch(
                                        (x0, y0), (x1, y1),
                                        arrowstyle="->", mutation_scale=mutation_scale,
                                        linewidth=ARROW_LW_PT, facecolor=color, edgecolor=color,
                                        alpha=alpha, zorder=5, shrinkA=shrink_pt, shrinkB=shrink_pt
                                    )
                                    ax.add_patch(arr)
                                    if annotate:
                                        mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
                                        ax.text(mx, my, f"{p:.2f}",
                                                ha="center", va="center", fontsize=font_pt,
                                                bbox=text_bbox, alpha=alpha, zorder=6,
                                                path_effects=text_effects)

                    # Shared colorbar
                    sm = cm.ScalarMappable(cmap=cmap, norm=prob_norm); sm.set_array([])
                    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.02)
                    cbar.set_label("Transition probability", fontsize=max(7, int(title_pt * 0.6)))
                    cbar.ax.tick_params(labelsize=max(6, int(font_pt * 0.9)))

                    # Save
                    fname = f"{filename_prefix}__door-{dcol}__key-{kcol}__dir-{ddir}__act-{a_name.lower()}.png"
                    out_path = os.path.join(output_dir, fname)
                    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=dpi)
                    plt.close(fig)
                    print(f"[OK] Saved MiniGrid transitions page: {os.path.abspath(out_path)}")
                    out_files.append(os.path.abspath(out_path))

    return out_files


# -----------------------------
# Scalar overlay (paged 2×3 facets)
# -----------------------------

def plot_minigrid_scalar_overlay(
    env: MiniGridEnv,
    value_map: "ValueTable",                 # supports .get_value(s) or dict-like
    output_dir: str,
    filename_prefix: str = "minigrid_scalar_facets",
    # paging
    door_colors: Optional[List[str]] = None,
    key_colors: Optional[List[str]] = None,
    agent_dirs: Optional[List[int]] = None,
    # style
    alpha: float = 0.65,
    annotate: bool = True,
    dpi: int = 200,
    font_scale: float = 0.15,
    cmap_name: str = "magma",
    gamma: float = 1.0,
    min_abs_label: float = 0.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "State Scalar",
    cbar_label: str = "Value",
    value_format: Optional[str] = None,      # e.g. ".2f" or ".2e"
) -> List[str]:
    """
    For each (door_color, key_color, agent_dir) page, render a 2×3 faceted heatmap of a per-state scalar.
    Each facet fixes the carrying flag (row) and the door state of the focus door (col).
    """
    os.makedirs(output_dir, exist_ok=True)

    def get_val(tbl, s: int) -> float:
        if hasattr(tbl, "get_value"):
            return float(tbl.get_value(int(s)))
        try:
            return float(tbl.get(int(s), 0.0))
        except Exception:
            return 0.0

    out_files: List[str] = []

    spec0 = _layout_spec(env)
    all_door_colors = door_colors if door_colors is not None else list(spec0["door_colors"])
    all_key_colors  = key_colors  if key_colors  is not None else list(spec0["key_colors"]) or ["red"]
    all_agent_dirs = agent_dirs if agent_dirs is not None else [0, 1, 2, 3]

    cmap = cm.get_cmap(cmap_name)
    norm_dummy = mcolors.PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)  # placeholder

    for dcol in all_door_colors:
        for kcol in all_key_colors:
            for ddir in all_agent_dirs:

                # Prepare facet data & backgrounds
                facet_grids: Dict[Tuple[int,int], np.ndarray] = {}
                data_min, data_max = +np.inf, -np.inf
                facet_bgs: Dict[Tuple[int,int], np.ndarray] = {}

                for r_carry in (0, 1):
                    for c_state in (0, 1, 2):
                        # background for geometry & figure sizing
                        bg = _render_minigrid_bg_image(
                            env, door_color=dcol, key_color=kcol,
                            agent_dir=ddir, carrying_flag=bool(r_carry),
                            door_state=c_state
                        )
                        facet_bgs[(r_carry, c_state)] = bg

                        # scalar grid
                        G = np.full((env.height, env.width), np.nan, dtype=float)
                        for ry in range(env.height):
                            for cx in range(env.width):
                                if not _is_walkable_cell(env.grid, cx, ry):
                                    continue
                                s_code = _encode_state_minigrid(
                                    spec=_layout_spec(env),
                                    agent_xy_idx=int(ry) * env.width + int(cx),
                                    agent_dir=int(ddir),
                                    carrying_color=(kcol if r_carry == 1 else None),
                                    focus_door_color=dcol,
                                    focus_door_state=c_state
                                )
                                G[ry, cx] = get_val(value_map, int(s_code))
                        facet_grids[(r_carry, c_state)] = G
                        if np.isfinite(G).any():
                            data_min = min(data_min, float(np.nanmin(G)))
                            data_max = max(data_max, float(np.nanmax(G)))

                # Color scale (shared within page)
                if not np.isfinite(data_min): data_min = 0.0
                if not np.isfinite(data_max): data_max = 1.0
                _vmin = data_min if vmin is None else float(vmin)
                _vmax = data_max if vmax is None else float(vmax)
                if _vmax <= _vmin: _vmax = _vmin + 1e-9
                norm = mcolors.PowerNorm(gamma=gamma, vmin=_vmin, vmax=_vmax)

                # Geometry
                sample_bg = facet_bgs[(0, 0)]
                geom = _geometry_from_bg(env, sample_bg)
                tile = float(geom["tile"])
                W_bg, H_bg = float(geom["W_bg"]), float(geom["H_bg"])
                W_in, H_in = float(geom["W_in"]), float(geom["H_in"])

                # Typography
                cell_min = tile
                font_pt  = max(5.0, min(12.0, _px_to_pt(font_scale * cell_min, dpi)))
                title_pt = max(9.0,  min(14.0, _px_to_pt(0.16 * cell_min, dpi)))
                text_bbox = dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.12")
                text_effects = [pe.withStroke(linewidth=_px_to_pt(0.8, dpi), foreground="black", alpha=0.3)]

                def fmt_val(v: float) -> str:
                    if value_format is not None:
                        return format(v, value_format)
                    return (f"{v:.2e}" if abs(v) < 0.01 and v != 0.0 else f"{v:.2f}")

                def inner_extent():
                    return [0.0, W_in, H_in, 0.0]

                def cell_center(c: int, r: int) -> Tuple[float, float]:
                    return _cell_center_px(c, r, tile)

                # Figure (2x3)
                fig = plt.figure(figsize=((W_bg / dpi) * 3, (H_bg / dpi) * 2), dpi=dpi)
                axes = [plt.subplot(2, 3, i) for i in (1, 2, 3, 4, 5, 6)]
                facet_list = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]

                for ax, (r_carry, c_state) in zip(axes, facet_list):
                    bg = facet_bgs[(r_carry, c_state)]
                    ax.imshow(bg, origin="upper", extent=[0, W_bg, H_bg, 0], zorder=0)
                    ax.set_xlim(0, W_bg); ax.set_ylim(H_bg, 0)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(
                        f"{title} · Door={dcol}({['closed','open','locked'][c_state]}) · "
                        f"Carry={'None' if r_carry==0 else kcol} · Dir={_DIR_GLYPHS.get(ddir,'?')}",
                        fontsize=title_pt
                    )

                    G = facet_grids[(r_carry, c_state)]
                    ax.imshow(G, origin="upper", cmap=cmap, norm=norm,
                              extent=inner_extent(), alpha=alpha, zorder=1, interpolation="nearest")

                    if annotate:
                        for r in range(env.height):
                            for c in range(env.width):
                                v = float(G[r, c])
                                if not np.isfinite(v) or (abs(v) < min_abs_label):
                                    continue
                                x, y = cell_center(c, r)
                                ax.text(x, y, fmt_val(v), ha="center", va="center", fontsize=font_pt,
                                        bbox=text_bbox, alpha=0.95, zorder=2, path_effects=text_effects)

                sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
                cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.02)
                cbar.set_label(cbar_label, fontsize=max(7, int(title_pt * 0.6)))
                cbar.ax.tick_params(labelsize=max(6, int(font_pt * 0.9)))

                fname = f"{filename_prefix}__door-{dcol}__key-{kcol}__dir-{ddir}.png"
                out_path = os.path.join(output_dir, fname)
                plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=dpi)
                plt.close(fig)
                print(f"[OK] Saved MiniGrid scalar facets: {os.path.abspath(out_path)}")
                out_files.append(os.path.abspath(out_path))
    return out_files


# -----------------------------
# Scalar *difference* overlay (paged 2×3 facets)
# -----------------------------

def plot_minigrid_scalar_diff_overlay(
    env: MiniGridEnv,
    values_a: "ValueTable",
    values_b: "ValueTable",
    output_dir: str,
    filename_prefix: str = "minigrid_scalar_diff_facets",
    # paging
    door_colors: Optional[List[str]] = None,
    key_colors: Optional[List[str]] = None,
    agent_dirs: Optional[List[int]] = None,
    # style
    alpha: float = 0.65,
    annotate: bool = True,
    dpi: int = 200,
    font_scale: float = 0.15,
    cmap_name: str = "coolwarm",
    min_abs_label: float = 0.0,
    vmin: Optional[float] = None,     # if None -> symmetric by max |Δ|
    vmax: Optional[float] = None,
    title: str = "Δ State Value (A − B)",
    cbar_label: str = "Δ value (A − B)",
    value_format: Optional[str] = "+.2f",
) -> List[str]:
    """
    As above, but renders (A − B) with a diverging colormap and 0-centered normalization.
    """
    os.makedirs(output_dir, exist_ok=True)

    def get_val(tbl, s: int) -> float:
        if hasattr(tbl, "get_value"):
            return float(tbl.get_value(int(s)))
        try:
            return float(tbl.get(int(s), 0.0))
        except Exception:
            return 0.0

    out_files: List[str] = []

    spec0 = _layout_spec(env)
    all_door_colors = door_colors if door_colors is not None else list(spec0["door_colors"])
    all_key_colors  = key_colors  if key_colors  is not None else list(spec0["key_colors"]) or ["red"]
    all_agent_dirs = agent_dirs if agent_dirs is not None else [0, 1, 2, 3]

    cmap = cm.get_cmap(cmap_name)

    for dcol in all_door_colors:
        for kcol in all_key_colors:
            for ddir in all_agent_dirs:

                facet_grids: Dict[Tuple[int,int], np.ndarray] = {}
                max_abs = 0.0
                facet_bgs: Dict[Tuple[int,int], np.ndarray] = {}

                for r_carry in (0, 1):
                    for c_state in (0, 1, 2):
                        bg = _render_minigrid_bg_image(
                            env, door_color=dcol, key_color=kcol,
                            agent_dir=ddir, carrying_flag=bool(r_carry),
                            door_state=c_state
                        )
                        facet_bgs[(r_carry, c_state)] = bg

                        G = np.full((env.height, env.width), np.nan, dtype=float)
                        for ry in range(env.height):
                            for cx in range(env.width):
                                if not _is_walkable_cell(env.grid, cx, ry):
                                    continue
                                s_code = _encode_state_minigrid(
                                    spec=_layout_spec(env),
                                    agent_xy_idx=int(ry) * env.width + int(cx),
                                    agent_dir=int(ddir),
                                    carrying_color=(kcol if r_carry == 1 else None),
                                    focus_door_color=dcol,
                                    focus_door_state=c_state
                                )
                                dv = get_val(values_a, int(s_code)) - get_val(values_b, int(s_code))
                                G[ry, cx] = dv
                        facet_grids[(r_carry, c_state)] = G
                        if np.isfinite(G).any():
                            max_abs = max(max_abs, float(np.nanmax(np.abs(G))))

                _vmin, _vmax = (vmin, vmax)
                if (_vmin is None) or (_vmax is None):
                    _vmin, _vmax = -max_abs, max_abs
                if not (_vmin < 0.0 < _vmax):
                    if _vmax <= 0.0 and _vmin < 0.0:
                        _vmax = abs(_vmin)
                    elif _vmin >= 0.0 and _vmax > 0.0:
                        _vmin = -_vmax
                    else:
                        _vmin, _vmax = -1e-9, 1e-9

                try:
                    norm = mcolors.TwoSlopeNorm(vmin=_vmin, vcenter=0.0, vmax=_vmax)
                except Exception:
                    norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)

                # Geometry
                sample_bg = facet_bgs[(0, 0)]
                geom = _geometry_from_bg(env, sample_bg)
                tile = float(geom["tile"])
                W_bg, H_bg = float(geom["W_bg"]), float(geom["H_bg"])
                W_in, H_in = float(geom["W_in"]), float(geom["H_in"])

                # Typography
                cell_min = tile
                font_pt  = max(5.0, min(12.0, _px_to_pt(font_scale * cell_min, dpi)))
                title_pt = max(9.0,  min(14.0, _px_to_pt(0.16 * cell_min, dpi)))
                text_bbox = dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.12")
                text_effects = [pe.withStroke(linewidth=_px_to_pt(0.8, dpi), foreground="black", alpha=0.3)]

                def fmt_val(v: float) -> str:
                    if value_format is not None:
                        return format(v, value_format)
                    return (f"{v:+.2e}" if abs(v) < 0.01 and v != 0.0 else f"{v:+.2f}")

                def inner_extent():
                    return [0.0, W_in, H_in, 0.0]

                def cell_center(c: int, r: int) -> Tuple[float, float]:
                    return _cell_center_px(c, r, tile)

                # Figure (2x3)
                fig = plt.figure(figsize=((W_bg / dpi) * 3, (H_bg / dpi) * 2), dpi=dpi)
                axes = [plt.subplot(2, 3, i) for i in (1, 2, 3, 4, 5, 6)]
                facet_list = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]

                for ax, (r_carry, c_state) in zip(axes, facet_list):
                    bg = facet_bgs[(r_carry, c_state)]
                    ax.imshow(bg, origin="upper", extent=[0, W_bg, H_bg, 0], zorder=0)
                    ax.set_xlim(0, W_bg); ax.set_ylim(H_bg, 0)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(
                        f"{title} · Door={dcol}({['closed','open','locked'][c_state]}) · "
                        f"Carry={'None' if r_carry==0 else kcol} · Dir={_DIR_GLYPHS.get(ddir,'?')}",
                        fontsize=title_pt
                    )

                    G = facet_grids[(r_carry, c_state)]
                    ax.imshow(G, origin="upper", cmap=cmap, norm=norm,
                              extent=inner_extent(), alpha=alpha, zorder=1, interpolation="nearest")

                    if annotate:
                        for r in range(env.height):
                            for c in range(env.width):
                                v = float(G[r, c])
                                if not np.isfinite(v) or (abs(v) < min_abs_label):
                                    continue
                                x, y = cell_center(c, r)
                                ax.text(x, y, fmt_val(v), ha="center", va="center", fontsize=font_pt,
                                        bbox=text_bbox, alpha=0.95, zorder=2, path_effects=text_effects)

                sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
                cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.02)
                cbar.set_label(cbar_label, fontsize=max(7, int(title_pt * 0.6)))
                cbar.ax.tick_params(labelsize=max(6, int(font_pt * 0.9)))

                fname = f"{filename_prefix}__door-{dcol}__key-{kcol}__dir-{ddir}.png"
                out_path = os.path.join(output_dir, fname)
                plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15, dpi=dpi)
                plt.close(fig)
                print(f"[OK] Saved MiniGrid scalar diff facets: {os.path.abspath(out_path)}")
                out_files.append(os.path.abspath(out_path))
    return out_files


if __name__ == "__main__":
    env = CustomMiniGridEnv(
        map_name="door-key-fixed",
        config=None,
        display_size=None,
        display_mode="random",
        random_rotate=True,
        random_flip=True,
        render_carried_objs=True,
        render_mode="human",
    )
    manual_control = SimpleManualControl(env)
    manual_control.start()
