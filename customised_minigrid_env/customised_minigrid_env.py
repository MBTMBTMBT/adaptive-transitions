import json
import random
from typing import Optional, Dict, Any, SupportsFloat, List
import itertools

from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import *
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, TILE_PIXELS

from .simple_actions import SimpleActions
from .simple_manual_control import SimpleManualControl
from customisable_env_abs import CustomisableEnvAbs

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
    A custom MiniGrid environment that loads its layout from a structured JSON configuration.
    This environment does not support random generation. All objects, colors, positions, and agent settings
    must be defined either in a JSON file or passed as a Python dictionary.
    Later layers overwrite earlier ones. Only 'door' objects are allowed to overwrite others (e.g., walls).
    Attributes:
        json_file_path (Optional[str]): Path to the JSON layout file.
        config (Optional[dict]): Parsed layout configuration dictionary.
        display_size (int): Size of the visualized grid (can be larger than the layout).
        display_mode (str): "middle" or "random" placement of layout in the full grid.
        random_rotate (bool): Whether to randomly rotate the layout (0°, 90°, 180°, 270°).
        random_flip (bool): Whether to randomly flip the layout horizontally.
        mission (str): Text description of the agent's task.
        render_carried_objs (bool): If True, renders the carried object in a separate visual tile.
        any_key_opens_the_door (bool): If True, any key can open any door regardless of color.
        reward_config (dict): Dictionary containing reward settings and values.
    """

    def __init__(
            self,
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
        """
        Initialize the environment from either a file or a config dictionary.
        Args:
            reward_config: Dictionary containing reward configuration. If None, uses default values.
                          Available keys:
                          - sparse (bool): If True, only return total reward at termination
                          - step_penalty (float): Penalty for each step taken
                          - goal_reward (float): Reward for reaching the goal
                          - lava_penalty (float): Penalty for stepping on lava
                          - pickup_reward (float): Reward for picking up an item
                          - door_open_reward (float): Reward for opening a door
                          - door_close_penalty (float): Penalty for closing a door
                          - key_drop_penalty (float): Penalty for dropping a key
                          - item_drop_penalty (float): Penalty for dropping an item
        """

        # Enforce exclusive choice: exactly one of the two must be provided
        assert (json_file_path is not None) ^ (config is not None), \
            "You must provide either 'json_file_path' or 'config', but not both."

        self.txt_file_path = json_file_path
        self.display_mode = display_mode
        self.random_rotate = random_rotate
        self.random_flip = random_flip
        self.render_carried_objs = render_carried_objs
        self.any_key_opens_the_door = any_key_opens_the_door
        self.mission = custom_mission
        self.tile_size = 32
        self.skip_reset = False

        if reward_config is None:
            self.reward_config = DEFAULT_REWARD_DICT
        else:
            # Merge provided config with defaults
            self.reward_config = {**DEFAULT_REWARD_DICT, **reward_config}

        # Initialize cumulative reward for sparse mode
        self.cumulative_reward = 0.0

        # Load config from file if needed
        if json_file_path is not None:
            with open(json_file_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config

        # Determine layout size
        height, width = self.config["height_width"]
        layout_size = max(height, width)

        if display_size is None:
            self.display_size = layout_size
        else:
            self.display_size = display_size

        assert display_mode in ["middle", "random"], "Invalid display_mode"
        assert self.display_size >= layout_size, "display_size must be >= layout layout_size"

        # Initialize parent classes
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: custom_mission),
            grid_size=self.display_size,
            max_steps=max_steps,
            **kwargs,
        )

        # Initialize CustomisableEnvAbs
        CustomisableEnvAbs.__init__(self, networkx_env=networkx_env)

        self.actions = SimpleActions
        self.action_space = spaces.Discrete(len(self.actions))

        self.step_count = 0

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        frame = super().get_frame(highlight, tile_size, agent_pov)
        if not self.render_carried_objs:
            return frame
        else:
            return self.render_with_carried_objects(frame)

    def render_with_carried_objects(self, full_image):
        """
        Renders the image of the environment with an extra row at the bottom and an extra column on the right,
        displaying the item carried by the agent in the bottom-right corner, if any.
        The agent can carry at most one item.
        :param full_image: The original image rendered by get_full_render.
        :return: Modified image with additional row and column displaying the carried item, if any.
        """
        tile_size = self.tile_size

        # Get original image dimensions
        full_height, full_width, _ = full_image.shape

        # Grid line color in float32 format [0,1] - equivalent to (100, 100, 100) in uint8
        grid_color = (100 / 255, 100 / 255, 100 / 255)

        def create_empty_tile():
            """Helper function to create an empty tile with grid lines"""
            canvas = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            fill_coords(canvas, point_in_rect(0, 0.031, 0, 1), grid_color)  # Left border
            fill_coords(canvas, point_in_rect(0, 1, 0, 0.031), grid_color)  # Top border
            canvas = np.clip(canvas, 0.0, 1.0)
            return (canvas * 255).astype(np.uint8)

        # Step 1: Add a column on the right
        # Calculate how many tiles we need for the height
        num_height_tiles = full_height // tile_size

        # Create the right column filled with empty tiles
        right_column = np.zeros((full_height, tile_size, 3), dtype=np.uint8)

        for i in range(num_height_tiles):
            empty_tile = create_empty_tile()
            start_y = i * tile_size
            end_y = start_y + tile_size
            right_column[start_y:end_y, :, :] = empty_tile

        # Handle any remaining pixels in height
        remaining_height = full_height % tile_size
        if remaining_height > 0:
            empty_tile = create_empty_tile()
            start_y = num_height_tiles * tile_size
            right_column[start_y:, :, :] = empty_tile[:remaining_height, :, :]

        # Combine original image with right column
        image_with_column = np.hstack([full_image, right_column])

        # Step 2: Add a row at the bottom
        new_width = image_with_column.shape[1]
        num_width_tiles = new_width // tile_size

        # Create the bottom row filled with empty tiles
        bottom_row = np.zeros((tile_size, new_width, 3), dtype=np.uint8)

        for i in range(num_width_tiles):
            empty_tile = create_empty_tile()
            start_x = i * tile_size
            end_x = start_x + tile_size
            bottom_row[:, start_x:end_x, :] = empty_tile

        # Handle any remaining pixels in width
        remaining_width = new_width % tile_size
        if remaining_width > 0:
            empty_tile = create_empty_tile()
            start_x = num_width_tiles * tile_size
            bottom_row[:, start_x:, :] = empty_tile[:, :remaining_width, :]

        # Step 3: If the agent is carrying an object, render it in the bottom-right corner
        if self.carrying is not None:
            # Create a canvas for rendering the carried object
            canvas = np.zeros((tile_size, tile_size, 3), dtype=np.float32)

            # First render the empty cell background
            fill_coords(canvas, point_in_rect(0, 0.031, 0, 1), grid_color)  # Left border
            fill_coords(canvas, point_in_rect(0, 1, 0, 0.031), grid_color)  # Top border

            # Then render the actual object on top
            self.carrying.render(canvas)

            # Convert from float32 [0,1] to uint8 [0,255]
            canvas = np.clip(canvas, 0.0, 1.0)
            item_tile = (canvas * 255).astype(np.uint8)

            # Place the item tile in the bottom-right corner of the bottom row
            bottom_row[:, -tile_size:, :] = item_tile

        # Step 4: Combine everything
        output_image = np.vstack([image_with_column, bottom_row])

        return output_image

    def determine_layout_size(self) -> int:
        if self.txt_file_path:
            with open(self.txt_file_path, 'r') as file:
                sections = file.read().split('\n\n')
                layout_lines = sections[0].strip().split('\n')
                height = len(layout_lines)
                width = max(len(line) for line in layout_lines)
                return max(width, height)
        else:
            return max(self.rand_gen_shape)

    def _read_file(self) -> Dict[str, Any]:
        """
        Load layout configuration from a JSON file.

        Returns:
            Dict containing metadata and layers.
        """
        with open(self.txt_file_path, 'r') as f:
            config = json.load(f)

        assert 'height_width' in config and 'layers' in config, "Invalid layout file structure."

        return config

    def _gen_grid(self, width: int, height: int) -> None:
        """
        Generate the environment grid from a multi-layer JSON layout.
        Layers are applied in order. Only doors are allowed to overwrite existing objects (e.g., walls).
        All other objects skip occupied positions to prevent overlap.
        """
        self.grid = Grid(width, height)

        # Load JSON config
        config = self.config
        H, W = config['height_width']
        layers = config['layers']

        # Compute anchor offsets
        free_width = self.display_size - W
        free_height = self.display_size - H
        if self.display_mode == "middle":
            anchor_x = free_width // 2
            anchor_y = free_height // 2
        elif self.display_mode == "random":
            anchor_x = random.choice(range(max(free_width, 1))) if free_width > 0 else 0
            anchor_y = random.choice(range(max(free_height, 1))) if free_height > 0 else 0
        else:
            raise ValueError("Invalid display mode")

        # Apply random rotation and flip
        image_direction = random.choice([0, 1, 2, 3]) if self.random_rotate else 0
        flip = random.choice([0, 1]) if self.random_flip else 0

        # Track filled positions to prevent unwanted overlap
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

            # Collect all candidate positions from the matrix
            raw_positions = [
                (x, y) for y in range(H) for x in range(W) if mat[y][x] == 1
            ]
            # Transform to actual coordinates with anchor, rotation, and flip
            candidates = []
            for x, y in raw_positions:
                x_shift, y_shift = anchor_x + x, anchor_y + y
                x_rot, y_rot = rotate_coordinate(x_shift, y_shift, image_direction, self.display_size)
                x_final, y_final = flip_coordinate(x_rot, y_rot, flip, self.display_size)
                candidates.append((x_final, y_final))

            # Only doors can overwrite filled positions
            if obj_type == "door":
                available = candidates
            else:
                available = [p for p in candidates if p not in filled]

            # Select final positions based on distribution
            if dist == "all":
                used = available
            elif dist == "one":
                if len(available) < 1:
                    raise ValueError(f"No available positions for object '{layer['name']}'")
                used = [random.choice(available)]
            elif isinstance(dist, float):
                count = max(1, int(len(available) * dist))
                used = random.sample(available, min(count, len(available)))
            else:
                raise ValueError(f"Unknown distribution type: {dist}")

            # Place objects at selected positions
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

        # Determine agent position
        if agent_positions:
            self.agent_pos = random.choice(agent_positions)
        else:
            raise ValueError("No available agent position")

        # Set agent orientation
        dir_map = {"right": 0, "down": 1, "left": 2, "up": 3, "random": random.randint(0, 3)}
        self.agent_dir = flip_direction(rotate_direction(dir_map.get(orientation, 0), image_direction), flip)

        # No object is carried at start
        self.carrying = None

    def create_object(self, obj_type: str, color: Optional[str], status: Optional[str]) -> Optional[WorldObj]:
        """
        Create a MiniGrid object given type, color and status.
        Args:
            obj_type: Type of the object, like "wall", "key", "door", etc.
            color: Optional color (e.g., red, blue)
            status: Optional status (e.g., locked, open)
        Returns:
            MiniGrid object or None
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
            return None  # agent is handled separately
        else:
            return None

    def _door_toggle_any_colour(self, door,):
        # If the player has the right key to open the door
        if door.is_locked:
            if isinstance(self.carrying, Key):
                door.is_locked = False
                door.is_open = True
                return True
            return False

        door.is_open = not door.is_open
        return True

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Reset the environment. Always regenerates the grid based on the JSON layout.
        """
        # Reset cumulative reward for sparse mode
        self.cumulative_reward = 0.0

        # Generate new grid from JSON layout
        self._gen_grid(self.width, self.height)

        if self.render_mode == "human":
            self.render()

        self.step_count = 0

        obs = self.gen_obs()

        # Encode carrying info
        obs["carrying"] = {
            "carrying": 1,
            "carrying_colour": 0,
        }
        if self.carrying is not None:
            obs["carrying"] = {
                "carrying": OBJECT_TO_IDX[self.carrying.type],
                "carrying_colour": COLOR_TO_IDX[self.carrying.color],
            }

        # Encode overlap info
        overlap = self.grid.get(*self.agent_pos)
        obs["overlap"] = {
            "obj": OBJECT_TO_IDX[overlap.type] if overlap else 0,
            "colour": COLOR_TO_IDX[overlap.color] if overlap else 0,
        }

        return obs, {}

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Override step method to optionally use NetworkX environment."""
        if self.networkx_env is not None:
            # Get current encoded state
            current_encoded_state = self.encode_state()

            # Set NetworkX environment to current state
            self.networkx_env.current_state = current_encoded_state

            # Execute step in NetworkX environment
            next_networkx_state, reward, terminated, truncated, info = self.networkx_env.step(action)

            # Decode the next state and set environment
            obs, decode_info = self.decode_state(next_networkx_state)

            # Update step count manually since we're bypassing normal step logic
            self.step_count += 1

            # Check for truncation due to max steps
            if self.step_count >= self.max_steps:
                truncated = True

            # Handle sparse vs dense reward mode using NetworkX reward
            if hasattr(self, 'reward_config') and self.reward_config.get("sparse", False):
                # Accumulate reward but only return it at termination
                self.cumulative_reward += reward
                if terminated or truncated:
                    # Return total accumulated reward
                    returned_reward = self.cumulative_reward
                else:
                    # Return zero reward for non-terminal steps
                    returned_reward = 0.0
            else:
                # Return immediate reward (dense mode)
                returned_reward = reward

            # Trigger rendering if in human mode
            if self.render_mode == "human":
                self.render()

            # Merge info dictionaries
            info.update(decode_info)

            return obs, returned_reward, terminated, truncated, info
        else:
            self.step_count += 1

            # Initialize reward with step penalty
            reward = self.reward_config["step_penalty"]

            terminated = False
            truncated = False

            # Get the position in front of the agent
            fwd_pos = self.front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            # Rotate left
            if action == self.actions.left:
                self.agent_dir -= 1
                if self.agent_dir < 0:
                    self.agent_dir += 4

            # Rotate right
            elif action == self.actions.right:
                self.agent_dir = (self.agent_dir + 1) % 4

            # Move forward
            elif action == self.actions.forward:
                if fwd_cell is None or fwd_cell.can_overlap():
                    self.agent_pos = tuple(fwd_pos)
                if fwd_cell is not None and fwd_cell.type == "goal":
                    terminated = True
                    reward = self.reward_config["goal_reward"]
                if fwd_cell is not None and fwd_cell.type == "lava":
                    terminated = True
                    reward = self.reward_config["lava_penalty"]

            # Unified toggle action (uni_toggle)
            elif action == self.actions.uni_toggle:
                # Case 1: Forward cell exists (not empty)
                if fwd_cell:
                    # If carrying nothing and forward cell can be picked up, perform pickup
                    if self.carrying is None and fwd_cell.can_pickup():
                        self.carrying = fwd_cell
                        self.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(fwd_pos[0], fwd_pos[1], None)
                        reward += self.reward_config["pickup_reward"]

                    # If forward cell is a door, perform toggle
                    elif fwd_cell.type == "door":
                        was_open = fwd_cell.is_open
                        if self.any_key_opens_the_door:
                            self._door_toggle_any_colour(fwd_cell, )
                        else:
                            fwd_cell.toggle(self, fwd_pos)
                        # Update rewards based on door status
                        if fwd_cell.is_open and not was_open:
                            reward += self.reward_config["door_open_reward"]
                        elif not fwd_cell.is_open and was_open:
                            reward += self.reward_config["door_close_penalty"]

                # Case 2: Forward cell is empty (None)
                else:
                    # If carrying an object, drop it in the empty space
                    if self.carrying:
                        self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                        self.carrying.cur_pos = fwd_pos

                        # Special case: if dropping a key, give different reward
                        if self.carrying.type == "key":
                            reward += self.reward_config["key_drop_penalty"]
                        else:
                            reward += self.reward_config["item_drop_penalty"]

                        self.carrying = None

            else:
                raise ValueError(f"Unknown action: {action}")

            if self.step_count >= self.max_steps:
                truncated = True

            # Handle sparse vs dense reward mode
            if self.reward_config["sparse"]:
                # Accumulate reward but only return it at termination
                self.cumulative_reward += reward
                if terminated or truncated:
                    # Return total accumulated reward
                    returned_reward = self.cumulative_reward
                else:
                    # Return zero reward for non-terminal steps
                    returned_reward = 0.0
            else:
                # Return immediate reward (dense mode)
                returned_reward = reward

            if self.render_mode == "human":
                self.render()

            obs = self.gen_obs()

            # Set carrying observation
            obs["carrying"] = {
                "carrying": 1,
                "carrying_colour": 0,
            }

            if self.carrying is not None and self.carrying != 0:
                carrying = OBJECT_TO_IDX[self.carrying.type]
                carrying_colour = COLOR_TO_IDX[self.carrying.color]

                obs["carrying"] = {
                    "carrying": carrying,
                    "carrying_colour": carrying_colour,
                }

            # Set overlap observation
            obs["overlap"] = {
                "obj": 0,
                "colour": 0,
            }

            overlap = self.grid.get(*self.agent_pos)
            if overlap is not None:
                overlap_colour = COLOR_TO_IDX[overlap.color]
                obs["overlap"] = {
                    "obj": OBJECT_TO_IDX[overlap.type],
                    "colour": overlap_colour,
                }

            return obs, returned_reward, terminated, truncated, {}

    def encode_state(self) -> str:
        """
        Encode the current environment state into a compact string representation.
        Format: <W>x<H>:<static_objects>:<dynamic_objects>:<agent_state>

        Returns:
            str: Encoded state string that uniquely represents the environment
        """
        # Get grid dimensions
        width, height = self.width, self.height

        # Color mapping for compact representation
        color_map = {
            'red': 'r', 'green': 'g', 'blue': 'b',
            'purple': 'p', 'yellow': 'y', 'orange': 'o'
        }

        # Object type mapping for compact representation
        obj_map = {
            'wall': 'W', 'floor': 'F', 'lava': 'L', 'goal': 'G',
            'key': 'K', 'ball': 'B', 'box': 'X', 'door': 'D'
        }

        # Collect objects by type in alphabetical order
        object_groups = {
            'B': [],  # Ball
            'D': [],  # Door
            'F': [],  # Floor
            'G': [],  # Goal
            'K': [],  # Key
            'L': [],  # Lava
            'W': [],  # Wall
            'X': []  # Box
        }

        # Scan the entire grid to collect objects
        for y in range(height):
            for x in range(width):
                pos = y * width + x  # Convert to linear index
                obj = self.grid.get(x, y)

                if obj is not None:
                    obj_type = obj_map.get(obj.type)
                    if obj_type:
                        if obj_type == 'D':  # Door needs special handling for state
                            color_code = color_map.get(obj.color, obj.color[0] if obj.color else '')
                            if obj.is_locked:
                                state_code = 'l'
                            elif obj.is_open:
                                state_code = 'o'
                            else:
                                state_code = 'c'
                            door_encoding = f"{obj_type}_{color_code}_{state_code}@{pos}"
                            object_groups[obj_type].append((pos, door_encoding))
                        elif hasattr(obj, 'color') and obj.color:  # Objects with color
                            color_code = color_map.get(obj.color, obj.color[0] if obj.color else '')
                            obj_encoding = f"{obj_type}_{color_code}@{pos}"
                            object_groups[obj_type].append((pos, obj_encoding))
                        else:  # Objects without color
                            obj_encoding = f"{obj_type}@{pos}"
                            object_groups[obj_type].append((pos, obj_encoding))

        # Build object sections, grouping same types and sorting by position
        object_sections = []
        for obj_type in sorted(object_groups.keys()):
            if object_groups[obj_type]:
                # Sort by position for deterministic encoding
                sorted_objects = sorted(object_groups[obj_type], key=lambda x: x[0])

                # Group objects of same type and attributes
                grouped = {}
                for pos, encoding in sorted_objects:
                    # Extract the part before '@' as the key
                    key = encoding.split('@')[0]
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append(pos)

                # Create compact representation for each group
                group_encodings = []
                for key in sorted(grouped.keys()):
                    positions = sorted(grouped[key])
                    pos_str = ','.join(map(str, positions))
                    group_encodings.append(f"{key}@{pos_str}")

                object_sections.append(';'.join(group_encodings))
            else:
                object_sections.append('')  # Empty section for missing object types

        # Build agent state
        agent_pos = self.agent_pos[1] * width + self.agent_pos[0]  # Convert to linear index
        agent_dir = self.agent_dir

        if self.carrying is not None:
            carry_type = obj_map.get(self.carrying.type, self.carrying.type)
            if hasattr(self.carrying, 'color') and self.carrying.color:
                carry_color = color_map.get(self.carrying.color, self.carrying.color[0])
                carrying = f"{carry_type}_{carry_color}"
            else:
                carrying = carry_type
        else:
            carrying = "none"

        agent_state = f"A@{agent_pos}_{agent_dir}_{carrying}"

        # Combine all parts
        grid_size = f"{width}x{height}"
        objects_part = ':'.join(object_sections)

        return f"{grid_size}:{objects_part}:{agent_state}"


    def decode_state(self, encoded_state: str) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Decode an encoded state string and set the environment to that state.
        This forcibly overwrites the current environment state.

        Args:
            encoded_state (str): The encoded state string to decode

        Raises:
            ValueError: If the encoded state is invalid or malformed
        """
        try:
            # Reset cumulative reward
            self.cumulative_reward = 0.0

            # Parse the main components
            parts = encoded_state.split(':')
            if len(parts) < 3:
                raise ValueError("Invalid encoded state format: insufficient parts")

            grid_part = parts[0]
            objects_part = ':'.join(parts[1:-1])  # Handle potential extra colons in object encoding
            agent_part = parts[-1]

            # Parse grid dimensions
            if 'x' not in grid_part:
                raise ValueError("Invalid grid size format")
            width_str, height_str = grid_part.split('x')
            width, height = int(width_str), int(height_str)

            # Reverse color mapping
            reverse_color_map = {
                'r': 'red', 'g': 'green', 'b': 'blue',
                'p': 'purple', 'y': 'yellow', 'o': 'orange'
            }

            # Reverse object mapping
            reverse_obj_map = {
                'W': 'wall', 'F': 'floor', 'L': 'lava', 'G': 'goal',
                'K': 'key', 'B': 'ball', 'X': 'box', 'D': 'door'
            }

            # Initialize new grid
            self.grid = Grid(width, height)

            # Parse objects if objects_part is not empty
            if objects_part.strip():
                object_sections = objects_part.split(':')

                # Process each object type section
                for section in object_sections:
                    if not section.strip():
                        continue

                    # Split multiple object groups in the same section
                    groups = section.split(';')
                    for group in groups:
                        if not group.strip():
                            continue

                        # Parse individual object group: "TYPE_COLOR@pos1,pos2,pos3"
                        if '@' not in group:
                            continue

                        obj_spec, positions_str = group.split('@', 1)
                        positions = [int(pos) for pos in positions_str.split(',')]

                        # Parse object specification
                        spec_parts = obj_spec.split('_')
                        obj_type_code = spec_parts[0]

                        if obj_type_code not in reverse_obj_map:
                            continue

                        obj_type = reverse_obj_map[obj_type_code]

                        # Handle different object types
                        for pos in positions:
                            x = pos % width
                            y = pos // width

                            if x >= width or y >= height:
                                raise ValueError(f"Position {pos} out of bounds for grid {width}x{height}")

                            obj = None

                            if obj_type == 'wall':
                                obj = Wall()
                            elif obj_type == 'floor':
                                obj = Floor()
                            elif obj_type == 'lava':
                                obj = Lava()
                            elif obj_type == 'goal':
                                obj = Goal()
                            elif obj_type in ['key', 'ball', 'box']:
                                if len(spec_parts) >= 2:
                                    color_code = spec_parts[1]
                                    color = reverse_color_map.get(color_code, color_code)
                                else:
                                    color = 'red'  # Default color

                                if obj_type == 'key':
                                    obj = Key(color)
                                elif obj_type == 'ball':
                                    obj = Ball(color)
                                elif obj_type == 'box':
                                    obj = Box(color)
                            elif obj_type == 'door':
                                if len(spec_parts) >= 3:
                                    color_code = spec_parts[1]
                                    state_code = spec_parts[2]
                                    color = reverse_color_map.get(color_code, color_code)

                                    is_open = state_code == 'o'
                                    is_locked = state_code == 'l'
                                    obj = Door(color, is_open=is_open, is_locked=is_locked)
                                else:
                                    obj = Door('red')  # Default door

                            if obj is not None:
                                self.grid.set(x, y, obj)

            # Parse agent state: "A@pos_dir_carrying"
            if not agent_part.startswith('A@'):
                raise ValueError("Invalid agent state format")

            agent_spec = agent_part[2:]  # Remove "A@"
            agent_parts = agent_spec.split('_')

            if len(agent_parts) < 3:
                raise ValueError("Invalid agent specification")

            agent_pos = int(agent_parts[0])
            agent_dir = int(agent_parts[1])
            carrying_spec = '_'.join(agent_parts[2:])  # Handle underscores in carrying spec

            # Set agent position
            agent_x = agent_pos % width
            agent_y = agent_pos // width

            if agent_x >= width or agent_y >= height:
                raise ValueError(f"Agent position {agent_pos} out of bounds")

            self.agent_pos = (agent_x, agent_y)
            self.agent_dir = agent_dir % 4  # Ensure valid direction

            # Set carrying object
            if carrying_spec == 'none':
                self.carrying = None
            else:
                carry_parts = carrying_spec.split('_')
                carry_type_code = carry_parts[0]

                if carry_type_code in reverse_obj_map:
                    carry_type = reverse_obj_map[carry_type_code]

                    if carry_type == 'key':
                        if len(carry_parts) >= 2:
                            color_code = carry_parts[1]
                            color = reverse_color_map.get(color_code, color_code)
                        else:
                            color = 'red'
                        self.carrying = Key(color)
                    elif carry_type == 'ball':
                        if len(carry_parts) >= 2:
                            color_code = carry_parts[1]
                            color = reverse_color_map.get(color_code, color_code)
                        else:
                            color = 'red'
                        self.carrying = Ball(color)
                    elif carry_type == 'box':
                        if len(carry_parts) >= 2:
                            color_code = carry_parts[1]
                            color = reverse_color_map.get(color_code, color_code)
                        else:
                            color = 'red'
                        self.carrying = Box(color)

                    # Set carried object position to invalid (carried objects don't have grid positions)
                    if self.carrying is not None:
                        self.carrying.cur_pos = np.array([-1, -1])
                else:
                    self.carrying = None

            # Reset step count
            self.step_count = 0

            # Update environment dimensions if they changed
            self.width = width
            self.height = height

            # get observation
            obs = self.gen_obs()

            # Set carrying observation
            obs["carrying"] = {
                "carrying": 1,
                "carrying_colour": 0,
            }

            if self.carrying is not None and self.carrying != 0:
                carrying = OBJECT_TO_IDX[self.carrying.type]
                carrying_colour = COLOR_TO_IDX[self.carrying.color]

                obs["carrying"] = {
                    "carrying": carrying,
                    "carrying_colour": carrying_colour,
                }

            # Set overlap observation
            obs["overlap"] = {
                "obj": 0,
                "colour": 0,
            }

            overlap = self.grid.get(*self.agent_pos)
            if overlap is not None:
                overlap_colour = COLOR_TO_IDX[overlap.color]
                obs["overlap"] = {
                    "obj": OBJECT_TO_IDX[overlap.type],
                    "colour": overlap_colour,
                }

            return obs, {}

        except (ValueError, IndexError, AttributeError) as e:
            raise ValueError(f"Failed to decode state: {str(e)}")

    def get_start_states(self) -> List[str]:
        """
        Get all possible starting states for the CustomMiniGrid environment.

        This method analyzes the JSON configuration to determine all possible initial
        configurations considering:
        - Multiple possible agent positions and orientations
        - Random object placements with different distribution types
        - Random transformations (rotation/flip if enabled)
        - Agent always starts empty-handed

        Returns:
            List[str]: List of all possible starting state encodings
        """

        start_states = []

        # Get configuration
        config = self.config
        H, W = config['height_width']
        layers = config['layers']

        # Determine all possible transformations
        rotations = [0, 1, 2, 3] if self.random_rotate else [0]
        flips = [0, 1] if self.random_flip else [0]

        # Determine all possible anchor positions
        free_width = self.display_size - W
        free_height = self.display_size - H

        if self.display_mode == "middle":
            anchor_positions = [(free_width // 2, free_height // 2)]
        elif self.display_mode == "random":
            anchor_x_options = list(range(max(free_width, 1))) if free_width > 0 else [0]
            anchor_y_options = list(range(max(free_height, 1))) if free_height > 0 else [0]
            anchor_positions = list(itertools.product(anchor_x_options, anchor_y_options))
        else:
            anchor_positions = [(0, 0)]

        # Process each layer to determine possible placements
        layer_possibilities = []
        agent_orientations = []

        for layer in layers:
            obj_type = layer["obj"]
            colour = layer.get("colour", None)
            status = layer.get("status", None)
            dist = layer.get("distribution", "all")
            mat = layer["matrix"]

            # Get raw positions from matrix
            raw_positions = [
                (x, y) for y in range(H) for x in range(W) if mat[y][x] == 1
            ]

            if not raw_positions:
                layer_possibilities.append([])
                continue

            # Handle agent orientation
            if obj_type == "agent":
                orientation = layer.get("orientation", "random")
                if orientation == "random":
                    agent_orientations = [0, 1, 2, 3]  # All possible directions
                else:
                    dir_map = {"right": 0, "down": 1, "left": 2, "up": 3}
                    agent_orientations = [dir_map.get(orientation, 0)]

            # Generate all possible position sets for this layer
            layer_position_sets = []

            if dist == "all":
                # All positions are used - only one possibility
                layer_position_sets = [raw_positions]
            elif dist == "one":
                # Each position could be the chosen one
                layer_position_sets = [[pos] for pos in raw_positions]
            elif isinstance(dist, float):
                # Generate all possible combinations for the given percentage
                count = max(1, int(len(raw_positions) * dist))
                count = min(count, len(raw_positions))  # Don't exceed available positions

                # Generate all combinations of 'count' positions
                from itertools import combinations
                layer_position_sets = [list(combo) for combo in combinations(raw_positions, count)]

            # Store layer info with its possibilities
            layer_possibilities.append({
                'obj_type': obj_type,
                'colour': colour,
                'status': status,
                'position_sets': layer_position_sets
            })

        # Generate all combinations of transformations and anchor positions
        for rotation in rotations:
            for flip in flips:
                for anchor_x, anchor_y in anchor_positions:

                    # Generate all combinations of layer placements
                    layer_combinations = [layer['position_sets'] for layer in layer_possibilities if layer]

                    if not layer_combinations:
                        continue

                    # Generate cartesian product of all layer possibilities
                    for combination in itertools.product(*layer_combinations):
                        try:
                            # Create a temporary grid to build this specific configuration
                            temp_grid = Grid(self.display_size, self.display_size)
                            filled = set()
                            agent_positions = []
                            temp_carrying = None  # Agent always starts empty-handed

                            # Apply each layer in this combination
                            for layer_idx, (layer_info, position_set) in enumerate(
                                    zip(layer_possibilities, combination)):
                                if not layer_info or not position_set:
                                    continue

                                obj_type = layer_info['obj_type']
                                colour = layer_info['colour']
                                status = layer_info['status']

                                # Transform positions
                                transformed_positions = []
                                for x, y in position_set:
                                    x_shift, y_shift = anchor_x + x, anchor_y + y
                                    x_rot, y_rot = rotate_coordinate(x_shift, y_shift, rotation, self.display_size)
                                    x_final, y_final = flip_coordinate(x_rot, y_rot, flip, self.display_size)
                                    transformed_positions.append((x_final, y_final))

                                # Check for conflicts (except doors can overwrite)
                                if obj_type != "door":
                                    available_positions = [pos for pos in transformed_positions if pos not in filled]
                                    if len(available_positions) != len(transformed_positions):
                                        # Skip this combination due to conflicts
                                        break
                                    used_positions = available_positions
                                else:
                                    used_positions = transformed_positions

                                # Place objects
                                for x, y in used_positions:
                                    if x < 0 or x >= self.display_size or y < 0 or y >= self.display_size:
                                        continue  # Skip out-of-bounds positions

                                    obj = self.create_object(obj_type, colour, status)

                                    if obj_type == "agent":
                                        agent_positions.append((x, y))
                                    else:
                                        temp_grid.set(x, y, obj)
                                        if obj_type != "door":
                                            filled.add((x, y))

                            else:  # Only executed if the for loop completed without break
                                # All layers placed successfully
                                if not agent_positions:
                                    continue  # Skip if no agent position

                                # Generate states for all agent position and orientation combinations
                                for agent_pos in agent_positions:
                                    for agent_orientation in agent_orientations:
                                        # Apply transformation to agent direction
                                        final_agent_dir = flip_direction(
                                            rotate_direction(agent_orientation, rotation),
                                            flip
                                        )

                                        # Create the state encoding
                                        # Temporarily set the environment state
                                        old_grid = self.grid
                                        old_agent_pos = getattr(self, 'agent_pos', None)
                                        old_agent_dir = getattr(self, 'agent_dir', None)
                                        old_carrying = getattr(self, 'carrying', None)
                                        old_width = getattr(self, 'width', None)
                                        old_height = getattr(self, 'height', None)

                                        try:
                                            # Set temporary state
                                            self.grid = temp_grid
                                            self.agent_pos = agent_pos
                                            self.agent_dir = final_agent_dir
                                            self.carrying = None  # Always empty-handed at start
                                            self.width = self.display_size
                                            self.height = self.display_size

                                            # Generate state encoding
                                            state_encoding = self.encode_state()

                                            # Add to results if not already present
                                            if state_encoding not in start_states:
                                                start_states.append(state_encoding)

                                        finally:
                                            # Restore original state
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
                            # Skip this combination if any error occurs
                            continue

        return start_states


def rotate_coordinate(x, y, rotation_mode, n):
    """
    Rotate a 2D coordinate in a gridworld.
    Parameters:
    x, y (int): Original coordinates.
    rotation_mode (int): Rotation mode (0, 1, 2, 3).
    n (int): Dimension of the matrix.
    Returns:
    tuple: The new coordinates (new_x, new_y) after rotation.
    """
    if rotation_mode == 0:
        # No rotation
        return x, y
    elif rotation_mode == 1:
        # Clockwise rotation by 90 degrees
        return y, n - 1 - x
    elif rotation_mode == 2:
        # Clockwise rotation by 180 degrees
        return n - 1 - x, n - 1 - y
    elif rotation_mode == 3:
        # Clockwise rotation by 270 degrees
        return n - 1 - y, x
    else:
        raise ValueError("Invalid rotation mode. Please choose between 0, 1, 2, or 3.")


def flip_coordinate(x, y, flip_mode, n):
    """
    Flip a 2D coordinate in a gridworld along the x-axis.
    Parameters:
    x, y (int): Original coordinates.
    flip_mode (int): Flip mode (0 for no flip, 1 for flip along x-axis).
    n (int): Dimension of the matrix.
    Returns:
    tuple: The new coordinates (new_x, new_y) after flipping.
    """
    if flip_mode == 0:
        # No flip
        return x, y
    elif flip_mode == 1:
        # Flip along the x-axis
        return n - 1 - x, y
    else:
        raise ValueError("Invalid flip mode. Please choose between 0 and 1.")


def rotate_direction(direction, rotation_mode):
    """
    Rotate a direction in a gridworld.
    Parameters:
    direction (int): Original direction (0, 1, 2, 3).
    rotation_mode (int): Rotation mode (0, 1, 2, 3).
    Returns:
    int: New direction after rotation.
    """
    # Apply rotation by adding the rotation mode and taking modulo 4 to cycle back to 0 after 3.
    if rotation_mode in [0, 1, 2, 3]:
        return (direction + rotation_mode) % 4
    else:
        raise ValueError("Invalid rotation mode. Please choose between 0, 1, 2, or 3.")


def flip_direction(direction, flip_mode):
    """
    Flip a direction in a gridworld along the x-axis.
    Parameters:
    direction (int): Original direction (0, 1, 2, 3).
    flip_mode (int): Flip mode (0 for no flip, 1 for flip).
    Returns:
    int: New direction after flipping.
    """
    if flip_mode == 0:
        # No flip
        return direction
    elif flip_mode == 1:
        # Flip direction: right/left remain the same, up/down are flipped
        flip_map = {0: 0, 1: 3, 2: 2, 3: 1}
        return flip_map[direction]
    else:
        raise ValueError("Invalid flip mode. Please choose between 0 and 1.")


if __name__ == "__main__":
    env = CustomMiniGridEnv(
        json_file_path='./maps/door-key-no-random-3x6.json',
        config=None,
        display_size=None,
        display_mode="random",
        random_rotate=True,
        random_flip=True,
        render_carried_objs=True,
        render_mode="human",
    )
    manual_control = SimpleManualControl(env)  # Allows manual control for testing and visualization
    manual_control.start()  # Start the manual control interface
