from gymnasium.envs.toy_text.taxi import TaxiEnv
from customisable_env_abs import CustomisableEnvAbs
from typing import Tuple, Dict, Any, Union, List
from gymnasium.core import ObsType
import numpy as np

from mdp_network import MDPNetwork


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
