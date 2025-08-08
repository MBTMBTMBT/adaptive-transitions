import json
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import networkx as nx


class MDPNetwork:
    """
    A class to represent MDP using NetworkX directed graph.
    Can load from JSON configuration and export back to JSON.
    Supports optional string state mapping for environments that use string states.
    """

    def __init__(self, config_data: Optional[Dict] = None, config_path: Optional[str] = None,
                 int_to_state: Optional[Dict[int, Union[str, int]]] = None,
                 state_to_int: Optional[Dict[Union[str, int], int]] = None):
        """
        Initialize MDP Network from config data or file path.

        Args:
            config_data: Dictionary containing MDP configuration
            config_path: Path to JSON configuration file
            int_to_state: Optional mapping from internal int IDs to original states
            state_to_int: Optional mapping from original states to internal int IDs
        """
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        elif config_data is not None:
            self.config = config_data.copy()
        else:
            raise ValueError("Either config_data or config_path must be provided")

        # Store state mappings (None if not using string states)
        self.int_to_state = int_to_state
        self.state_to_int = state_to_int
        self.has_string_mapping = (int_to_state is not None) and (state_to_int is not None)

        # Extract configuration parameters
        self.num_actions = self.config["num_actions"]
        self.states = self.config["states"]
        self.terminal_states = set(self.config["terminal_states"])
        self.start_states = self.config["start_states"]
        self.default_reward = self.config.get("default_reward", 0.0)
        self.state_rewards = self.config.get("state_rewards", {})

        # Build NetworkX directed graph
        self._build_graph()

    def _normalize_state_input(self, state: Union[int, str]) -> int:
        """Convert state input to internal int representation."""
        if self.has_string_mapping:
            if isinstance(state, str):
                if state in self.state_to_int:
                    return self.state_to_int[state]
                else:
                    raise ValueError(f"Unknown string state: {state}")
            elif isinstance(state, int):
                return state
        return int(state)

    def _format_state_output(self, int_state: int, as_string: bool = False) -> Union[int, str]:
        """Format state output based on requirements."""
        if as_string and self.has_string_mapping:
            return self.int_to_state.get(int_state, int_state)
        return int_state

    def _build_graph(self):
        """Build the NetworkX directed graph from configuration."""
        self.graph = nx.DiGraph()

        # Add all states as nodes
        for state in self.states:
            reward = float(self.state_rewards.get(str(state), self.default_reward))
            self.graph.add_node(state, reward=reward, is_terminal=state in self.terminal_states)

        # Add transitions as edges with probabilities
        transitions = self.config.get("transitions", {})
        for state_str, actions in transitions.items():
            state = int(state_str)
            for action_str, next_states in actions.items():
                action = int(action_str)

                # Normalize probabilities if they don't sum to 1
                total_prob = sum(next_states.values())
                if total_prob > 0:
                    for next_state_str, prob in next_states.items():
                        next_state = int(next_state_str)
                        normalized_prob = prob / total_prob

                        # Add or update edge
                        if self.graph.has_edge(state, next_state):
                            if 'transitions' not in self.graph[state][next_state]:
                                self.graph[state][next_state]['transitions'] = {}
                            self.graph[state][next_state]['transitions'][action] = normalized_prob
                        else:
                            self.graph.add_edge(state, next_state, transitions={action: normalized_prob})

    def get_transition_probabilities(self, state: Union[int, str], action: int) -> Dict[Union[int, str], float]:
        """Get transition probabilities for a given state-action pair."""
        int_state = self._normalize_state_input(state)
        probs = {}

        for next_state in self.graph.successors(int_state):
            edge_data = self.graph[int_state][next_state]
            if 'transitions' in edge_data and action in edge_data['transitions']:
                # Return in same format as input
                output_state = self._format_state_output(next_state, isinstance(state, str))
                probs[output_state] = edge_data['transitions'][action]

        return probs

    def get_state_reward(self, state: Union[int, str]) -> float:
        """Get reward for entering a specific state."""
        int_state = self._normalize_state_input(state)
        return self.graph.nodes[int_state]['reward']

    def is_terminal_state(self, state: Union[int, str]) -> bool:
        """Check if a state is terminal."""
        int_state = self._normalize_state_input(state)
        return int_state in self.terminal_states

    def sample_next_state(self, state: Union[int, str], action: int, rng: np.random.Generator,
                          as_string: bool = False) -> Union[int, str]:
        """Sample next state based on transition probabilities."""
        int_state = self._normalize_state_input(state)
        probs = {}

        for next_state in self.graph.successors(int_state):
            edge_data = self.graph[int_state][next_state]
            if 'transitions' in edge_data and action in edge_data['transitions']:
                probs[next_state] = edge_data['transitions'][action]

        if not probs:
            # No valid transitions for this action, stay in current state
            return self._format_state_output(int_state, as_string)

        # Sample next state based on probabilities
        states = list(probs.keys())
        probabilities = list(probs.values())
        next_int_state = rng.choice(states, p=probabilities)

        return self._format_state_output(next_int_state, as_string)

    def sample_start_state(self, rng: np.random.Generator, as_string: bool = False) -> Union[int, str]:
        """Sample a random start state."""
        int_start_state = rng.choice(self.start_states)
        return self._format_state_output(int_start_state, as_string)

    def export_to_json(self, output_path: str):
        """Export the MDP configuration to a JSON file."""
        # Reconstruct transitions from graph
        transitions = {}

        for state in self.states:
            state_transitions = {}

            for next_state in self.graph.successors(state):
                edge_data = self.graph[state][next_state]
                if 'transitions' in edge_data:
                    for action, prob in edge_data['transitions'].items():
                        action_str = str(int(action))  # Ensure string conversion
                        next_state_str = str(int(next_state))  # Ensure string conversion
                        prob_float = float(prob)  # Ensure float conversion

                        if action_str not in state_transitions:
                            state_transitions[action_str] = {}
                        state_transitions[action_str][next_state_str] = prob_float

            if state_transitions:
                transitions[str(int(state))] = state_transitions

        # Reconstruct state rewards (only non-default ones)
        state_rewards = {}
        for state in self.states:
            reward = self.graph.nodes[state]['reward']
            reward_float = float(reward)  # Ensure float conversion
            if reward_float != self.default_reward:
                state_rewards[str(int(state))] = reward_float

        # Create export configuration with explicit type conversions
        export_config = {
            "num_actions": int(self.num_actions),
            "states": [int(s) for s in self.states],
            "start_states": [int(s) for s in self.start_states],
            "terminal_states": [int(s) for s in self.terminal_states],
            "default_reward": float(self.default_reward),
            "state_rewards": state_rewards,
            "transitions": transitions
        }

        # Add mapping information if available
        if self.has_string_mapping:
            export_config["state_mapping"] = {
                "int_to_state": {str(k): v for k, v in self.int_to_state.items()},
                "state_to_int": {str(k): v for k, v in self.state_to_int.items()}
            }

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(export_config, f, indent=2)

    def add_state(self, state: Union[int, str], reward: Optional[float] = None,
                  is_terminal: bool = False, is_start: bool = False):
        """Add a new state to the MDP."""
        int_state = self._normalize_state_input(state)

        if int_state not in self.states:
            self.states.append(int_state)

        reward_value = reward if reward is not None else self.default_reward
        self.graph.add_node(int_state, reward=reward_value, is_terminal=is_terminal)

        if is_terminal:
            self.terminal_states.add(int_state)

        if is_start and int_state not in self.start_states:
            self.start_states.append(int_state)

        # Update state rewards if different from default
        if reward is not None and reward != self.default_reward:
            self.state_rewards[str(int_state)] = reward

    def add_transition(self, from_state: Union[int, str], to_state: Union[int, str],
                       action: int, probability: float):
        """Add a transition between states."""
        int_from_state = self._normalize_state_input(from_state)
        int_to_state = self._normalize_state_input(to_state)

        if not self.graph.has_edge(int_from_state, int_to_state):
            self.graph.add_edge(int_from_state, int_to_state, transitions={})

        if 'transitions' not in self.graph[int_from_state][int_to_state]:
            self.graph[int_from_state][int_to_state]['transitions'] = {}

        self.graph[int_from_state][int_to_state]['transitions'][action] = probability

    def get_graph_copy(self) -> nx.DiGraph:
        """Get a copy of the underlying NetworkX graph."""
        return self.graph.copy()
