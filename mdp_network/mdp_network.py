import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import networkx as nx


class MDPNetwork:
    """
    A class to represent MDP using NetworkX directed graph.
    Can load from JSON configuration and export back to JSON.
    """

    def __init__(self, config_data: Optional[Dict] = None, config_path: Optional[str] = None):
        """
        Initialize MDP Network from config data or file path.

        Args:
            config_data: Dictionary containing MDP configuration
            config_path: Path to JSON configuration file
        """
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        elif config_data is not None:
            self.config = config_data.copy()
        else:
            raise ValueError("Either config_data or config_path must be provided")

        # Extract configuration parameters
        self.num_actions = self.config["num_actions"]
        self.states = self.config["states"]
        self.terminal_states = set(self.config["terminal_states"])
        self.start_states = self.config["start_states"]
        self.default_reward = self.config.get("default_reward", 0.0)
        self.state_rewards = self.config.get("state_rewards", {})

        # Build NetworkX directed graph
        self._build_graph()

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

    def get_transition_probabilities(self, state: int, action: int) -> Dict[int, float]:
        """Get transition probabilities for a given state-action pair."""
        probs = {}

        for next_state in self.graph.successors(state):
            edge_data = self.graph[state][next_state]
            if 'transitions' in edge_data and action in edge_data['transitions']:
                probs[next_state] = edge_data['transitions'][action]

        return probs

    def get_state_reward(self, state: int) -> float:
        """Get reward for entering a specific state."""
        return self.graph.nodes[state]['reward']

    def is_terminal_state(self, state: int) -> bool:
        """Check if a state is terminal."""
        return state in self.terminal_states

    def sample_next_state(self, state: int, action: int, rng: np.random.Generator) -> int:
        """Sample next state based on transition probabilities."""
        probs = self.get_transition_probabilities(state, action)

        if not probs:
            # No valid transitions for this action, stay in current state
            return state

        # Sample next state based on probabilities
        states = list(probs.keys())
        probabilities = list(probs.values())
        return rng.choice(states, p=probabilities)

    def sample_start_state(self, rng: np.random.Generator) -> int:
        """Sample a random start state."""
        return rng.choice(self.start_states)

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
                        if str(action) not in state_transitions:
                            state_transitions[str(action)] = {}
                        state_transitions[str(action)][str(next_state)] = prob

            if state_transitions:
                transitions[str(state)] = state_transitions

        # Reconstruct state rewards (only non-default ones)
        state_rewards = {}
        for state in self.states:
            reward = self.graph.nodes[state]['reward']
            if reward != self.default_reward:
                state_rewards[str(state)] = reward

        # Create export configuration
        export_config = {
            "num_actions": self.num_actions,
            "states": self.states,
            "start_states": self.start_states,
            "terminal_states": list(self.terminal_states),
            "default_reward": self.default_reward,
            "state_rewards": state_rewards,
            "transitions": transitions
        }

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(export_config, f, indent=2)

    def add_state(self, state: int, reward: Optional[float] = None, is_terminal: bool = False, is_start: bool = False):
        """Add a new state to the MDP."""
        if state not in self.states:
            self.states.append(state)

        reward_value = reward if reward is not None else self.default_reward
        self.graph.add_node(state, reward=reward_value, is_terminal=is_terminal)

        if is_terminal:
            self.terminal_states.add(state)

        if is_start and state not in self.start_states:
            self.start_states.append(state)

        # Update state rewards if different from default
        if reward is not None and reward != self.default_reward:
            self.state_rewards[str(state)] = reward

    def add_transition(self, from_state: int, to_state: int, action: int, probability: float):
        """Add a transition between states."""
        if not self.graph.has_edge(from_state, to_state):
            self.graph.add_edge(from_state, to_state, transitions={})

        if 'transitions' not in self.graph[from_state][to_state]:
            self.graph[from_state][to_state]['transitions'] = {}

        self.graph[from_state][to_state]['transitions'][action] = probability

    def get_graph_copy(self) -> nx.DiGraph:
        """Get a copy of the underlying NetworkX graph."""
        return self.graph.copy()
