from typing import Dict, List, Tuple, Optional, Any
import csv
import os

import numpy as np


class QTable:
    """
    Q-table representation for MDP Q-values.
    Supports CSV import/export functionality.
    """

    def __init__(self, q_values: Optional[Dict[int, Dict[int, float]]] = None):
        """
        Initialize Q-table.

        Args:
            q_values: Dictionary mapping state -> action -> q_value
        """
        self.q_values = q_values if q_values is not None else {}

    def get_q_value(self, state: int, action: int) -> float:
        """Get Q-value for a state-action pair."""
        return self.q_values.get(state, {}).get(action, 0.0)

    def set_q_value(self, state: int, action: int, value: float):
        """Set Q-value for a state-action pair."""
        if state not in self.q_values:
            self.q_values[state] = {}
        self.q_values[state][action] = value

    def get_best_action(self, state: int) -> Tuple[int, float]:
        """
        Get the best action and its Q-value for a given state.

        Returns:
            Tuple of (best_action, best_q_value)
        """
        if state not in self.q_values or not self.q_values[state]:
            return 0, 0.0

        best_action = max(self.q_values[state], key=self.q_values[state].get)
        best_value = self.q_values[state][best_action]
        return best_action, best_value

    def get_all_states(self) -> List[int]:
        """Get all states in the Q-table."""
        return list(self.q_values.keys())

    def get_all_actions(self, state: int) -> List[int]:
        """Get all actions available for a given state."""
        return list(self.q_values.get(state, {}).keys())

    def export_to_csv(self, file_path: str):
        """
        Export Q-table to CSV file.
        CSV format: state,action,q_value
        """
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['state', 'action', 'q_value'])

            for state in sorted(self.q_values.keys()):
                for action in sorted(self.q_values[state].keys()):
                    writer.writerow([state, action, self.q_values[state][action]])

    def import_from_csv(self, file_path: str):
        """
        Import Q-table from CSV file.
        CSV format: state,action,q_value
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        self.q_values = {}

        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                state = int(row['state'])
                action = int(row['action'])
                q_value = float(row['q_value'])

                if state not in self.q_values:
                    self.q_values[state] = {}

                self.q_values[state][action] = q_value

    def __str__(self) -> str:
        """String representation of Q-table."""
        result = "Q-Table:\n"
        for state in sorted(self.q_values.keys()):
            result += f"State {state}: "
            actions = self.q_values[state]
            result += ", ".join([f"A{action}={value:.4f}" for action, value in sorted(actions.items())])
            result += "\n"
        return result


class ValueTable:
    """
    Value table representation for MDP state values.
    Supports CSV import/export functionality.
    """

    def __init__(self, values: Optional[Dict[int, float]] = None):
        """
        Initialize Value table.

        Args:
            values: Dictionary mapping state -> value
        """
        self.values = values if values is not None else {}

    def get_value(self, state: int) -> float:
        """Get value for a state."""
        return self.values.get(state, 0.0)

    def set_value(self, state: int, value: float):
        """Set value for a state."""
        self.values[state] = value

    def get_best_state(self) -> Tuple[int, float]:
        """
        Get the state with highest value and its value.

        Returns:
            Tuple of (best_state, best_value)
        """
        if not self.values:
            return 0, 0.0

        best_state = max(self.values, key=self.values.get)
        best_value = self.values[best_state]
        return best_state, best_value

    def get_all_states(self) -> List[int]:
        """Get all states in the Value table."""
        return list(self.values.keys())

    def export_to_csv(self, file_path: str):
        """
        Export Value table to CSV file.
        CSV format: state,value
        """
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['state', 'value'])

            for state in sorted(self.values.keys()):
                writer.writerow([state, self.values[state]])

    def import_from_csv(self, file_path: str):
        """
        Import Value table from CSV file.
        CSV format: state,value
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        self.values = {}

        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                state = int(row['state'])
                value = float(row['value'])
                self.values[state] = value

    def __str__(self) -> str:
        """String representation of Value table."""
        result = "Value Table:\n"
        for state in sorted(self.values.keys()):
            result += f"State {state}: V={self.values[state]:.4f}\n"
        return result


class PolicyTable:
    """
    Policy table representation for MDP policies.
    Supports probabilistic policies (action probability distributions).
    Supports CSV import/export functionality.
    """

    def __init__(self, policy: Optional[Dict[int, Dict[int, float]]] = None):
        """
        Initialize Policy table.

        Args:
            policy: Dictionary mapping state -> {action: probability}
        """
        self.policy = policy if policy is not None else {}

    def get_action_probabilities(self, state: int) -> Dict[int, float]:
        """Get action probability distribution for a state."""
        return self.policy.get(state, {0: 1.0})

    def set_action_probabilities(self, state: int, action_probs: Dict[int, float]):
        """Set action probability distribution for a state."""
        # Normalize probabilities to sum to 1
        total_prob = sum(action_probs.values())
        if total_prob > 0:
            normalized_probs = {action: prob / total_prob for action, prob in action_probs.items()}
        else:
            # If all probabilities are zero, default to uniform distribution
            num_actions = len(action_probs)
            normalized_probs = {action: 1.0 / num_actions for action in action_probs.keys()}

        self.policy[state] = normalized_probs

    def get_action(self, state: int) -> int:
        """Get most likely action for a state (for compatibility)."""
        action_probs = self.get_action_probabilities(state)
        if not action_probs:
            return 0
        return max(action_probs, key=action_probs.get)

    def set_action(self, state: int, action: int):
        """Set deterministic action for a state (for compatibility)."""
        self.policy[state] = {action: 1.0}

    def sample_action(self, state: int, rng: np.random.Generator) -> int:
        """Sample an action according to the policy distribution."""
        action_probs = self.get_action_probabilities(state)
        actions = list(action_probs.keys())
        probabilities = list(action_probs.values())
        return rng.choice(actions, p=probabilities)

    def get_action_probability(self, state: int, action: int) -> float:
        """Get probability of taking a specific action in a state."""
        action_probs = self.get_action_probabilities(state)
        return action_probs.get(action, 0.0)

    def get_most_common_action(self) -> Tuple[int, int]:
        """
        Get the most common action and its count.

        Returns:
            Tuple of (most_common_action, count)
        """
        if not self.policy:
            return 0, 0

        action_counts = {}
        for state_policy in self.policy.values():
            most_likely_action = max(state_policy, key=state_policy.get)
            action_counts[most_likely_action] = action_counts.get(most_likely_action, 0) + 1

        most_common_action = max(action_counts, key=action_counts.get)
        count = action_counts[most_common_action]
        return most_common_action, count

    def get_all_states(self) -> List[int]:
        """Get all states in the Policy table."""
        return list(self.policy.keys())

    def get_all_actions(self) -> List[int]:
        """Get all unique actions in the Policy table."""
        all_actions = set()
        for state_policy in self.policy.values():
            all_actions.update(state_policy.keys())
        return list(all_actions)

    def export_to_csv(self, file_path: str):
        """
        Export Policy table to CSV file.
        CSV format: state,action,probability
        """
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['state', 'action', 'probability'])

            for state in sorted(self.policy.keys()):
                for action, prob in sorted(self.policy[state].items()):
                    writer.writerow([state, action, prob])

    def import_from_csv(self, file_path: str):
        """
        Import Policy table from CSV file.
        CSV format: state,action,probability
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        self.policy = {}

        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                state = int(row['state'])
                action = int(row['action'])
                probability = float(row['probability'])

                if state not in self.policy:
                    self.policy[state] = {}
                self.policy[state][action] = probability

    def __str__(self) -> str:
        """String representation of Policy table."""
        result = "Policy Table:\n"
        for state in sorted(self.policy.keys()):
            result += f"State {state}: "
            action_strs = []
            for action, prob in sorted(self.policy[state].items()):
                action_strs.append(f"A{action}={prob:.3f}")
            result += "{" + ", ".join(action_strs) + "}\n"
        return result


def q_table_to_policy(q_table: QTable,
                      states: List[int],
                      num_actions: int,
                      temperature: float = 1.0) -> PolicyTable:
    """
    Convert Q-table to policy using softmax (Boltzmann) distribution.

    Args:
        q_table: Q-table containing state-action values
        states: List of all states
        num_actions: Number of possible actions
        temperature: Temperature parameter for softmax
                    - temperature > 1: more exploration (more uniform)
                    - temperature = 1: standard softmax
                    - temperature < 1: more exploitation (more peaked)
                    - temperature → 0: approaches greedy policy

    Returns:
        PolicyTable with probabilistic action distributions
    """
    policy = PolicyTable()

    for state in states:
        # Get Q-values for all actions in this state
        q_values = []
        for action in range(num_actions):
            q_values.append(q_table.get_q_value(state, action))

        # Convert to numpy array for easier computation
        q_array = np.array(q_values)

        # Apply temperature scaling and softmax
        if temperature > 0:
            # Scale by temperature
            scaled_q = q_array / temperature
            # Subtract max for numerical stability
            scaled_q = scaled_q - np.max(scaled_q)
            # Compute softmax
            exp_q = np.exp(scaled_q)
            probabilities = exp_q / np.sum(exp_q)
        else:
            # Temperature = 0 means greedy policy
            probabilities = np.zeros(num_actions)
            best_action = np.argmax(q_values)
            probabilities[best_action] = 1.0

        # Create action probability dictionary
        action_probs = {}
        for action in range(num_actions):
            action_probs[action] = float(probabilities[action])

        policy.set_action_probabilities(state, action_probs)

    return policy
