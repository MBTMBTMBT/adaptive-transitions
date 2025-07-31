from typing import Dict, List, Tuple, Optional, Any
import csv
import os


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
    Supports CSV import/export functionality.
    """

    def __init__(self, policy: Optional[Dict[int, int]] = None):
        """
        Initialize Policy table.

        Args:
            policy: Dictionary mapping state -> action
        """
        self.policy = policy if policy is not None else {}

    def get_action(self, state: int) -> int:
        """Get action for a state."""
        return self.policy.get(state, 0)

    def set_action(self, state: int, action: int):
        """Set action for a state."""
        self.policy[state] = action

    def get_most_common_action(self) -> Tuple[int, int]:
        """
        Get the most common action and its count.

        Returns:
            Tuple of (most_common_action, count)
        """
        if not self.policy:
            return 0, 0

        action_counts = {}
        for action in self.policy.values():
            action_counts[action] = action_counts.get(action, 0) + 1

        most_common_action = max(action_counts, key=action_counts.get)
        count = action_counts[most_common_action]
        return most_common_action, count

    def get_all_states(self) -> List[int]:
        """Get all states in the Policy table."""
        return list(self.policy.keys())

    def get_all_actions(self) -> List[int]:
        """Get all unique actions in the Policy table."""
        return list(set(self.policy.values()))

    def export_to_csv(self, file_path: str):
        """
        Export Policy table to CSV file.
        CSV format: state,action
        """
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['state', 'action'])

            for state in sorted(self.policy.keys()):
                writer.writerow([state, self.policy[state]])

    def import_from_csv(self, file_path: str):
        """
        Import Policy table from CSV file.
        CSV format: state,action
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        self.policy = {}

        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                state = int(row['state'])
                action = int(row['action'])
                self.policy[state] = action

    def __str__(self) -> str:
        """String representation of Policy table."""
        result = "Policy Table:\n"
        for state in sorted(self.policy.keys()):
            result += f"State {state}: A={self.policy[state]}\n"
        return result
