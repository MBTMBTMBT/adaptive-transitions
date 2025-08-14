from typing import Dict, List, Tuple, Optional, Any, Union
import csv
import os
import scipy
import numpy as np

from mdp_network import MDPNetwork
from serialisable import Serialisable


class QTable(Serialisable):
    """
    Q-table representation for MDP Q-values.
    Supports CSV import/export functionality.
    Also implements Serialisable for in-memory, cross-process transport.
    """

    def __init__(self, q_values: Optional[Dict[int, Dict[int, float]]] = None):
        """
        Initialize Q-table.

        Args:
            q_values: Dictionary mapping state -> action -> q_value
        """
        self.q_values: Dict[int, Dict[int, float]] = q_values if q_values is not None else {}

    # -------- Serialisable interface --------
    def to_portable(self) -> Dict[str, Any]:
        """
        Build a JSON-serializable dict:
          {"q_values": {state: {action: q}}}
        Keys are kept as ints; values cast to float for JSON safety.
        """
        return {
            "q_values": {
                int(s): {int(a): float(v) for a, v in acts.items()}
                for s, acts in self.q_values.items()
            }
        }

    @classmethod
    def from_portable(cls, portable: Dict[str, Any]) -> "QTable":
        """
        Rebuild from dict produced by to_portable(). Tolerates string/int keys.
        """
        raw = portable.get("q_values", {})
        q_values: Dict[int, Dict[int, float]] = {}
        for s_k, acts in raw.items():
            s = int(s_k)
            q_values[s] = {int(a_k): float(v) for a_k, v in acts.items()}
        return cls(q_values=q_values)

    # clone() comes from Serialisable via round-trip

    # -------- QTable API --------
    def get_q_value(self, state: int, action: int) -> float:
        """Get Q-value for a state-action pair."""
        return self.q_values.get(state, {}).get(action, 0.0)

    def set_q_value(self, state: int, action: int, value: float):
        """Set Q-value for a state-action pair."""
        if state not in self.q_values:
            self.q_values[state] = {}
        self.q_values[state][action] = float(value)

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
        return best_action, float(best_value)

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

    def __str__(self, max_states: int = 20, max_actions_per_state: int = 10) -> str:
        """
        String representation of Q-table with flexible formatting for large dimensions.
        """
        if not self.q_values:
            return "Q-Table: Empty"

        states = sorted(self.q_values.keys())
        total_states = len(states)
        total_entries = sum(len(self.q_values[state]) for state in states)

        all_actions = set()
        for state_actions in self.q_values.values():
            all_actions.update(state_actions.keys())
        all_actions = sorted(all_actions)

        result = f"Q-Table ({total_states} states, {total_entries} entries):\n"

        if total_states <= max_states and len(all_actions) <= max_actions_per_state:
            header = "State".ljust(8)
            for action in all_actions:
                header += f"Action {action}".rjust(12)
            result += header + "\n"
            result += "-" * len(header) + "\n"

            for state in states:
                row = f"{state}".ljust(8)
                for action in all_actions:
                    value = self.q_values[state].get(action, 0.0)
                    row += f"{value:11.4f}".rjust(12)
                result += row + "\n"
        else:
            display_states = states[:max_states]
            for state in display_states:
                result += f"State {state}: "
                actions = sorted(self.q_values[state].items())
                if len(actions) <= max_actions_per_state:
                    action_strs = [f"A{action}={value:.4f}" for action, value in actions]
                else:
                    top_actions = sorted(actions, key=lambda x: x[1], reverse=True)
                    action_strs = [f"A{action}={value:.4f}" for action, value in top_actions[:max_actions_per_state]]
                    action_strs.append(f"... and {len(actions) - max_actions_per_state} more")
                result += ", ".join(action_strs) + "\n"

            if total_states > max_states:
                result += f"... and {total_states - max_states} more states\n"

        return result


class ValueTable(Serialisable):
    """
    Value table representation for MDP state values.
    Supports CSV import/export functionality.
    Also implements Serialisable.
    """

    def __init__(self, values: Optional[Dict[int, float]] = None):
        """
        Initialize Value table.

        Args:
            values: Dictionary mapping state -> value
        """
        self.values: Dict[int, float] = values if values is not None else {}

    # -------- Serialisable interface --------
    def to_portable(self) -> Dict[str, Any]:
        """
        Build a JSON-serializable dict:
          {"values": {state: value}}
        """
        return {"values": {int(s): float(v) for s, v in self.values.items()}}

    @classmethod
    def from_portable(cls, portable: Dict[str, Any]) -> "ValueTable":
        raw = portable.get("values", {})
        values = {int(s_k): float(v) for s_k, v in raw.items()}
        return cls(values=values)

    # clone() comes from Serialisable

    # -------- ValueTable API --------
    def get_value(self, state: int) -> float:
        """Get value for a state."""
        return float(self.values.get(state, 0.0))

    def set_value(self, state: int, value: float):
        """Set value for a state."""
        self.values[state] = float(value)

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
        return int(best_state), float(best_value)

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

    def __str__(self, max_states: int = 50, columns: int = 4) -> str:
        """
        String representation of Value table with flexible formatting for large dimensions.
        """
        if not self.values:
            return "Value Table: Empty"

        states = sorted(self.values.keys())
        total_states = len(states)
        result = f"Value Table ({total_states} states):\n"

        if total_states <= max_states:
            if total_states <= 20:
                for state in states:
                    result += f"State {state:3d}: V = {self.values[state]:8.4f}\n"
            else:
                result += "\n"
                for i in range(0, len(states), columns):
                    row_states = states[i:i + columns]
                    line = ""
                    for state in row_states:
                        line += f"S{state:3d}:{self.values[state]:7.4f}  "
                    result += line + "\n"
        else:
            display_states = states[:max_states]
            values_array = [self.values[state] for state in states]
            min_val = min(values_array)
            max_val = max(values_array)
            avg_val = sum(values_array) / len(values_array)
            result += f"Stats: Min={min_val:.4f}, Max={max_val:.4f}, Avg={avg_val:.4f}\n\n"
            result += "Sample states:\n"
            for i in range(0, len(display_states), columns):
                row_states = display_states[i:i + columns]
                line = ""
                for state in row_states:
                    line += f"S{state:3d}:{self.values[state]:7.4f}  "
                    result += line + "\n"
            if total_states > max_states:
                result += f"\n... and {total_states - max_states} more states"
        return result


class PolicyTable(Serialisable):
    """
    Policy table representation for MDP policies.
    Supports probabilistic policies (action probability distributions).
    Supports CSV import/export functionality.
    Also implements Serialisable.
    """

    def __init__(self, policy: Optional[Dict[int, Dict[int, float]]] = None):
        """
        Initialize Policy table.

        Args:
            policy: Dictionary mapping state -> {action: probability}
        """
        self.policy: Dict[int, Dict[int, float]] = policy if policy is not None else {}

    # -------- Serialisable interface --------
    def to_portable(self) -> Dict[str, Any]:
        """
        Build a JSON-serializable dict:
          {"policy": {state: {action: prob}}}
        """
        return {
            "policy": {
                int(s): {int(a): float(p) for a, p in acts.items()}
                for s, acts in self.policy.items()
            }
        }

    @classmethod
    def from_portable(cls, portable: Dict[str, Any]) -> "PolicyTable":
        raw = portable.get("policy", {})
        policy: Dict[int, Dict[int, float]] = {}
        for s_k, acts in raw.items():
            s = int(s_k)
            policy[s] = {int(a_k): float(p) for a_k, p in acts.items()}
        return cls(policy=policy)

    # clone() comes from Serialisable

    # -------- PolicyTable API --------
    def get_action_probabilities(self, state: int) -> Dict[int, float]:
        """Get action probability distribution for a state."""
        return self.policy.get(state, {0: 1.0})

    def set_action_probabilities(self, state: int, action_probs: Dict[int, float]):
        """Set action probability distribution for a state (normalized)."""
        total_prob = float(sum(action_probs.values()))
        if total_prob > 0:
            normalized_probs = {int(a): float(p) / total_prob for a, p in action_probs.items()}
        else:
            num_actions = max(1, len(action_probs))
            normalized_probs = {int(a): 1.0 / num_actions for a in action_probs.keys()}
        self.policy[state] = normalized_probs

    def get_action(self, state: int) -> int:
        """Get most likely action for a state (for compatibility)."""
        action_probs = self.get_action_probabilities(state)
        if not action_probs:
            return 0
        return max(action_probs, key=action_probs.get)

    def set_action(self, state: int, action: int):
        """Set deterministic action for a state (for compatibility)."""
        self.policy[state] = {int(action): 1.0}

    def sample_action(self, state: int, rng: np.random.Generator) -> int:
        """Sample an action according to the policy distribution."""
        action_probs = self.get_action_probabilities(state)
        actions = list(action_probs.keys())
        probabilities = list(action_probs.values())
        return int(rng.choice(actions, p=probabilities))

    def get_action_probability(self, state: int, action: int) -> float:
        """Get probability of taking a specific action in a state."""
        action_probs = self.get_action_probabilities(state)
        return float(action_probs.get(action, 0.0))

    def get_most_common_action(self) -> Tuple[int, int]:
        """
        Get the most common action and its count.
        Returns: (most_common_action, count)
        """
        if not self.policy:
            return 0, 0
        action_counts: Dict[int, int] = {}
        for state_policy in self.policy.values():
            if not state_policy:
                continue
            most_likely_action = max(state_policy, key=state_policy.get)
            action_counts[most_likely_action] = action_counts.get(most_likely_action, 0) + 1
        if not action_counts:
            return 0, 0
        most_common_action = max(action_counts, key=action_counts.get)
        count = action_counts[most_common_action]
        return int(most_common_action), int(count)

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

    def __str__(self, max_states: int = 30, show_deterministic_only: bool = False) -> str:
        """
        String representation of Policy table with flexible formatting for large dimensions.
        """
        if not self.policy:
            return "Policy Table: Empty"

        states = sorted(self.policy.keys())
        total_states = len(states)

        deterministic_count = 0
        for state_policy in self.policy.values():
            max_prob = max(state_policy.values()) if state_policy else 0.0
            if max_prob >= 0.99:
                deterministic_count += 1

        result = f"Policy Table ({total_states} states, {deterministic_count} deterministic):\n"

        if show_deterministic_only or (total_states > 20 and deterministic_count / max(total_states, 1) > 0.8):
            result += "\nDeterministic actions (probability >= 0.99):\n"
            display_states = states[:max_states]
            action_groups: Dict[int, List[int]] = {}
            for state in display_states:
                state_policy = self.policy[state]
                if not state_policy:
                    continue
                best_action = max(state_policy, key=state_policy.get)
                best_prob = state_policy[best_action]
                if best_prob >= 0.99:
                    action_groups.setdefault(best_action, []).append(state)

            for action in sorted(action_groups.keys()):
                states_list = action_groups[action]
                if len(states_list) <= 10:
                    result += f"Action {action}: States {states_list}\n"
                else:
                    result += f"Action {action}: States {states_list[:10]} ... and {len(states_list) - 10} more\n"

            non_det_states = []
            for state in display_states:
                state_policy = self.policy[state]
                max_prob = max(state_policy.values()) if state_policy else 0.0
                if max_prob < 0.99:
                    non_det_states.append(state)

            if non_det_states:
                result += f"\nNon-deterministic states ({len(non_det_states)}): "
                if len(non_det_states) <= 10:
                    result += str(non_det_states) + "\n"
                else:
                    result += f"{non_det_states[:10]} ... and {len(non_det_states) - 10} more\n"
        else:
            display_states = states[:max_states]
            for state in display_states:
                result += f"State {state}: "
                state_policy = self.policy[state]
                sorted_actions = sorted(state_policy.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_actions) <= 5:
                    action_strs = [f"A{action}={prob:.3f}" for action, prob in sorted_actions]
                else:
                    action_strs = [f"A{action}={prob:.3f}" for action, prob in sorted_actions[:5]]
                    action_strs.append(f"... +{len(sorted_actions) - 5} more")
                result += "{" + ", ".join(action_strs) + "}\n"

        if total_states > max_states:
            result += f"\n... and {total_states - max_states} more states"

        return result


def create_random_policy(mdp_network: MDPNetwork) -> PolicyTable:
    """Create a uniform probabilistic policy for the MDP."""
    policy = PolicyTable()
    for state in mdp_network.states:
        if mdp_network.is_terminal_state(state):
            # Terminal states have deterministic action 0
            policy.set_action(state, 0)
        else:
            # Create uniform probability distribution over actions
            uniform_prob = 1.0 / mdp_network.num_actions

            action_probs = {}
            for action in range(mdp_network.num_actions):
                action_probs[action] = uniform_prob

            policy.set_action_probabilities(state, action_probs)

    return policy


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


def blend_policies(target: PolicyTable,
                   prior: PolicyTable,
                   weight: float) -> PolicyTable:
    """
    Blend two policies using linear interpolation.

    Args:
        target: Target policy (weight=0 returns this)
        prior: Prior policy (weight=1 returns this)
        weight: Blend weight [0,1]. 0=target, 1=prior

    Returns:
        Blended policy: (1-weight)*target + weight*prior
    """
    if not (0.0 <= weight <= 1.0):
        raise ValueError("Weight must be in [0,1]")

    # Get all states from both policies
    all_states = set(target.get_all_states()) | set(prior.get_all_states())
    all_actions = set(target.get_all_actions()) | set(prior.get_all_actions())

    blended = PolicyTable()

    for state in all_states:
        target_probs = target.get_action_probabilities(state)
        prior_probs = prior.get_action_probabilities(state)

        # Blend probabilities for each action
        blended_probs = {}
        for action in all_actions:
            target_prob = target_probs.get(action, 0.0)
            prior_prob = prior_probs.get(action, 0.0)
            blended_probs[action] = (1 - weight) * target_prob + weight * prior_prob

        # Remove zero probabilities for cleaner representation
        blended_probs = {a: p for a, p in blended_probs.items() if p > 0.0}

        if blended_probs:  # Only set if non-empty
            blended.set_action_probabilities(state, blended_probs)

    return blended


class RewardDistributionTable:
    """
    Reward distribution table representation for MDP reward analysis.
    Supports both count-based and probability-based distributions.
    Handles float reward values with configurable precision delta.
    """

    def __init__(self, values: Optional[Dict[float, Union[int, float]]] = None, delta: float = 0.01):
        """
        Initialize Reward Distribution table.

        Args:
            values: Dictionary mapping reward -> count/probability
            delta: Precision delta for grouping similar reward values
        """
        self.values = values if values is not None else {}
        self.delta = delta

    def _round_reward(self, reward: float) -> float:
        """
        Round reward value to the nearest multiple of delta for grouping.

        Args:
            reward: Raw reward value

        Returns:
            Rounded reward value for use as dictionary key
        """
        if self.delta <= 0:
            return reward
        return round(reward / self.delta) * self.delta

    def add_count(self, reward: float, count: Union[int, float]):
        """
        Add count for a reward value.

        Args:
            reward: Reward value
            count: Count or probability to add
        """
        rounded_reward = self._round_reward(reward)
        if rounded_reward in self.values:
            self.values[rounded_reward] += count
        else:
            self.values[rounded_reward] = count

    def get_value(self, reward: float) -> Union[int, float]:
        """
        Get count/probability for a reward value.

        Args:
            reward: Reward value

        Returns:
            Count or probability for the reward
        """
        rounded_reward = self._round_reward(reward)
        return self.values.get(rounded_reward, 0.0)

    def set_value(self, reward: float, value: Union[int, float]):
        """
        Set count/probability for a reward value.

        Args:
            reward: Reward value
            value: Count or probability to set
        """
        rounded_reward = self._round_reward(reward)
        self.values[rounded_reward] = value

    def get_total_count(self) -> Union[int, float]:
        """
        Get total count/probability across all rewards.

        Returns:
            Sum of all values in the distribution
        """
        return sum(self.values.values())

    def get_most_frequent_reward(self) -> Tuple[float, Union[int, float]]:
        """
        Get the reward with highest count/probability and its value.

        Returns:
            Tuple of (most_frequent_reward, highest_count)
        """
        if not self.values:
            return 0.0, 0.0

        best_reward = max(self.values, key=self.values.get)
        best_count = self.values[best_reward]
        return best_reward, best_count

    def get_all_rewards(self) -> List[float]:
        """Get all reward values in the distribution table."""
        return list(self.values.keys())

    def normalize_rewards_to_0_1(self) -> 'RewardDistributionTable':
        """
        Normalize the REWARD KEYS to 0-1 range using min-max normalization.
        The values (counts/probabilities) remain unchanged, only reward keys are normalized.

        Returns:
            New RewardDistributionTable with reward keys normalized to [0, 1] range.
            Values and delta precision remain unchanged.

        Example:
            Original: {-5.0: 10, 0.0: 50, 10.0: 30}  # rewards -> counts
            Result:   {0.0: 10, 0.33: 50, 1.0: 30}   # normalized rewards -> same counts

        Note:
            Uses formula: normalized_reward = (reward - min_reward) / (max_reward - min_reward)
            If all rewards are identical, returns table with single reward key 0.5
        """
        if not self.values:
            return RewardDistributionTable(delta=self.delta)

        # Get min and max REWARDS (not values) for normalization
        all_rewards = list(self.values.keys())  # These are the reward keys
        min_reward = min(all_rewards)
        max_reward = max(all_rewards)

        # Handle edge case where all rewards are identical
        if max_reward == min_reward:
            # All rewards are the same, set to middle of [0,1] range
            total_value = sum(self.values.values())
            normalized_values = {0.5: total_value}
            print(f"All REWARDS identical ({min_reward:.6f}), normalized to 0.5")
        else:
            # Apply min-max normalization to REWARDS: (reward - min) / (max - min)
            reward_range = max_reward - min_reward
            normalized_values = {}

            for reward, value in self.values.items():
                normalized_reward = (reward - min_reward) / reward_range
                # Round to delta precision
                normalized_reward = self._round_reward(normalized_reward)

                # Handle potential collisions from rounding
                if normalized_reward in normalized_values:
                    normalized_values[normalized_reward] += value
                else:
                    normalized_values[normalized_reward] = value

            print(f"REWARDS 0-1 Normalization: [{min_reward:.6f}, {max_reward:.6f}] -> [0.0, 1.0]")
            print("Values (counts/probabilities) unchanged, only reward keys normalized")

        # Create new table with same delta and normalized reward keys
        return RewardDistributionTable(normalized_values, delta=self.delta)

    def normalize_to_probabilities(self) -> 'RewardDistributionTable':
        """
        Convert count-based distribution to probability distribution.

        Returns:
            New RewardDistributionTable with normalized probabilities
        """
        total_count = self.get_total_count()
        if total_count == 0:
            return RewardDistributionTable(delta=self.delta)

        prob_values = {reward: count / total_count for reward, count in self.values.items()}
        return RewardDistributionTable(prob_values, delta=self.delta)

    def _is_probability_distribution(self, tolerance: float = 1e-6) -> bool:
        """
        Check if the distribution values represent probabilities (sum ≈ 1).

        Args:
            tolerance: Tolerance for checking if sum equals 1

        Returns:
            True if this appears to be a probability distribution
        """
        total = self.get_total_count()
        return abs(total - 1.0) <= tolerance

    def generate_samples(self,
                         num_samples: int = 10000,
                         force_count: Optional[bool] = None,
                         min_samples_per_reward: int = 1) -> np.ndarray:
        """
        Generate samples from the reward distribution for statistical analysis.

        Args:
            num_samples: Target number of samples to generate (used for probability distributions)
            force_count: Force interpretation as count (True) or probability (False).
                        If None, auto-detect based on whether sum ≈ 1
            min_samples_per_reward: Minimum samples per reward value (for count distributions)

        Returns:
            numpy array of reward samples
        """
        if not self.values:
            return np.array([])

        # Determine distribution type
        if force_count is None:
            is_count = not self._is_probability_distribution()
        else:
            is_count = force_count

        samples = []

        if is_count:
            # Count distribution: interpret counts as actual sample frequencies
            print("Using exact count distribution (no sampling needed)")

            for reward, count in self.values.items():
                # For fractional counts, use probabilistic rounding
                integer_count = int(count)
                fractional_part = count - integer_count

                # Add the integer part
                samples.extend([reward] * integer_count)

                # Add one more sample with probability equal to fractional part
                if fractional_part > 0 and np.random.random() < fractional_part:
                    samples.append(reward)

                # Ensure minimum samples per reward
                if len([s for s in samples if s == reward]) < min_samples_per_reward:
                    needed = min_samples_per_reward - len([s for s in samples if s == reward])
                    samples.extend([reward] * needed)

        else:
            # Probability distribution: generate samples proportional to probabilities
            print(f"Generating {num_samples} samples from probability distribution")

            rewards = list(self.values.keys())
            probabilities = list(self.values.values())

            # Generate samples using multinomial sampling
            sample_counts = np.random.multinomial(num_samples, probabilities)

            for reward, count in zip(rewards, sample_counts):
                samples.extend([reward] * count)

        print(f"Generated {len(samples)} samples from {len(self.values)} unique reward values")
        return np.array(samples)

    def fit_gaussian(self,
                     num_samples: int = 10000,
                     force_count: Optional[bool] = None,
                     min_samples_per_reward: int = 1) -> Tuple[float, float, Dict[str, float]]:
        """
        Fit a Gaussian distribution to the reward distribution.

        Args:
            num_samples: Target number of samples for probability distributions
            force_count: Force interpretation as count (True) or probability (False)
            min_samples_per_reward: Minimum samples per reward value for count distributions

        Returns:
            Tuple of (mu, sigma, fit_statistics)
            - mu: fitted mean
            - sigma: fitted standard deviation
            - fit_statistics: dictionary with fitting statistics and diagnostics
        """
        if not self.values:
            return 0.0, 0.0, {'error': 'Empty distribution'}

        # Determine distribution type
        is_count = force_count if force_count is not None else not self._is_probability_distribution()

        # Common fit statistics initialization
        fit_stats = {
            'unique_rewards': len(self.values),
            'reward_range': (min(self.values.keys()), max(self.values.keys()))
        }

        if is_count:
            # Direct calculation from counts
            print("Fitting Gaussian directly from count distribution")

            total_count = sum(self.values.values())
            if total_count == 0:
                return 0.0, 0.0, {'error': 'Zero total count'}

            # Calculate weighted mean and variance
            weighted_sum = sum(reward * count for reward, count in self.values.items())
            mu = weighted_sum / total_count

            weighted_var_sum = sum(count * (reward - mu) ** 2 for reward, count in self.values.items())
            variance = weighted_var_sum / total_count
            sigma = np.sqrt(variance)

            # Update fit statistics
            fit_stats.update({
                'method': 'direct_from_counts',
                'total_count': total_count,
                'weighted_mean': mu,
                'weighted_variance': variance,
                'fitted_mu': mu,
                'fitted_sigma': sigma
            })

            # Generate samples for KS test
            samples = self.generate_samples(num_samples=None, force_count=True,
                                            min_samples_per_reward=min_samples_per_reward)
            if len(samples) > 1:
                fit_stats.update({
                    'num_samples': len(samples),
                    'sample_mean': np.mean(samples),
                    'sample_std': np.std(samples, ddof=1)
                })

                # KS test
                ks_statistic, ks_p_value = scipy.stats.kstest(samples,
                                                              lambda x: scipy.stats.norm.cdf(x, mu, sigma))
                fit_stats.update({
                    'ks_statistic': ks_statistic,
                    'ks_p_value': ks_p_value,
                    'ks_significant': ks_p_value < 0.05
                })

            print(f"Gaussian fit completed (direct): μ={mu:.6f}, σ={sigma:.6f}")
            print(f"  Total count: {total_count:.1f}, KS p-value: {fit_stats.get('ks_p_value', 'N/A')}")

        else:
            # Sample-based fitting for probability distributions
            print(f"Fitting Gaussian from {num_samples} samples")

            samples = self.generate_samples(num_samples, force_count, min_samples_per_reward)
            if len(samples) == 0:
                return 0.0, 0.0, {'error': 'No samples generated'}

            # Fit using scipy
            mu, sigma = scipy.stats.norm.fit(samples)

            # Update fit statistics
            empirical_mean = np.mean(samples)
            empirical_var = np.var(samples, ddof=1)

            fit_stats.update({
                'method': 'scipy_fit_from_samples',
                'num_samples': len(samples),
                'sample_mean': empirical_mean,
                'sample_std': np.std(samples, ddof=1),
                'fitted_mu': mu,
                'fitted_sigma': sigma,
                'mean_difference': abs(mu - empirical_mean),
                'variance_ratio': (sigma ** 2) / empirical_var if empirical_var > 0 else float('inf')
            })

            # KS test
            ks_statistic, ks_p_value = scipy.stats.kstest(samples,
                                                          lambda x: scipy.stats.norm.cdf(x, mu, sigma))
            fit_stats.update({
                'ks_statistic': ks_statistic,
                'ks_p_value': ks_p_value,
                'ks_significant': ks_p_value < 0.05
            })

            print(f"Gaussian fit completed (sampled): μ={mu:.6f}, σ={sigma:.6f}")
            print(f"  Samples: {len(samples)}, KS p-value: {ks_p_value:.6f}")

        return mu, sigma, fit_stats

    def export_to_csv(self, file_path: str):
        """
        Export Reward Distribution table to CSV file.
        CSV format: reward,count_or_probability
        """
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['reward', 'count_or_probability'])

            for reward in sorted(self.values.keys()):
                writer.writerow([reward, self.values[reward]])

    def import_from_csv(self, file_path: str):
        """
        Import Reward Distribution table from CSV file.
        CSV format: reward,count_or_probability
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        self.values = {}

        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                reward = float(row['reward'])
                value = float(row['count_or_probability'])
                # Use add_count to ensure proper rounding
                self.add_count(reward, value)

    def __str__(self, max_rewards: int = 20, precision: int = 4) -> str:
        """
        String representation of Reward Distribution table.

        Args:
            max_rewards: Maximum number of rewards to display
            precision: Number of decimal places for display
        """
        if not self.values:
            return "Reward Distribution Table: Empty"

        rewards = sorted(self.values.keys())
        total_rewards = len(rewards)
        total_count = self.get_total_count()

        result = f"Reward Distribution Table ({total_rewards} unique rewards, total: {total_count:.{precision}f}):\n"
        result += f"Delta precision: {self.delta:.2e}\n\n"

        # Show statistics
        if rewards:
            min_reward = min(rewards)
            max_reward = max(rewards)

            # Calculate weighted average reward
            weighted_sum = sum(reward * count for reward, count in self.values.items())
            avg_reward = weighted_sum / total_count if total_count > 0 else 0.0

            result += f"Stats: Min Reward={min_reward:.{precision}f}, Max Reward={max_reward:.{precision}f}, "
            result += f"Weighted Avg={avg_reward:.{precision}f}\n\n"

        # Display rewards and their counts/probabilities
        display_rewards = rewards[:max_rewards]

        for reward in display_rewards:
            count = self.values[reward]
            percentage = (count / total_count * 100) if total_count > 0 else 0.0
            result += f"Reward {reward:>{precision + 3}.{precision}f}: {count:>{precision + 3}.{precision}f} ({percentage:5.1f}%)\n"

        if total_rewards > max_rewards:
            result += f"\n... and {total_rewards - max_rewards} more reward values"

        return result
