from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .mdp_tables import QTable, ValueTable, PolicyTable, RewardDistributionTable
from .mdp_network import MDPNetwork


def policy_evaluation(mdp_network: MDPNetwork,
                      policy: PolicyTable,
                      gamma: float = 0.99,
                      theta: float = 1e-6,
                      max_iterations: int = 1000,
                      verbose: bool = False) -> ValueTable:
    """
    Evaluate V^π using R(s,a,s').
    Set verbose=True to print progress.
    """
    value_table = ValueTable()
    states = mdp_network.states
    for s in states:
        value_table.set_value(s, 0.0)

    if verbose:
        print(f"Policy evaluation started: {len(states)} states, gamma={gamma}, theta={theta}")

    for it in range(max_iterations):
        max_delta = 0.0
        for s in states:
            if mdp_network.is_terminal_state(s):
                value_table.set_value(s, 0.0)
                continue

            old_v = value_table.get_value(s)
            new_v = 0.0
            action_probs = policy.get_action_probabilities(s)

            for a, pi_sa in action_probs.items():
                if pi_sa <= 0:
                    continue
                # Sum over s': P * (R + gamma * V)
                sa_val = 0.0
                trans = mdp_network.get_transition_probabilities(s, a)
                if not trans:
                    # fallback: self-loop with default reward
                    r = mdp_network.default_reward
                    sa_val = r + gamma * value_table.get_value(s)
                else:
                    for sp, p in trans.items():
                        r = mdp_network.get_transition_reward(s, a, sp)
                        sa_val += p * (r + gamma * value_table.get_value(sp))
                new_v += pi_sa * sa_val

            value_table.set_value(s, new_v)
            max_delta = max(max_delta, abs(new_v - old_v))

        if verbose and (it + 1) % 100 == 0:
            print(f"  Policy evaluation iteration {it + 1}: max_change = {max_delta:.6f}")
        if max_delta < theta:
            if verbose:
                print(f"Policy evaluation converged after {it + 1} iterations, final max_change = {max_delta:.6f}")
            break
    else:
        if verbose:
            print(f"Policy evaluation reached maximum iterations ({max_iterations}), final max_change = {max_delta:.6f}")

    return value_table


def optimal_value_iteration(mdp_network: MDPNetwork,
                            gamma: float = 0.99,
                            theta: float = 1e-6,
                            max_iterations: int = 1000,
                            verbose: bool = False) -> Tuple[ValueTable, QTable]:
    """
    Compute V* and Q* using R(s,a,s').
    Set verbose=True to print progress.
    """
    value_table = ValueTable()
    q_table = QTable()

    states = mdp_network.states
    A = mdp_network.num_actions

    for s in states:
        value_table.set_value(s, 0.0)
        for a in range(A):
            q_table.set_q_value(s, a, 0.0)

    if verbose:
        print(f"Optimal value iteration started: {len(states)} states, {A} actions, gamma={gamma}, theta={theta}")

    for it in range(max_iterations):
        max_delta = 0.0
        for s in states:
            if mdp_network.is_terminal_state(s):
                value_table.set_value(s, 0.0)
                for a in range(A):
                    q_table.set_q_value(s, a, 0.0)
                continue

            old_v = value_table.get_value(s)
            action_vals = []

            for a in range(A):
                q = 0.0
                trans = mdp_network.get_transition_probabilities(s, a)
                if not trans:
                    r = mdp_network.default_reward
                    q = r + gamma * value_table.get_value(s)
                else:
                    for sp, p in trans.items():
                        r = mdp_network.get_transition_reward(s, a, sp)
                        q += p * (r + gamma * value_table.get_value(sp))
                q_table.set_q_value(s, a, q)
                action_vals.append(q)

            new_v = max(action_vals) if action_vals else 0.0
            value_table.set_value(s, new_v)
            max_delta = max(max_delta, abs(new_v - old_v))

        if verbose and (it + 1) % 100 == 0:
            print(f"  Optimal value iteration {it + 1}: max_change = {max_delta:.6f}")
        if max_delta < theta:
            if verbose:
                print(f"Optimal value iteration converged after {it + 1} iterations, final max_change = {max_delta:.6f}")
            break
    else:
        if verbose:
            print(f"Optimal value iteration reached maximum iterations ({max_iterations}), final max_change = {max_delta:.6f}")

    return value_table, q_table


def q_learning(mdp_network: MDPNetwork,
               alpha: float = 0.1,
               gamma: float = 0.99,
               epsilon: float = 0.1,
               num_episodes: int = 10000,
               max_steps_per_episode: int = 1000,
               seed: Optional[int] = None,
               verbose: bool = False) -> Tuple[QTable, ValueTable]:
    """
    Tabular Q-learning with R(s,a,s') via mdp.sample_step.
    Set verbose=True to print progress.
    """
    rng = np.random.default_rng(seed)
    q_table = QTable()
    value_table = ValueTable()

    states = mdp_network.states
    A = mdp_network.num_actions

    for s in states:
        value_table.set_value(s, 0.0)
        for a in range(A):
            q_table.set_q_value(s, a, 0.0)

    if verbose:
        print(f"Q-Learning started: {len(states)} states, {A} actions, {num_episodes} episodes")
        print(f"  Parameters: alpha={alpha}, gamma={gamma}, epsilon={epsilon}")

    for ep in range(num_episodes):
        s = mdp_network.sample_start_state(rng)
        episode_steps = 0

        for _ in range(max_steps_per_episode):
            if mdp_network.is_terminal_state(s):
                break
            episode_steps += 1

            # epsilon-greedy
            if rng.random() < epsilon:
                a = int(rng.integers(0, A))
            else:
                a, _ = q_table.get_best_action(s)

            # one-step transition and reward
            sp, r = mdp_network.sample_step(s, a, rng)

            # TD update
            q_sa = q_table.get_q_value(s, a)
            if mdp_network.is_terminal_state(sp):
                target = r
            else:
                _, max_next_q = q_table.get_best_action(sp)
                target = r + gamma * max_next_q
            q_table.set_q_value(s, a, q_sa + alpha * (target - q_sa))

            s = sp

        if verbose and (ep + 1) % 1000 == 0:
            total_q = sum(q_table.get_q_value(s, a) for s in states for a in range(A))
            avg_q = total_q / (len(states) * A)
            print(f"  Q-Learning episode {ep + 1}/{num_episodes}: avg_q_value = {avg_q:.4f}, last_episode_steps = {episode_steps}")

    # derive V(s) = max_a Q(s,a)
    for s in states:
        if mdp_network.is_terminal_state(s):
            value_table.set_value(s, 0.0)
        else:
            _, best_q = q_table.get_best_action(s)
            value_table.set_value(s, best_q)

    if verbose:
        print("Q-Learning completed")
    return q_table, value_table


def compute_occupancy_measure(mdp_network: MDPNetwork,
                              policy: PolicyTable,
                              gamma: float = 0.99,
                              theta: float = 1e-6,
                              max_iterations: int = 1000,
                              verbose: bool = False) -> ValueTable:
    """
    Compute occupancy measure for a given policy in MDP (supports probabilistic policies).
    Returns a ValueTable containing the expected cumulative frequency of visiting each state.
    Set verbose=True to print progress.
    """
    # Initialize occupancy measure table
    occupancy_table = ValueTable()

    # Get all states
    states = mdp_network.states

    # Initialize occupancy measures to zero
    for state in states:
        occupancy_table.set_value(state, 0.0)

    # Initialize current state distribution (uniform over start states)
    current_distribution = {}
    start_states = list(mdp_network.start_states)

    if not start_states:
        if verbose:
            print("Warning: No start states found")
        return occupancy_table

    # Uniform initial distribution over start states
    initial_prob = 1.0 / len(start_states)
    for state in start_states:
        current_distribution[state] = initial_prob

    if verbose:
        print(f"Occupancy measure computation started: {len(states)} states, {len(start_states)} start states")
        print(f"  Parameters: gamma={gamma}, theta={theta}")

    # Iterative computation of occupancy measures
    for iteration in range(max_iterations):
        max_change = 0.0
        next_distribution = {state: 0.0 for state in states}

        # For each state in current distribution
        for state, state_prob in current_distribution.items():
            if state_prob <= 0:
                continue

            # Update occupancy measure for this state (including terminal states)
            old_occupancy = occupancy_table.get_value(state)
            new_occupancy = old_occupancy + state_prob
            occupancy_table.set_value(state, new_occupancy)

            # Track maximum change for convergence
            change = abs(new_occupancy - old_occupancy)
            max_change = max(max_change, change)

            # Terminal states don't propagate probability to next states
            if mdp_network.is_terminal_state(state):
                continue

            # Get action probabilities from policy
            action_probs = policy.get_action_probabilities(state)

            # For each possible action
            for action, action_prob in action_probs.items():
                if action_prob <= 0:
                    continue

                # Get transition probabilities for this state-action pair
                transition_probs = mdp_network.get_transition_probabilities(state, action)

                if not transition_probs:
                    # No transitions defined, assume staying in same state
                    transition_probs = {state: 1.0}

                # Propagate probability to next states with discount factor
                for next_state, transition_prob in transition_probs.items():
                    # Discounted probability of reaching next state
                    propagated_prob = state_prob * action_prob * transition_prob * gamma
                    next_distribution[next_state] += propagated_prob

        # Update current distribution for next iteration
        current_distribution = next_distribution

        # Print debug info every 100 iterations
        if verbose and (iteration + 1) % 100 == 0:
            total_occupancy = sum(occupancy_table.get_value(state) for state in states)
            print(f"  Occupancy measure iteration {iteration + 1}: max_change = {max_change:.6f}, total_occupancy = {total_occupancy:.4f}")

        # Check for convergence
        if max_change < theta:
            if verbose:
                total_occupancy = sum(occupancy_table.get_value(state) for state in states)
                print(f"Occupancy measure computation converged after {iteration + 1} iterations")
                print(f"  Final: max_change = {max_change:.6f}, total_occupancy = {total_occupancy:.4f}")
            break
    else:
        if verbose:
            total_occupancy = sum(occupancy_table.get_value(state) for state in states)
            print(f"Occupancy measure computation reached maximum iterations ({max_iterations})")
            print(f"  Final: max_change = {max_change:.6f}, total_occupancy = {total_occupancy:.4f}")

    # Terminal states keep their occupancy; no zeroing.

    return occupancy_table


def compute_reward_distribution(mdp_network: MDPNetwork,
                                occupancy_table: ValueTable,
                                policy: PolicyTable,
                                delta: float = 0.01,
                                verbose: bool = False) -> Tuple[RewardDistributionTable, RewardDistributionTable]:
    """
    Compute distribution over transition rewards under (occupancy, policy).
    Count for reward r is: sum_s Occ(s) * sum_a pi(a|s) * sum_{s'} P(s'|s,a) * 1[r_{s,a,s'}=r].
    Set verbose=True to print progress.
    """
    if verbose:
        print(f"Computing reward distribution with delta precision: {delta:.2e}")

    count_dist = RewardDistributionTable(delta=delta)
    states = mdp_network.states
    processed_states = 0

    for s in states:
        occ = occupancy_table.get_value(s)
        if occ <= 0:
            continue
        processed_states += 1

        # terminal states contribute occupancy but have no outgoing transitions
        if mdp_network.is_terminal_state(s):
            continue

        action_probs = policy.get_action_probabilities(s)
        for a, pi_sa in action_probs.items():
            if pi_sa <= 0:
                continue
            trans = mdp_network.get_transition_probabilities(s, a)
            if not trans:
                # no outgoing: self-loop with default reward
                r = mdp_network.default_reward
                count_dist.add_count(r, occ * pi_sa)  # weight fully to that reward
            else:
                for sp, p in trans.items():
                    r = mdp_network.get_transition_reward(s, a, sp)
                    count_dist.add_count(r, occ * pi_sa * p)

    if verbose:
        print(f"Processed {processed_states} states with positive occupancy")
    prob_dist = count_dist.normalize_to_probabilities()

    if verbose:
        total = count_dist.get_total_count()
        uniq = len(count_dist.get_all_rewards())
        print(f"Reward distribution computed:\n  Total occupancy-weighted transitions: {total:.6f}\n  Unique rewards: {uniq}")
        if uniq > 0:
            mr, mc = count_dist.get_most_frequent_reward()
            print(f"  Most frequent reward: {mr:.6f} (weight: {mc:.6f})")

    return count_dist, prob_dist
