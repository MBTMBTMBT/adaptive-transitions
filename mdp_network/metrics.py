from typing import Dict, Tuple, List

import numpy as np

from mdp_network import MDPNetwork
from mdp_network.solvers import policy_evaluation
from mdp_network.mdp_tables import PolicyTable, ValueTable, blend_policies, QTable


def kl_policies(
        prior_policy: PolicyTable,
        target_policy: PolicyTable,
        mdp_network: MDPNetwork,
        gamma: float = 0.99,
        theta: float = 1e-6,
        max_iterations: int = 1000,
    ) -> float:
    """
    Compute discounted expected sum of per-state KL(target || prior) under the target policy.
    Returns the uniform average of V(s) over mdp_network.start_states.

    Notes
    -----
    - Per-state immediate "reward": r(s) = KL( target(.|s) || prior(.|s) ).
      Defined as sum_a p_t(a) * log(p_t(a) / p_p(a)); 0 * log(0/q) := 0.
      If p_p(a) == 0 while p_t(a) > 0, r(s) = +inf.
    - Value iteration style policy evaluation with target_policy:
        V(s) = r(s) + gamma * E_{a~pi_t, s'~P(·|s,a)}[ V(s') ]
    - Terminal states are fixed to V(s)=0 and do not accrue KL reward.
    """
    states = mdp_network.states
    A = mdp_network.num_actions
    start_states = list(mdp_network.start_states)

    if not states or not start_states:
        return 0.0

    # Precompute per-state KL rewards
    kl_reward: Dict[int, float] = {}
    for s in states:
        if mdp_network.is_terminal_state(s):
            kl_reward[s] = 0.0
            continue

        tgt = target_policy.get_action_probabilities(s)
        pri = prior_policy.get_action_probabilities(s)

        # KL(target || prior) with standard conventions:
        # Only actions with p_t > 0 contribute; if p_p == 0 while p_t > 0 -> +inf.
        kl = 0.0
        inf_flag = False
        for a in range(A):
            pt = float(tgt.get(a, 0.0))
            if pt <= 0.0:
                continue  # 0 * log(0/q) := 0
            pq = float(pri.get(a, 0.0))
            if pq <= 0.0:
                inf_flag = True
                break
            kl += pt * (np.log(pt) - np.log(pq))

        kl_reward[s] = float("inf") if inf_flag else float(kl)

    # If any reachable state's KL is +inf, the value becomes +inf for its ancestors.
    # We still run VI; it will propagate +inf forward. (Fast exit not strictly needed.)

    # Initialize values
    V: Dict[int, float] = {s: 0.0 for s in states}

    # Value iteration under target policy with state-reward kl_reward[s]
    for _ in range(max_iterations):
        max_delta = 0.0
        for s in states:
            if mdp_network.is_terminal_state(s):
                V[s] = 0.0
                continue

            old_v = V[s]
            # Expected next value under the target policy
            exp_next = 0.0
            action_probs = target_policy.get_action_probabilities(s)
            for a, pi_sa in action_probs.items():
                if pi_sa <= 0.0:
                    continue
                trans = mdp_network.get_transition_probabilities(s, a)
                if not trans:
                    # Fallback: self-loop if no explicit transitions
                    exp_next += pi_sa * V[s]
                else:
                    for sp, p in trans.items():
                        exp_next += pi_sa * p * V[sp]

            # Bellman update with state-only reward
            new_v = kl_reward[s] + gamma * exp_next
            V[s] = new_v
            # Track largest absolute change; handles inf naturally
            delta = (
                abs(new_v - old_v)
                if np.isfinite(new_v) and np.isfinite(old_v)
                else (0.0 if (np.isinf(new_v) and np.isinf(old_v)) else float("inf"))
            )
            max_delta = max(max_delta, delta)

        if max_delta < theta:
            break

    # Uniform average over labeled start states
    start_vals = [V[s] for s in start_states]
    # If any start value is +inf, the mean is +inf; numpy handles this naturally.
    return float(np.mean(start_vals))


def value_diff(
    target_policy: PolicyTable,
    prior_q: QTable,
    target_q: QTable,
    mdp_network: MDPNetwork,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iterations: int = 1000,
) -> float:
    """
    Compute discounted expected sum of per-state absolute value-difference under target policy.
    Per-state immediate reward:
        r(s) = | E_{a~pi_t}[ Q_target(s,a) ] - E_{a~pi_t}[ Q_prior(s,a) ] |
    Then evaluate V(s) under target_policy and MDP dynamics:
        V(s) = r(s) + gamma * E_{a~pi_t, s'~P(·|s,a)}[ V(s') ]
    Return the uniform average of V(s) over start states.
    """
    states = mdp_network.states
    A = mdp_network.num_actions
    start_states = list(mdp_network.start_states)

    if not states or not start_states:
        return 0.0

    # -------- Per-state immediate reward from Q expectations --------
    abs_diff_reward: Dict[int, float] = {}
    for s in states:
        if mdp_network.is_terminal_state(s):
            abs_diff_reward[s] = 0.0
            continue

        action_probs = target_policy.get_action_probabilities(s)
        # Expected Q under target policy for both prior and target Q-tables
        exp_q_target = 0.0
        exp_q_prior = 0.0
        for a in range(A):
            pi_sa = float(action_probs.get(a, 0.0))
            if pi_sa <= 0.0:
                continue
            exp_q_target += pi_sa * float(target_q.get_q_value(s, a))
            exp_q_prior  += pi_sa * float(prior_q.get_q_value(s, a))

        abs_diff_reward[s] = abs(exp_q_target - exp_q_prior)

    # -------- Policy evaluation under target policy with state-only rewards --------
    V: Dict[int, float] = {s: 0.0 for s in states}

    for _ in range(max_iterations):
        max_delta = 0.0
        for s in states:
            if mdp_network.is_terminal_state(s):
                V[s] = 0.0
                continue

            old_v = V[s]

            # Expected next value under target policy and MDP dynamics
            exp_next = 0.0
            action_probs = target_policy.get_action_probabilities(s)
            for a, pi_sa in action_probs.items():
                if pi_sa <= 0.0:
                    continue
                trans = mdp_network.get_transition_probabilities(s, a)
                if not trans:
                    # Fallback: stay in s if no explicit transitions
                    exp_next += pi_sa * V[s]
                else:
                    for sp, p in trans.items():
                        exp_next += pi_sa * p * V[sp]

            new_v = abs_diff_reward[s] + gamma * exp_next
            V[s] = new_v

            delta = abs(new_v - old_v)
            max_delta = max(max_delta, delta)

        if max_delta < theta:
            break

    # -------- Average over start states --------
    start_vals = [V[s] for s in start_states]
    return float(np.mean(start_vals))


def performance_curve_and_integral(
        prior_policy: PolicyTable,
        target_policy: PolicyTable,
        mdp_network: MDPNetwork,
        numpoints: int = 100,
        gamma: float = 0.99,
        theta: float = 1e-6,
        max_iterations: int = 1000,
    ) -> Tuple[List[float], float]:
    """
    Evaluate avg start-state value while blending from prior(0) -> target(1).

    Note: blend_policies uses `weight` as prior weight; i.e., weight=1 -> prior, 0 -> target.
    We therefore call it with `blend_w = 1 - w_user`, where w_user in [0,1] is prior->target.

    Returns:
        (curve_values, curve_mean) where curve_values[i] is the average V over start states
        at user weight w_i in linspace(0,1,numpoints), and curve_mean is mean(curve_values).
    """

    # x-axis: 0 (all prior) -> 1 (all target)
    w_user_list = np.linspace(0.0, 1.0, numpoints).tolist()
    curve_values: List[float] = []

    # Fallback: if no explicit start states, average over all states
    start_states = (
        mdp_network.start_states if mdp_network.start_states else mdp_network.states
    )

    for w_user in w_user_list:
        blend_w = 1.0 - w_user  # convert to blend_policies' convention
        blended = blend_policies(
            target=target_policy, prior=prior_policy, weight=blend_w
        )
        vt = policy_evaluation(
            mdp_network=mdp_network,
            policy=blended,
            gamma=gamma,
            theta=theta,
            max_iterations=max_iterations,
        )
        # Average V over start states
        avg_v = (
            float(np.mean([vt.get_value(s) for s in start_states]))
            if start_states
            else 0.0
        )
        curve_values.append(avg_v)

    # Mean of the curve (also an approximation to the integral over [0,1])
    curve_mean = float(np.mean(curve_values))
    return curve_values, curve_mean
