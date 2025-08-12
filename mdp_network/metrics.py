import math
from collections.abc import Set
from typing import Dict, Tuple, List

import numpy as np

from mdp_network import MDPNetwork
from mdp_network.solvers import policy_evaluation
from mdp_tables import PolicyTable, ValueTable, blend_policies


# How to actually compute this, we need a discussion.
def kl_policies(
    policy1: PolicyTable,
    occupancy1: ValueTable,
    policy2: PolicyTable,
    occupancy2: ValueTable,
    delta: float = 1e-3,
) -> float:
    """
    Symmetric, occupancy-weighted KL between two policies. (THIS IS NOT JS THOUGH!)

    For each state s in the union of states (from policies/occupancies):
      KL12(s) = KL(pi1(.|s) || pi2(.|s)), KL21(s) = KL(pi2(.|s) || pi1(.|s)).
    Use the union action set and additive smoothing (delta) so all probs > 0.
    Aggregate: sum1 = Σ_s occ1[s]*KL12(s), sum2 = Σ_s occ2[s]*KL21(s).
    Return 0.5 * (sum1 + sum2) as a single float.

    Args: policy1, occupancy1, policy2, occupancy2, delta.
    """

    # Build the union of all states that appear anywhere.
    states: Set[int] = set(policy1.get_all_states()) | set(policy2.get_all_states()) \
                       | set(occupancy1.get_all_states()) | set(occupancy2.get_all_states())

    def smoothed_dist(dist: Dict[int, float], actions: Set[int], eps: float) -> Dict[int, float]:
        """Additive-smooth distribution over `actions`, then renormalize."""
        K = len(actions)
        # Sum of raw probs over the union action set (missing -> 0.0)
        raw_sum = 0.0
        for a in actions:
            raw_sum += float(dist.get(a, 0.0))
        denom = raw_sum + eps * K if K > 0 else 1.0  # guard K=0 (shouldn't happen)

        # Return (p(a)+eps)/denom for all a in union
        return {int(a): (float(dist.get(a, 0.0)) + eps) / denom for a in actions} if K > 0 else {}

    def kl(p: Dict[int, float], q: Dict[int, float], actions: Set[int], eps: float) -> float:
        """KL( P || Q ) with additive smoothing on the union action set."""
        if not actions:
            return 0.0
        P = smoothed_dist(p, actions, eps)
        Q = smoothed_dist(q, actions, eps)
        acc = 0.0
        for a in actions:
            pa = P[a]
            qa = Q[a]
            acc += pa * math.log(pa / qa)
        return acc

    sum1 = 0.0
    sum2 = 0.0

    for s in states:
        # Action union at this state (if a state is unseen by a policy, the API gives {0:1.0})
        acts1 = set(policy1.get_action_probabilities(s).keys())
        acts2 = set(policy2.get_action_probabilities(s).keys())
        actions = acts1 | acts2
        if not actions:
            # Extremely defensive; PolicyTable.get_action_probabilities should never yield empty.
            actions = {0}

        # Per-state KLs
        p1 = policy1.get_action_probabilities(s)
        p2 = policy2.get_action_probabilities(s)
        kl12 = kl(p1, p2, actions, delta)
        kl21 = kl(p2, p1, actions, delta)

        # Occupancy-weighted accumulation
        w1 = float(occupancy1.get_value(s))
        w2 = float(occupancy2.get_value(s))
        sum1 += w1 * kl12
        sum2 += w2 * kl21

    return 0.5 * (sum1 + sum2)


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
    start_states = mdp_network.start_states if mdp_network.start_states else mdp_network.states

    for w_user in w_user_list:
        blend_w = 1.0 - w_user  # convert to blend_policies' convention
        blended = blend_policies(
            target=target_policy,
            prior=prior_policy,
            weight=blend_w
        )
        vt = policy_evaluation(
            mdp_network=mdp_network,
            policy=blended,
            gamma=gamma,
            theta=theta,
            max_iterations=max_iterations
        )
        # Average V over start states
        avg_v = float(np.mean([vt.get_value(s) for s in start_states])) if start_states else 0.0
        curve_values.append(avg_v)

    # Mean of the curve (also an approximation to the integral over [0,1])
    curve_mean = float(np.mean(curve_values))
    return curve_values, curve_mean

