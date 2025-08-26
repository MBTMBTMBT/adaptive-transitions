from __future__ import annotations

from typing import Any, List, Tuple, Dict, Sequence, Callable

from mdp_network import MDPNetwork
from mdp_network.mdp_tables import PolicyTable, ValueTable, q_table_to_policy, create_random_policy, blend_policies
from mdp_network.metrics import kl_policies, performance_curve_and_integral
from mdp_network.solvers import optimal_value_iteration, compute_occupancy_measure


def _normalize_score_spec(spec: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Normalize user-facing score spec into a list of (name, params) tuples.
    """
    default = [("obj_multi_perf", {})]

    if spec is None:
        items = default
    elif isinstance(spec, str):
        items = [(spec, {})]
    elif isinstance(spec, tuple):
        if len(spec) != 2 or not isinstance(spec[0], str) or not isinstance(spec[1], dict):
            raise TypeError("Score tuple must be (name: str, params: dict).")
        items = [(spec[0], dict(spec[1]))]
    elif isinstance(spec, list):
        items = []
        for it in spec:
            if isinstance(it, str):
                items.append((it, {}))
            elif isinstance(it, tuple) and len(it) == 2 and isinstance(it[0], str) and isinstance(it[1], dict):
                items.append((it[0], dict(it[1])))
            else:
                raise TypeError("Score list items must be 'name' or ('name', {params}).")
    else:
        raise TypeError("Unsupported score spec type.")

    # Validate names early for better error messages
    unknown = [n for (n, _) in items if n not in SCORE_FNS]
    if unknown:
        valid = ", ".join(sorted(SCORE_FNS.keys()))
        raise KeyError(f"Unknown score function(s): {unknown}. Available: {valid}")

    return items


def obj_multi_kl_and_perf(mdp: MDPNetwork, shared: Dict[str, Any], *, kl_delta: float = 1e-3) -> Sequence[float]:
    """
    Returns [ -KL(baseline || current), performance_integral ].
    Uses shared['precomputed'] = [base_policy, base_occupancy].
    """
    solver = shared.get("solver", {})
    gamma = float(solver.get("vi_gamma", 0.99))
    theta = float(solver.get("vi_theta", 1e-6))
    max_iter = int(solver.get("vi_max_iterations", 1000))
    temperature = float(solver.get("policy_temperature", 1.0))
    mixing = tuple(solver.get("policy_mix", (0.0, 1.0, 0.0)))
    tie_tol = float(solver.get("policy_tie_tol", 1e-6))

    pgamma = float(solver.get("perf_gamma", gamma))
    ptheta = float(solver.get("perf_theta", theta))
    pmax_iter = int(solver.get("perf_max_iterations", max_iter))
    numpoints = int(solver.get("perf_numpoints", 100))

    pre = shared.get("precomputed", None) or []
    base_policy = PolicyTable.from_portable(pre[0]) if len(pre) >= 1 else None
    base_occupancy = ValueTable.from_portable(pre[1]) if len(pre) >= 2 else None

    _, Q2 = optimal_value_iteration(mdp, gamma=gamma, theta=theta, max_iterations=max_iter)
    policy2: PolicyTable = q_table_to_policy(
        Q2, states=list(mdp.states), num_actions=mdp.num_actions,
        mixing=mixing, temperature=temperature, tie_tol=tie_tol,
    )
    occupancy2: ValueTable = compute_occupancy_measure(mdp, policy=policy2, gamma=gamma, theta=theta, max_iterations=max_iter)

    if base_policy is not None and base_occupancy is not None:
        kl = kl_policies(policy1=base_policy, occupancy1=base_occupancy,
                         policy2=policy2, occupancy2=occupancy2, delta=float(kl_delta))
        obj1 = -float(kl)
    else:
        obj1 = 0.0

    prior = create_random_policy(mdp)
    _curve, integral = performance_curve_and_integral(
        prior_policy=prior, target_policy=policy2, mdp_network=mdp,
        numpoints=numpoints, gamma=pgamma, theta=ptheta, max_iterations=pmax_iter,
    )
    return [obj1, float(integral)]


def obj_multi_perf(mdp: MDPNetwork, shared: Dict[str, Any], *, blend_weight: float = 0.8) -> Sequence[float]:
    """
    Returns:
      1) integral( blend(policy2, random, w) -> baseline_policy )
      2) integral( random -> policy2 )
    """
    solver = shared.get("solver", {})
    gamma = float(solver.get("vi_gamma", 0.99))
    theta = float(solver.get("vi_theta", 1e-6))
    max_iter = int(solver.get("vi_max_iterations", 1000))
    temperature = float(solver.get("policy_temperature", 1.0))
    mixing = tuple(solver.get("policy_mix", (0.0, 1.0, 0.0)))
    tie_tol = float(solver.get("policy_tie_tol", 1e-6))

    pgamma = float(solver.get("perf_gamma", gamma))
    ptheta = float(solver.get("perf_theta", theta))
    pmax_iter = int(solver.get("perf_max_iterations", max_iter))
    numpoints = int(solver.get("perf_numpoints", 100))

    pre = shared.get("precomputed", None) or []
    base_policy = PolicyTable.from_portable(pre[0]) if len(pre) >= 1 else None

    _, Q2 = optimal_value_iteration(mdp, gamma=gamma, theta=theta, max_iterations=max_iter)
    policy2: PolicyTable = q_table_to_policy(
        Q2, states=list(mdp.states), num_actions=mdp.num_actions,
        mixing=mixing, temperature=temperature, tie_tol=tie_tol,
    )

    prior_rand = create_random_policy(mdp)
    blended = blend_policies(policy2, prior_rand, weight=float(blend_weight))

    if base_policy is not None:
        _curve, integral1 = performance_curve_and_integral(
            prior_policy=blended, target_policy=base_policy, mdp_network=mdp,
            numpoints=numpoints, gamma=pgamma, theta=ptheta, max_iterations=pmax_iter,
        )
    else:
        integral1 = 0.0

    _curve, integral2 = performance_curve_and_integral(
        prior_policy=prior_rand, target_policy=policy2, mdp_network=mdp,
        numpoints=numpoints, gamma=pgamma, theta=ptheta, max_iterations=pmax_iter,
    )
    return [float(integral1), float(integral2)]


SCORE_FNS: Dict[str, Callable[..., Sequence[float]]] = {
    "obj_multi_kl_and_perf": obj_multi_kl_and_perf,
    "obj_multi_perf": obj_multi_perf,
}
