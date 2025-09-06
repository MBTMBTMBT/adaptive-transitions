from __future__ import annotations

from typing import Any, List, Tuple, Dict, Sequence, Callable, Union

from mdp_network import MDPNetwork
from mdp_network.mdp_tables import (
    PolicyTable,
    ValueTable,
    q_table_to_policy,
    create_random_policy,
    blend_policies,
)
from mdp_network.metrics import kl_policies, performance_curve_and_integral
from mdp_network.solvers import optimal_value_iteration, compute_occupancy_measure
from two_stage_cl.tabular_curriculum_trainer_ray import run_curriculum


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
        if (
            len(spec) != 2
            or not isinstance(spec[0], str)
            or not isinstance(spec[1], dict)
        ):
            raise TypeError("Score tuple must be (name: str, params: dict).")
        items = [(spec[0], dict(spec[1]))]
    elif isinstance(spec, list):
        items = []
        for it in spec:
            if isinstance(it, str):
                items.append((it, {}))
            elif (
                isinstance(it, tuple)
                and len(it) == 2
                and isinstance(it[0], str)
                and isinstance(it[1], dict)
            ):
                items.append((it[0], dict(it[1])))
            else:
                raise TypeError(
                    "Score list items must be 'name' or ('name', {params})."
                )
    else:
        raise TypeError("Unsupported score spec type.")

    # Validate names early for better error messages
    unknown = [n for (n, _) in items if n not in SCORE_FNS]
    if unknown:
        valid = ", ".join(sorted(SCORE_FNS.keys()))
        raise KeyError(f"Unknown score function(s): {unknown}. Available: {valid}")

    return items


def obj_multi_kl_and_perf(
    mdp: MDPNetwork,
    shared: Dict[str, Any],
    *,
    # --- explicit VI / policy params (decoupled from GA) ---
    vi_gamma: float = 0.99,
    vi_theta: float = 1e-6,
    vi_max_iterations: int = 1000,
    policy_temperature: float = 1.0,
    policy_mixing: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    policy_tie_tol: float = 1e-6,
    # --- explicit performance-curve params (decoupled from GA) ---
    perf_numpoints: int = 100,
    perf_gamma: float | None = None,
    perf_theta: float | None = None,
    perf_max_iterations: int | None = None,
    # --- KL smoothing ---
    kl_delta: float = 1e-3,
) -> Sequence[float]:
    """
    Returns [ -KL(base_policy || current_opt_policy), integral(random -> current_opt_policy) ].

    Strict requirements on `shared['precomputed']`:
      - pre[0] = base_policy (PolicyTable portable) for the ORIGINAL base MDP.
      - pre[1] = base_occupancy (ValueTable portable) for the ORIGINAL base MDP.
    These must exist; otherwise a ValueError is raised.
    """
    # ---- strict validation of precomputed materials ----
    pre = shared.get("precomputed", None)
    if not (
        isinstance(pre, list)
        and len(pre) >= 2
        and pre[0] is not None
        and pre[1] is not None
    ):
        raise ValueError(
            "obj_multi_kl_and_perf requires shared['precomputed'][0]=base_policy and [1]=base_occupancy."
        )
    base_policy = PolicyTable.from_portable(pre[0])
    base_occupancy = ValueTable.from_portable(pre[1])

    # ---- compute current optimal policy on candidate MDP (explicit params) ----
    _, Q2 = optimal_value_iteration(
        mdp,
        gamma=float(vi_gamma),
        theta=float(vi_theta),
        max_iterations=int(vi_max_iterations),
    )
    policy2: PolicyTable = q_table_to_policy(
        Q2,
        states=list(mdp.states),
        num_actions=mdp.num_actions,
        mixing=tuple(policy_mixing),
        temperature=float(policy_temperature),
        tie_tol=float(policy_tie_tol),
    )
    occupancy2: ValueTable = compute_occupancy_measure(
        mdp,
        policy=policy2,
        gamma=float(vi_gamma),
        theta=float(vi_theta),
        max_iterations=int(vi_max_iterations),
    )

    # ---- KL term (strict, no fallbacks) ----
    kl = kl_policies(
        policy1=base_policy,
        occupancy1=base_occupancy,
        policy2=policy2,
        occupancy2=occupancy2,
        delta=float(kl_delta),
    )
    obj_kl = -float(kl)

    # ---- performance integral: random -> current policy on candidate MDP ----
    pgamma = float(vi_gamma) if perf_gamma is None else float(perf_gamma)
    ptheta = float(vi_theta) if perf_theta is None else float(perf_theta)
    pmax_iter = (
        int(vi_max_iterations)
        if perf_max_iterations is None
        else int(perf_max_iterations)
    )

    prior = create_random_policy(mdp)
    _curve, integral = performance_curve_and_integral(
        prior_policy=prior,
        target_policy=policy2,
        mdp_network=mdp,
        numpoints=int(perf_numpoints),
        gamma=pgamma,
        theta=ptheta,
        max_iterations=pmax_iter,
    )
    return [obj_kl, float(integral)]


def obj_multi_perf(
    mdp: MDPNetwork,
    shared: Dict[str, Any],
    *,
    # --- explicit VI / policy params (decoupled) ---
    vi_gamma: float = 0.99,
    vi_theta: float = 1e-6,
    vi_max_iterations: int = 1000,
    policy_temperature: float = 1.0,
    policy_mixing: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    policy_tie_tol: float = 1e-6,
    # --- explicit performance-curve params (decoupled) ---
    perf_numpoints: int = 100,
    perf_gamma: float | None = None,
    perf_theta: float | None = None,
    perf_max_iterations: int | None = None,
    # --- blend knob ---
    blend_weight: float = 0.8,
) -> Sequence[float]:
    """
    Returns, in this order (maximize both):
      1) integral( random -> blended(policy_opt_cand, random, w) )  on the candidate MDP `mdp`.
      2) integral( blended(policy_opt_cand, random, w) -> base_optimal_policy ) on the ORIGINAL base MDP.

    Strict requirements on `shared['precomputed']`:
      - pre[0] = base_policy (PolicyTable portable) for the ORIGINAL base MDP.
      - pre[2] = base_mdp (MDPNetwork portable) — ORIGINAL base MDP graph.
    These must exist; otherwise a ValueError is raised.
    """
    # ---- strict validation of precomputed materials ----
    pre = shared.get("precomputed", None)
    if not (
        isinstance(pre, list)
        and len(pre) >= 3
        and pre[0] is not None
        and pre[2] is not None
    ):
        raise ValueError(
            "obj_multi_perf requires shared['precomputed'][0]=base_policy and [2]=base_mdp."
        )
    base_policy = PolicyTable.from_portable(pre[0])
    base_mdp = MDPNetwork.from_portable(pre[2])

    # ---- compute candidate's optimal policy (explicit params) ----
    _, Q2 = optimal_value_iteration(
        mdp,
        gamma=float(vi_gamma),
        theta=float(vi_theta),
        max_iterations=int(vi_max_iterations),
    )
    policy2: PolicyTable = q_table_to_policy(
        Q2,
        states=list(mdp.states),
        num_actions=mdp.num_actions,
        mixing=tuple(policy_mixing),
        temperature=float(policy_temperature),
        tie_tol=float(policy_tie_tol),
    )

    # ---- blended prior between candidate-optimal and random ----
    prior_rand = create_random_policy(mdp)
    blended = blend_policies(policy2, prior_rand, weight=float(blend_weight))

    # ---- performance params (fallbacks to VI defaults) ----
    pgamma = float(vi_gamma) if perf_gamma is None else float(perf_gamma)
    ptheta = float(vi_theta) if perf_theta is None else float(perf_theta)
    pmax_iter = (
        int(vi_max_iterations)
        if perf_max_iterations is None
        else int(perf_max_iterations)
    )
    N = int(perf_numpoints)

    # 0) random -> blended on candidate MDP
    _curve, integral0 = performance_curve_and_integral(
        prior_policy=prior_rand,
        target_policy=blended,
        mdp_network=mdp,
        numpoints=N,
        gamma=pgamma,
        theta=ptheta,
        max_iterations=pmax_iter,
    )

    # 1) blended -> base_optimal on ORIGINAL base MDP
    _curve, integral1 = performance_curve_and_integral(
        prior_policy=blended,
        target_policy=base_policy,
        mdp_network=base_mdp,
        numpoints=N,
        gamma=pgamma,
        theta=ptheta,
        max_iterations=pmax_iter,
    )

    return [float(integral0), float(integral1)]


def obj_cl_phase_auc(
    mdp: MDPNetwork,
    shared: Dict[str, Any],
    *,
    # ---- required environment inputs ----
    target_factory_path: str,
    target_cfg: Dict[str, Any],
    item_factory_path: str,
    item_max_steps: int,
    # ---- required curriculum/training inputs ----
    phase_steps: Sequence[int],
    seeds: Union[int, Sequence[int]],
    agent_ctor_path: str,
    agent_kwargs: Dict[str, Any],
    eval_every: int,
    n_eval_episodes: int,
    # ---- optional knobs ----
    evals: Sequence[Dict[str, Any]] | None = None,
    curve: str = "greedy",        # "greedy" | "train"
    eval_scope: str = "target",   # "target" | "item" — which eval branch to score on
) -> Sequence[float]:
    """
    Score by curriculum-learning phase-wise AUCs on the chosen curve/scope.

    Returns:
        [auc_p1, auc_p2]  (maximize both)

    Contract (strict):
    - Runs CL ONLY (no baseline), no W&B, no filesystem outputs.
    - Uses the candidate MDP as the 'item' env via in-memory portable.
    - Requires >= 2 phases: phase-1 on item, phase-2 on target.
    - Expects metrics at:
        summary["metrics"]["items"]["CAND"][eval_scope][curve]["auc_p1"|"auc_p2"]
    """

    # ---- strict arg checks ----
    curve_key = str(curve).lower()
    if curve_key not in ("greedy", "train"):
        raise ValueError("obj_cl_phase_auc: 'curve' must be 'greedy' or 'train'.")
    scope_key = str(eval_scope).lower()
    if scope_key not in ("target", "item"):
        raise ValueError("obj_cl_phase_auc: 'eval_scope' must be 'target' or 'item'.")
    if not isinstance(phase_steps, (list, tuple)) or len(phase_steps) < 2:
        raise ValueError("obj_cl_phase_auc: needs at least two phases (p1, p2).")

    # ---- default evals: record Target; add Item if you want item-scope curves too ----
    if evals is None:
        evals_final: List[Dict[str, Any]] = [{"name": "Target", "env": "target"}]
        # If you also want item-scope metrics produced during the run, include:
        # evals_final.append({"name": "CAND", "env": "CAND"})
    else:
        if not isinstance(evals, (list, tuple)) or len(evals) == 0:
            raise ValueError("obj_cl_phase_auc: 'evals' must be a non-empty list.")
        evals_final = []
        for es in evals:
            if not isinstance(es, dict) or "name" not in es or "env" not in es:
                raise ValueError("obj_cl_phase_auc: each eval needs 'name' and 'env'.")
            evals_final.append({"name": str(es["name"]), "env": str(es["env"])})

    # ---- build registry: target + item=CAND (in-memory portable) ----
    envs = {
        "target": {"factory_path": target_factory_path, "cfg": dict(target_cfg)},
        "items": {
            "CAND": {
                "factory_path": item_factory_path,
                "cfg": {"mdp_portable": mdp.to_portable(), "max_steps": int(item_max_steps)},
            }
        },
    }

    # ---- curriculum: p1 on item, p2 on target ----
    p1, p2 = int(phase_steps[0]), int(phase_steps[1])
    item_phases_map = {"CAND": [{"env": "CAND", "steps": p1}, {"env": "target", "steps": p2}]}
    evals_map = {"CAND": list(evals_final)}

    # ---- CL run: baseline OFF, no I/O/W&B ----
    if isinstance(seeds, int):
        if seeds < 1:
            raise ValueError("obj_cl_phase_auc: seeds must be >= 1.")
        seeds = [i for i in range(seeds)]

    metrics_opts = {
        "enabled": True,
        "compute_greedy": (curve_key == "greedy"),
        "compute_train": (curve_key == "train"),
    }

    summary = run_curriculum(
        seeds=list(seeds),
        envs=envs,
        baseline_phases=[],
        baseline_evals=[],
        item_phases_map=item_phases_map,
        evals_map=evals_map,
        agent_ctor_path=agent_ctor_path,
        agent_kwargs=dict(agent_kwargs),
        eval_every=int(eval_every),
        n_eval_episodes=int(n_eval_episodes),
        output_dir=None,
        save_intermediate=False,
        wandb_actor=None,
        media_opts=None,
        wandb_step_base=0,
        run_baseline=False,
        run_items=True,
        metrics_opts=metrics_opts,
    )

    # ---- strict extraction: expect auc_p1/auc_p2 (NOT 'auc_phase') ----
    if not isinstance(summary, dict):
        raise ValueError("obj_cl_phase_auc: invalid summary (not a dict).")
    metrics = summary.get("metrics", None)
    if not isinstance(metrics, dict):
        raise ValueError("obj_cl_phase_auc: metrics missing from trainer summary.")
    items_m = metrics.get("items", None)
    if not (isinstance(items_m, dict) and "CAND" in items_m):
        raise ValueError("obj_cl_phase_auc: metrics.items['CAND'] missing.")

    cand = items_m["CAND"]
    if scope_key not in cand:
        raise ValueError(f"obj_cl_phase_auc: items['CAND']['{scope_key}'] missing.")
    scope_block = cand[scope_key]

    if curve_key not in scope_block:
        raise ValueError(
            f"obj_cl_phase_auc: items['CAND']['{scope_key}']['{curve_key}'] missing."
        )
    ch = scope_block[curve_key]

    if "auc_p1" not in ch or "auc_p2" not in ch:
        raise ValueError(
            f"obj_cl_phase_auc: 'auc_p1'/'auc_p2' missing at items['CAND']['{scope_key}']['{curve_key}']."
        )

    p1_auc, p2_auc = ch["auc_p1"], ch["auc_p2"]
    if p1_auc is None or p2_auc is None:
        raise ValueError(
            f"obj_cl_phase_auc: auc_p1/auc_p2 contain None at items['CAND']['{scope_key}']['{curve_key}']."
        )

    return [float(p1_auc), float(p2_auc)]


SCORE_FNS: Dict[str, Callable[..., Sequence[float]]] = {
    "obj_multi_kl_and_perf": obj_multi_kl_and_perf,
    "obj_multi_perf": obj_multi_perf,
    "obj_cl_phase_auc": obj_cl_phase_auc,
}
