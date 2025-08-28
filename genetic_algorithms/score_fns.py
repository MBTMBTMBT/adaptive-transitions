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
    curve: str = "greedy",  # default to greedy as requested
) -> Sequence[float]:
    """
    Score by curriculum-learning phase-wise AUCs on the chosen eval curve.

    Returns:
        [phase1_auc, phase2_auc]  (maximize both)

    Contract:
    - Runs CL ONLY (no baseline), no W&B, no filesystem outputs.
    - Uses the candidate MDP as the 'item' env by passing mdp.to_portable()
      directly into the item factory cfg (no temp JSON files).
    - Requires exactly >= 2 phases; phase-1 on item, phase-2 on target.
    - Computes phase-wise AUCs produced by the trainer's metrics:
        summary["metrics"]["items"]["CAND"][curve]["auc_phase"] -> {"p1", "p2", ...}
    """

    curve_key = str(curve).lower()
    if curve_key not in ("greedy", "train"):
        raise ValueError(
            "obj_cl_phase_auc: 'curve' must be either 'greedy' or 'train'."
        )

    # default evals -> evaluate only on Target
    evals_final: List[Dict[str, Any]]
    if evals is None:
        evals_final = [{"name": "Target", "env": "target"}]
    else:
        if not isinstance(evals, (list, tuple)) or len(evals) == 0:
            raise ValueError(
                "obj_cl_phase_auc: 'evals' must be a non-empty list when provided."
            )
        # shallow validation
        evals_final = []
        for es in evals:
            if not isinstance(es, dict) or "name" not in es or "env" not in es:
                raise ValueError(
                    "obj_cl_phase_auc: each eval spec must be a dict with keys 'name' and 'env'."
                )
            evals_final.append({"name": str(es["name"]), "env": str(es["env"])})

    # ----------------- build CL registry (target + item=CAND) -----------------
    # NOTE: item env uses in-memory candidate MDP (mdp.to_portable()), no files.
    envs = {
        "target": {
            "factory_path": target_factory_path,
            "cfg": dict(target_cfg),
        },
        "items": {
            "CAND": {
                "factory_path": item_factory_path,
                "cfg": {
                    "mdp_portable": mdp.to_portable(),
                    "max_steps": int(item_max_steps),
                },
            }
        },
    }

    # curriculum: phase-1 on item, phase-2 on target
    p1, p2 = int(phase_steps[0]), int(phase_steps[1])
    item_phases_map = {
        "CAND": [
            {"env": "CAND", "steps": p1},
            {"env": "target", "steps": p2},
        ]
    }
    evals_map = {"CAND": list(evals_final)}

    # baseline placeholders (won't be used because run_baseline=False)
    baseline_phases: List[Dict[str, Any]] = []
    baseline_evals: List[Dict[str, Any]] = []

    # metrics config: compute only the requested curve
    metrics_opts = {
        "enabled": True,
        "compute_greedy": (curve_key == "greedy"),
        "compute_train": (curve_key == "train"),
        # other options fall back to trainer defaults internally
    }

    # ----------------- run CL (no baseline, no IO, no W&B) -----------------
    if isinstance(seeds, int):
        assert seeds >= 1, "Number of seeds must be >= 1."
        seeds = [i for i in range(seeds)]
    summary = run_curriculum(
        seeds=list(seeds),
        envs=envs,
        baseline_phases=baseline_phases,
        baseline_evals=baseline_evals,
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

    # ----------------- extract metrics (strict checks) -----------------
    if not isinstance(summary, dict):
        raise ValueError("obj_cl_phase_auc: invalid summary (not a dict).")

    metrics = summary.get("metrics", None)
    if not isinstance(metrics, dict):
        raise ValueError("obj_cl_phase_auc: metrics missing from trainer summary.")

    items_metrics = metrics.get("items", None)
    if not (isinstance(items_metrics, dict) and "CAND" in items_metrics):
        raise ValueError("obj_cl_phase_auc: metrics.items['CAND'] missing.")

    cand = items_metrics["CAND"]
    if curve_key not in cand:
        raise ValueError(
            f"obj_cl_phase_auc: metrics for curve '{curve_key}' missing under items['CAND']."
        )

    auc_phase = cand[curve_key].get("auc_phase", None)
    if not (isinstance(auc_phase, dict) and "p1" in auc_phase and "p2" in auc_phase):
        raise ValueError(
            "obj_cl_phase_auc: 'auc_phase' with 'p1' and 'p2' is missing for items['CAND']."
        )

    p1_auc = float(auc_phase["p1"])
    p2_auc = float(auc_phase["p2"])
    return [p1_auc, p2_auc]


SCORE_FNS: Dict[str, Callable[..., Sequence[float]]] = {
    "obj_multi_kl_and_perf": obj_multi_kl_and_perf,
    "obj_multi_perf": obj_multi_perf,
    "obj_cl_phase_auc": obj_cl_phase_auc,
}
