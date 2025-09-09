from __future__ import annotations

from typing import Any, List, Tuple, Dict, Sequence, Callable, Union, Optional

import numpy as np

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
    Strict normalization:
    - Only accepts List[Tuple[str, Dict]].
    - No compatibility branches.
    """
    if not isinstance(spec, list):
        raise TypeError("score_spec must be a list of (name: str, params: dict).")
    items: List[Tuple[str, Dict[str, Any]]] = []
    for it in spec:
        if (not isinstance(it, tuple)) or len(it) != 2 or not isinstance(it[0], str) or not isinstance(it[1], dict):
            raise TypeError("Each score item must be ('name', {params}).")
        items.append((it[0], dict(it[1])))

    unknown = [n for (n, _) in items if n not in SCORE_FNS]
    if unknown:
        valid = ", ".join(sorted(SCORE_FNS.keys()))
        raise KeyError(f"Unknown score function(s): {unknown}. Available: {valid}")
    return items

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
    vi_gamma: float = 0.99,
    vi_theta: float = 1e-6,
    vi_max_iterations: int = 1000,
    policy_temperature: float = 1.0,
    policy_mixing: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    policy_tie_tol: float = 1e-6,
    perf_numpoints: int = 100,
    perf_gamma: float | None = None,
    perf_theta: float | None = None,
    perf_max_iterations: int | None = None,
    kl_delta: float = 1e-3,
) -> Dict[str, Optional[float]]:
    """
    Outputs:
      - 'kl_neg' (maximize): -KL(base||cand_opt) with occupancy weighting.
      - 'perf_integral' (maximize): integral(random -> cand_opt) on candidate MDP.
    """
    pre = shared.get("precomputed", None)
    if not (isinstance(pre, list) and len(pre) >= 2 and pre[0] is not None and pre[1] is not None):
        raise ValueError("obj_multi_kl_and_perf requires precomputed[0]=base_policy, [1]=base_occupancy.")

    base_policy = PolicyTable.from_portable(pre[0])
    base_occupancy = ValueTable.from_portable(pre[1])

    _, Q2 = optimal_value_iteration(mdp, gamma=float(vi_gamma), theta=float(vi_theta), max_iterations=int(vi_max_iterations))
    policy2: PolicyTable = q_table_to_policy(
        Q2, states=list(mdp.states), num_actions=mdp.num_actions,
        mixing=tuple(policy_mixing), temperature=float(policy_temperature), tie_tol=float(policy_tie_tol),
    )
    occupancy2: ValueTable = compute_occupancy_measure(
        mdp, policy=policy2, gamma=float(vi_gamma), theta=float(vi_theta), max_iterations=int(vi_max_iterations),
    )

    kl = kl_policies(
        policy1=base_policy, occupancy1=base_occupancy, policy2=policy2, occupancy2=occupancy2, delta=float(kl_delta),
    )
    obj_kl = -float(kl)

    pgamma = float(vi_gamma) if perf_gamma is None else float(perf_gamma)
    ptheta = float(vi_theta) if perf_theta is None else float(perf_theta)
    pmax_iter = int(vi_max_iterations) if perf_max_iterations is None else int(perf_max_iterations)

    prior = create_random_policy(mdp)
    _curve, integral = performance_curve_and_integral(
        prior_policy=prior, target_policy=policy2, mdp_network=mdp,
        numpoints=int(perf_numpoints), gamma=pgamma, theta=ptheta, max_iterations=pmax_iter,
    )

    return {"kl_neg": obj_kl, "perf_integral": float(integral)}


def obj_multi_perf(
    mdp: MDPNetwork,
    shared: Dict[str, Any],
    *,
    vi_gamma: float = 0.99,
    vi_theta: float = 1e-6,
    vi_max_iterations: int = 1000,
    policy_temperature: float = 1.0,
    policy_mixing: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    policy_tie_tol: float = 1e-6,
    perf_numpoints: int = 100,
    perf_gamma: float | None = None,
    perf_theta: float | None = None,
    perf_max_iterations: int | None = None,
    blend_weight: float = 0.8,
) -> Dict[str, Optional[float]]:
    """
    Outputs:
      - 'int_rand_to_blend' (maximize): integral(random -> blended(cand_opt, random, w)) on candidate MDP.
      - 'int_blend_to_base' (maximize): integral(blended -> base_opt) on ORIGINAL base MDP.
    """
    pre = shared.get("precomputed", None)
    if not (isinstance(pre, list) and len(pre) >= 3 and pre[0] is not None and pre[2] is not None):
        raise ValueError("obj_multi_perf requires precomputed[0]=base_policy, [2]=base_mdp.")
    base_policy = PolicyTable.from_portable(pre[0])
    base_mdp = MDPNetwork.from_portable(pre[2])

    _, Q2 = optimal_value_iteration(mdp, gamma=float(vi_gamma), theta=float(vi_theta), max_iterations=int(vi_max_iterations))
    policy2: PolicyTable = q_table_to_policy(
        Q2, states=list(mdp.states), num_actions=mdp.num_actions,
        mixing=tuple(policy_mixing), temperature=float(policy_temperature), tie_tol=float(policy_tie_tol),
    )

    prior_rand = create_random_policy(mdp)
    blended = blend_policies(policy2, prior_rand, weight=float(blend_weight))

    pgamma = float(vi_gamma) if perf_gamma is None else float(perf_gamma)
    ptheta = float(vi_theta) if perf_theta is None else float(perf_theta)
    pmax_iter = int(vi_max_iterations) if perf_max_iterations is None else int(perf_max_iterations)
    N = int(perf_numpoints)

    _c0, integral0 = performance_curve_and_integral(
        prior_policy=prior_rand, target_policy=blended, mdp_network=mdp,
        numpoints=N, gamma=pgamma, theta=ptheta, max_iterations=pmax_iter,
    )
    _c1, integral1 = performance_curve_and_integral(
        prior_policy=blended, target_policy=base_policy, mdp_network=base_mdp,
        numpoints=N, gamma=pgamma, theta=ptheta, max_iterations=pmax_iter,
    )

    return {"int_rand_to_blend": float(integral0), "int_blend_to_base": float(integral1)}


def obj_cl_phase_mean(
    mdp: "MDPNetwork",
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
    evals: Optional[Sequence[Dict[str, Any]]] = None,
    curve: str = "greedy",        # "greedy" | "train"
    eval_scope: str = "target",   # "target" | "item"
    wandb_actor: Optional["ActorHandle"] = None,  # optional console dump
) -> Dict[str, Optional[float]]:
    """
    Return a flat metrics dict (float or None) for the chosen (scope, curve) on a 2-phase CL run.
    We mirror keys produced by the trainer's `_compute_metrics` for the selected branch only.

    Output keys (all present; values may be None if undefined):
      - "p1_mean", "p2_mean", "p1_auc", "p2_auc"
      - "mean_total", "auc_total", "ap_last_k", "ttt_frac"
      - "js_target_start", "js_p2_head", "js_baseline_B"
    """
    # ---- strict arg checks (shape and semantics) ----
    curve_key = str(curve).lower()
    if curve_key not in ("greedy", "train"):
        raise ValueError("obj_cl_phase_mean: 'curve' must be 'greedy' or 'train'.")
    scope_key = str(eval_scope).lower()
    if scope_key not in ("target", "item"):
        raise ValueError("obj_cl_phase_mean: 'eval_scope' must be 'target' or 'item'.")
    if not isinstance(phase_steps, (list, tuple)) or len(phase_steps) < 2:
        raise ValueError("obj_cl_phase_mean: needs at least two phases (p1, p2).")
    if int(phase_steps[0]) <= 0 or int(phase_steps[1]) <= 0:
        raise ValueError("obj_cl_phase_mean: phase steps must be positive for p1 and p2.")

    # ---- eval declarations: always include Target; include CAND if item-scope requested ----
    if evals is None:
        evals_final: List[Dict[str, Any]] = [{"name": "Target", "env": "target"}]
        if scope_key == "item":
            evals_final.append({"name": "CAND", "env": "CAND"})
    else:
        if not (isinstance(evals, (list, tuple)) and len(evals) > 0):
            raise ValueError("obj_cl_phase_mean: 'evals' must be a non-empty list.")
        evals_final = []
        for es in evals:
            if not (isinstance(es, dict) and "name" in es and "env" in es):
                raise ValueError("obj_cl_phase_mean: each eval needs 'name' and 'env'.")
            evals_final.append({"name": str(es["name"]), "env": str(es["env"])})
        required_name = "CAND" if scope_key == "item" else "Target"
        if required_name not in {e["name"] for e in evals_final}:
            raise ValueError(f"obj_cl_phase_mean: evals must include '{required_name}' for scope='{scope_key}'.")

    # ---- registry: target + item=CAND (in-memory portable) ----
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

    # ---- seeds normalization ----
    if isinstance(seeds, int):
        if seeds < 1:
            raise ValueError("obj_cl_phase_mean: seeds must be >= 1.")
        seeds = [i for i in range(seeds)]
    else:
        seeds = list(seeds)
        if len(seeds) < 1:
            raise ValueError("obj_cl_phase_mean: provide at least one seed.")

    # ---- metrics options: compute both channels so the dict is complete; baseline OFF ----
    metrics_opts = {
        "enabled": True,
        "compute_greedy": True,
        "compute_train": True,
        # rely on trainer defaults for cap_steps/js_first_n, etc.
    }

    summary = run_curriculum(
        seeds=seeds,
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
        wandb_actor=None,   # keep training silent
        media_opts=None,
        wandb_step_base=0,
        run_baseline=False,  # baseline disabled -> js_baseline_B will likely be None
        run_items=True,
        metrics_opts=metrics_opts,
    )

    # ---- strict structural checks, but DO NOT require non-None numeric values ----
    if not isinstance(summary, dict):
        raise ValueError("obj_cl_phase_mean: invalid summary (not a dict).")
    metrics = summary.get("metrics", None)
    if not isinstance(metrics, dict):
        raise ValueError("obj_cl_phase_mean: metrics missing from trainer summary.")
    items_m = metrics.get("items", None)
    if not (isinstance(items_m, dict) and "CAND" in items_m):
        raise ValueError("obj_cl_phase_mean: metrics.items['CAND'] missing.")

    cand = items_m["CAND"]
    if scope_key not in cand:
        raise ValueError(f"obj_cl_phase_mean: items['CAND']['{scope_key}'] missing.")
    scope_block = cand[scope_key]
    if curve_key not in scope_block:
        raise ValueError(f"obj_cl_phase_mean: items['CAND']['{scope_key}']['{curve_key}'] missing.")
    ch = scope_block[curve_key]

    # ---- primary metrics (p1/p2 means & AUCs, totals, last-k, TTT) ----
    out: Dict[str, Optional[float]] = {
        "p1_mean": ch.get("mean_p1"),
        "p2_mean": ch.get("mean_p2"),
        "p1_auc": ch.get("auc_p1"),
        "p2_auc": ch.get("auc_p2"),
        "mean_total": ch.get("mean_total"),
        "auc_total": ch.get("auc_total"),
        "ap_last_k": ch.get("ap_last_k"),
        "ttt_frac": ch.get("ttt_fraction"),
    }

    # ---- jumpstart (absolute levels) from trainer (may be None if undefined) ----
    js_block = (cand.get("jumpstart", {}) or {}).get(curve_key, {}) or {}
    out["js_target_start"] = js_block.get("target_start")
    out["js_p2_head"] = js_block.get("p2_head")
    out["js_baseline_B"] = js_block.get("baseline_B")

    # ---- optional console dump (compact) ----
    if wandb_actor is not None:
        try:
            wandb_actor.write_console.remote(
                "[obj_cl_phase_mean] "
                f"scope={scope_key} curve={curve_key} "
                f"p1_mean={out['p1_mean']} p2_mean={out['p2_mean']} "
                f"auc_total={out['auc_total']}"
            )
        except Exception:
            pass

    return out


SCORE_FNS: Dict[str, Callable[..., Dict[str, Optional[float]]]] = {
    "obj_multi_kl_and_perf": obj_multi_kl_and_perf,
    "obj_multi_perf": obj_multi_perf,
    "obj_cl_phase_mean": obj_cl_phase_mean,
}
