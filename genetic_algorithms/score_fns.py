from __future__ import annotations

from typing import Any, List, Tuple, Dict, Sequence, Callable, Union, Optional

from mdp_network import MDPNetwork
from mdp_network.mdp_tables import (
    PolicyTable,
    q_table_to_policy, QTable,
)
from mdp_network.metrics import kl_policies, performance_curve_and_integral, value_diff
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
        if (
            (not isinstance(it, tuple))
            or len(it) != 2
            or not isinstance(it[0], str)
            or not isinstance(it[1], dict)
        ):
            raise TypeError("Each score item must be ('name', {params}).")
        items.append((it[0], dict(it[1])))

    unknown = [n for (n, _) in items if n not in SCORE_FNS]
    if unknown:
        valid = ", ".join(sorted(SCORE_FNS.keys()))
        raise KeyError(f"Unknown score function(s): {unknown}. Available: {valid}")
    return items


def obj_multi_kl(
    mdp: MDPNetwork,
    shared: Dict[str, Any],
    *,
    kl_gamma: float | None = None,
    kl_theta: float | None = None,
    kl_max_iterations: int | None = None,
) -> Dict[str, Optional[float]]:
    """
    Outputs:
      - 'kl_neg' (maximize): -KL(base||cand_opt) with occupancy weighting.
      - 'perf_integral' (maximize): integral(random -> cand_opt) on candidate MDP.
    """
    solver = shared["solver"]
    vi_gamma = float(solver.get("vi_gamma", 0.99))
    vi_theta = float(solver.get("vi_theta", 1e-6))
    vi_max_iterations = int(solver.get("vi_max_iterations", 1000))
    policy_temperature = float(solver.get("policy_temperature", 1.0))
    policy_mix = tuple(solver.get("policy_mix", (0.0, 1.0, 0.0)))
    policy_tie_tol = float(solver.get("policy_tie_tol", 1e-6))

    pgamma = float(vi_gamma) if kl_gamma is None else float(kl_gamma)
    ptheta = float(vi_theta) if kl_theta is None else float(kl_theta)
    pmax_iter = (
        int(vi_max_iterations) if kl_max_iterations is None else int(kl_max_iterations)
    )

    pre = shared["precomputed"]
    target_policy = PolicyTable.from_portable(pre["base_policy"])
    rand_policy = PolicyTable.from_portable(pre["rand_policy"])

    _, Q2 = optimal_value_iteration(
        mdp,
        gamma=float(vi_gamma),
        theta=float(vi_theta),
        max_iterations=int(vi_max_iterations),
    )
    prior_policy: PolicyTable = q_table_to_policy(
        Q2,
        states=list(mdp.states),
        num_actions=mdp.num_actions,
        mixing=tuple(policy_mix),
        temperature=float(policy_temperature),
        tie_tol=float(policy_tie_tol),
    )

    control_kl = kl_policies(
        prior_policy=rand_policy,
        target_policy=prior_policy,
        mdp_network=mdp,
        gamma=float(pgamma),
        theta=float(ptheta),
        max_iterations=int(pmax_iter),
    )
    obj_control_kl = -float(control_kl)

    target_kl = kl_policies(
        prior_policy=prior_policy,
        target_policy=target_policy,
        mdp_network=mdp,
        gamma=float(pgamma),
        theta=float(ptheta),
        max_iterations=int(pmax_iter),
    )
    obj_target_kl = -float(target_kl)

    return {
        "control_kl": control_kl,
        "target_kl": target_kl,
        "minus_control_kl": obj_control_kl,
        "minus_target_kl": obj_target_kl,
    }


def obj_val_diff(
    mdp: MDPNetwork,
    shared: Dict[str, Any],
    *,
    diff_gamma: float | None = None,
    diff_theta: float | None = None,
    diff_max_iterations: int | None = None,
) -> Dict[str, Optional[float]]:
    """
    Outputs:
      - 'kl_neg' (maximize): -KL(base||cand_opt) with occupancy weighting.
      - 'perf_integral' (maximize): integral(random -> cand_opt) on candidate MDP.
    """
    solver = shared["solver"]
    vi_gamma = float(solver.get("vi_gamma", 0.99))
    vi_theta = float(solver.get("vi_theta", 1e-6))
    vi_max_iterations = int(solver.get("vi_max_iterations", 1000))

    pgamma = float(vi_gamma) if diff_gamma is None else float(diff_gamma)
    ptheta = float(vi_theta) if diff_theta is None else float(diff_theta)
    pmax_iter = (
        int(vi_max_iterations) if diff_max_iterations is None else int(diff_max_iterations)
    )

    pre = shared["precomputed"]
    target_policy = PolicyTable.from_portable(pre["base_policy"])
    target_q = QTable.from_portable(pre["base_q"])

    _, source_q = optimal_value_iteration(
        mdp,
        gamma=float(vi_gamma),
        theta=float(vi_theta),
        max_iterations=int(vi_max_iterations),
    )

    val_diff = value_diff(
        target_policy=target_policy,
        prior_q=source_q,
        target_q=target_q,
        mdp_network=mdp,
        gamma=float(pgamma),
        theta=float(ptheta),
        max_iterations=int(pmax_iter),
    )

    return {
        "value_diff": val_diff,
        "minus_value_diff": -float(val_diff),
    }


def obj_multi_perf(
    mdp: MDPNetwork,
    shared: Dict[str, Any],
    *,
    perf_numpoints: int = 100,
    perf_gamma: float | None = None,
    perf_theta: float | None = None,
    perf_max_iterations: int | None = None,
) -> Dict[str, Optional[float]]:
    """
    Outputs:
      - 'int_rand_to_blend' (maximize): integral(random -> blended(cand_opt, random, w)) on candidate MDP.
      - 'int_blend_to_base' (maximize): integral(blended -> base_opt) on ORIGINAL base MDP.
    """
    solver = shared["solver"]
    vi_gamma = float(solver.get("vi_gamma", 0.99))
    vi_theta = float(solver.get("vi_theta", 1e-6))
    vi_max_iterations = int(solver.get("vi_max_iterations", 1000))
    policy_temperature = float(solver.get("policy_temperature", 1.0))
    policy_mix = tuple(solver.get("policy_mix", (0.0, 1.0, 0.0)))
    policy_tie_tol = float(solver.get("policy_tie_tol", 1e-6))

    pgamma = float(vi_gamma) if perf_gamma is None else float(perf_gamma)
    ptheta = float(vi_theta) if perf_theta is None else float(perf_theta)
    pmax_iter = (
        int(vi_max_iterations)
        if perf_max_iterations is None
        else int(perf_max_iterations)
    )
    N = int(perf_numpoints)

    pre = shared["precomputed"]
    target_policy = PolicyTable.from_portable(pre["base_policy"])
    base_mdp = MDPNetwork.from_portable(pre["base_mdp"])
    rand_policy = PolicyTable.from_portable(pre["rand_policy"])

    _, Q2 = optimal_value_iteration(
        mdp,
        gamma=float(vi_gamma),
        theta=float(vi_theta),
        max_iterations=int(vi_max_iterations),
    )
    prior_policy: PolicyTable = q_table_to_policy(
        Q2,
        states=list(mdp.states),
        num_actions=mdp.num_actions,
        mixing=tuple(policy_mix),
        temperature=float(policy_temperature),
        tie_tol=float(policy_tie_tol),
    )

    _c0, int_rand_to_source_on_source = performance_curve_and_integral(
        prior_policy=rand_policy,
        target_policy=prior_policy,
        mdp_network=mdp,
        numpoints=N,
        gamma=pgamma,
        theta=ptheta,
        max_iterations=pmax_iter,
    )
    _c01, int_rand_to_source = performance_curve_and_integral(
        prior_policy=rand_policy,
        target_policy=prior_policy,
        mdp_network=base_mdp,
        numpoints=N,
        gamma=pgamma,
        theta=ptheta,
        max_iterations=pmax_iter,
    )
    _c1, int_source_to_target = performance_curve_and_integral(
        prior_policy=prior_policy,
        target_policy=target_policy,
        mdp_network=base_mdp,
        numpoints=N,
        gamma=pgamma,
        theta=ptheta,
        max_iterations=pmax_iter,
    )

    return {
        "int_rand_to_source_on_source": float(int_rand_to_source_on_source),
        "int_rand_to_source": float(int_rand_to_source),
        "int_source_to_target": float(int_source_to_target),
    }


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
    item_label: str = "Source",
    wandb_actor: Optional["ActorHandle"] = None,  # optional console dump
) -> Dict[str, Optional[float]]:
    """
    Return a flat metrics dict (float or None) for the chosen (scope, curve) on a 2-phase CL run.
    Keys mirror trainer's `_compute_metrics` for the selected branch, plus source-segment metrics.

    Output keys (always present; values may be None if undefined):
      - "mean_p1", "mean_p2", "auc_p1", "auc_p2"
      - "mean_total", "auc_total", "ap_last_k", "ttt_frac"
      - "mean_p1_source", "auc_p1_source"
      - "js_target_start", "js_p2_head", "js_baseline_B"
    """
    # ---- strict arg checks ----
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

    # ---- eval declarations: ALWAYS include both 'Target' and item_label ----
    if evals is None:
        evals_final: List[Dict[str, Any]] = [
            {"name": "Target",    "env": "target"},
            {"name": str(item_label), "env": str(item_label)},
        ]
    else:
        if not (isinstance(evals, (list, tuple)) and len(evals) > 0):
            raise ValueError("obj_cl_phase_mean: 'evals' must be a non-empty list.")
        evals_final = []
        for es in evals:
            if not (isinstance(es, dict) and "name" in es):
                raise ValueError("obj_cl_phase_mean: each eval needs 'name' (and usually 'env').")
            nm = str(es["name"])
            # Coerce env for the two required evals; keep others as provided (if any).
            if nm == "Target":
                ev = "target"
            elif nm == str(item_label):
                ev = str(item_label)
            else:
                ev = str(es.get("env", nm))
            evals_final.append({"name": nm, "env": ev})
        names = {e["name"] for e in evals_final}
        if "Target" not in names:
            evals_final.append({"name": "Target", "env": "target"})
        if str(item_label) not in names:
            evals_final.append({"name": str(item_label), "env": str(item_label)})

    # ---- registry: target + one item labeled by `item_label` ----
    envs = {
        "target": {"factory_path": target_factory_path, "cfg": dict(target_cfg)},
        "items": {
            str(item_label): {
                "factory_path": item_factory_path,
                "cfg": {"mdp_portable": mdp.to_portable(), "max_steps": int(item_max_steps)},
            }
        },
    }

    # ---- curriculum: p1 on item_label, p2 on target ----
    p1, p2 = int(phase_steps[0]), int(phase_steps[1])
    item_phases_map = {
        str(item_label): [{"env": str(item_label), "steps": p1}, {"env": "target", "steps": p2}]
    }
    evals_map = {str(item_label): list(evals_final)}

    # ---- seeds normalization ----
    if isinstance(seeds, int):
        if seeds < 1:
            raise ValueError("obj_cl_phase_mean: seeds must be >= 1.")
        seeds = [i for i in range(seeds)]
    else:
        seeds = list(seeds)
        if len(seeds) < 1:
            raise ValueError("obj_cl_phase_mean: provide at least one seed.")

    # ---- metrics options: compute both channels; baseline OFF ----
    metrics_opts = {"enabled": True, "compute_greedy": True, "compute_train": True}

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
        wandb_actor=None,
        media_opts=None,
        wandb_step_base=0,
        run_baseline=False,
        run_items=True,
        metrics_opts=metrics_opts,
    )

    # ---- structural checks ----
    if not isinstance(summary, dict):
        raise ValueError("obj_cl_phase_mean: invalid summary (not a dict).")
    metrics = summary.get("metrics", None)
    if not isinstance(metrics, dict):
        raise ValueError("obj_cl_phase_mean: metrics missing from trainer summary.")
    items_m = metrics.get("items", None)
    if not (isinstance(items_m, dict) and str(item_label) in items_m):
        raise ValueError(f"obj_cl_phase_mean: metrics.items['{item_label}'] missing.")

    item_metrics = items_m[str(item_label)]
    if scope_key not in item_metrics:
        raise ValueError(f"obj_cl_phase_mean: items['{item_label}']['{scope_key}'] missing.")
    scope_block = item_metrics[scope_key]
    if curve_key not in scope_block:
        raise ValueError(
            f"obj_cl_phase_mean: items['{item_label}']['{scope_key}']['{curve_key}'] missing."
        )
    ch = scope_block[curve_key]

    # ---- pack outputs ----
    out: Dict[str, Optional[float]] = {
        "mean_p1": ch.get("mean_p1"),
        "mean_p2": ch.get("mean_p2"),
        "auc_p1": ch.get("auc_p1"),
        "auc_p2": ch.get("auc_p2"),
        "mean_total": ch.get("mean_total"),
        "auc_total": ch.get("auc_total"),
        "ap_last_k": ch.get("ap_last_k"),
        "ttt_frac": ch.get("ttt_fraction"),
        "mean_p1_source": ch.get("mean_p1_source"),
        "auc_p1_source": ch.get("auc_p1_source"),
    }

    js_block = (item_metrics.get("jumpstart", {}) or {}).get(curve_key, {}) or {}
    out["js_target_start"] = js_block.get("target_start")
    out["js_p2_head"] = js_block.get("p2_head")
    out["js_baseline_B"] = js_block.get("baseline_B")

    if wandb_actor is not None:
        try:
            wandb_actor.write_console.remote(
                "[obj_cl_phase_mean] "
                f"item_label={item_label} scope={scope_key} curve={curve_key} "
                f"mean_p1={out['mean_p1']} mean_p1_source={out['mean_p1_source']} "
                f"mean_p2={out['mean_p2']} auc_total={out['auc_total']}"
            )
        except Exception:
            pass

    return out


SCORE_FNS: Dict[str, Callable[..., Dict[str, Optional[float]]]] = {
    "obj_multi_kl": obj_multi_kl,
    "obj_multi_perf": obj_multi_perf,
    "obj_cl_phase_mean": obj_cl_phase_mean,
    "obj_val_diff": obj_val_diff,
}

SCORE_FN_OUTPUTS: Dict[str, List[str]] = {
    "obj_cl_phase_mean": [
        "mean_p1","mean_p2","auc_p1","auc_p2","mean_total","auc_total",
        "ap_last_k","ttt_frac","mean_p1_source","auc_p1_source",
        "js_target_start","js_p2_head","js_baseline_B",
    ],
    "obj_multi_perf": [
        "int_rand_to_source_on_source","int_rand_to_source","int_source_to_target",
    ],
    "obj_multi_kl": [
        "control_kl","target_kl","minus_control_kl","minus_target_kl",
    ],
    "obj_val_diff": [
        "value_diff","minus_value_diff",
    ],
}

METRIC_TO_FN: Dict[str, str] = {}
for _fn, _keys in SCORE_FN_OUTPUTS.items():
    for _k in _keys:
        METRIC_TO_FN[_k] = _fn

def needed_score_fns_for_metrics(metrics: Sequence[str]) -> List[str]:
    """Return minimal set of score fn names to produce given metrics."""
    return sorted({METRIC_TO_FN[m] for m in metrics if m in METRIC_TO_FN})
