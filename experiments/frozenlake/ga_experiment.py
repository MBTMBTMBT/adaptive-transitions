#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_full_experiment.py
# English comments only.

from __future__ import annotations

import argparse
import json
import os
import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import wandb

from customised_toy_text_envs.customised_frozenlake import (
    CustomisedFrozenLakeEnv,
    plot_frozenlake_transition_overlays,
    plot_frozenlake_scalar_overlay,
    plot_frozenlake_scalar_diff_overlay,
)
from mdp_network.mdp_network import MDPNetwork
from mdp_network.mdp_tables import q_table_to_policy, create_random_policy
from mdp_network.solvers import (
    optimal_value_iteration,
    compute_occupancy_measure,
    policy_evaluation,
)
from mdp_network.ga_mdp_search import (
    GAConfig,
    MDPEvolutionGA,
    register_score_fn,
    evaluate_mdp_objectives,
    obj_multi_perf,
)

from expetiment_utils.tabular_curriculum_trainer import (
    EnvFactorySpec,
    SourceFactorySpec,
    PhaseSpec,
    EvalSpec,
    TabularCurriculumTrainer as TrainerClass,
)

# -------- Fixed factory dotted paths (must exist as real module) --------
TARGET_FACTORY_PATH = "expetiment_utils.env_factories:make_frozenlake_target"
SOURCE_FACTORY_PATH = "expetiment_utils.env_factories:make_nx_env_from_mdp"


# =============================================================================
# Utilities
# =============================================================================

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "t", "yes", "y", "on")


def _parse_csv_numbers(s: str, typ=float) -> List:
    if s is None or s == "":
        return []
    return [typ(x.strip()) for x in s.split(",") if x.strip() != ""]


def _parse_tuple3(s: str) -> Tuple[float, float, float]:
    parts = _parse_csv_numbers(s, float)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated floats, e.g. '1.0,0.0,0.0'")
    return tuple(parts)  # type: ignore


def _now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_json(path: Path, obj: Any):
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _wandb_init(args) -> "wandb.sdk.wandb_run.Run":
    if args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
        print("[W&B] Offline mode.")
    elif args.wandb_mode != "online":
        raise ValueError("--wandb-mode must be 'online' or 'offline'")

    name = args.run_name or f"fullexp_{args.map}_{'slip' if args.slippery else 'noslip'}_" \
           f"ph{'-'.join(str(x) for x in args.phase_steps)}_seeds{len(args.train_seeds)}_{_now_tag()}"

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=name,
        job_type="full-exp",
        config={},  # updated below
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    cfg = {k: getattr(args, k) for k in vars(args)}
    run.config.update(cfg, allow_val_change=True)
    return run


def _wandb_log_image(run, key: str, path: Path):
    run.log({key: wandb.Image(str(path))})


# =============================================================================
# GA Stage
# =============================================================================

def _build_native_mdp(map_name: str, slippery: bool) -> MDPNetwork:
    env = CustomisedFrozenLakeEnv(render_mode=None, map_name=map_name, is_slippery=slippery)
    env.reset(seed=0)
    return env.get_mdp_network()


def stage_ga(args, run) -> List[Path]:
    out_dir = Path(args.outdir) / "ga"
    mdp_out_dir = out_dir / "mdps"
    _ensure_dir(mdp_out_dir)

    if args.skip_ga and mdp_out_dir.exists():
        files = sorted(mdp_out_dir.glob("*.json"))
        if files:
            print(f"[GA] Skipped (existing {len(files)} JSON found).")
            return files

    print("[GA] Building native MDP…")
    mdp = _build_native_mdp(args.map, args.slippery)
    workers = args.ga_workers or (os.cpu_count() or 1)

    register_score_fn("obj_multi_perf", obj_multi_perf)

    cfg = GAConfig(
        population_size=args.ga_pop_size,
        generations=args.ga_generations,
        tournament_k=args.ga_tournament_k,
        elitism_num=args.ga_elitism,
        crossover_rate=args.ga_crossover,

        allow_self_loops=True,
        min_out_degree=1,
        max_out_degree=4,
        prob_floor=1e-6,
        add_edge_attempts_per_child=10,
        epsilon_new_prob=0.1,
        gamma_sample=1.0,
        gamma_prob=0.0,
        prune_prob_threshold=1e-3,
        prob_tweak_actions_per_child=50,
        prob_pairwise_step=0.05,
        reward_tweak_edges_per_child=args.ga_reward_tweak_edges_per_child,
        reward_k_percent=args.ga_reward_k_percent,
        reward_ref_floor=1e-3,
        add_edge_allow_out_of_scope=False,

        n_workers=workers,
        score_fn_names=["obj_multi_perf"],
        score_args=None,
        score_kwargs={
            "policy_mixing": tuple(args.ga_policy_mix),
            "policy_tie_tol": args.ga_tie_tol,
            "blend_weight": args.ga_blend_weight,
        },

        mutation_n_workers=workers,

        dist_max_hops=10,
        dist_node_cap=64,
        dist_weight_eps=1e-6,
        dist_unreachable=1e9,

        vi_gamma=args.ga_vi_gamma,
        vi_theta=args.ga_vi_theta,
        vi_max_iterations=args.ga_vi_max_iters,
        policy_temperature=args.ga_policy_temperature,
        perf_numpoints=args.ga_perf_numpoints,
        perf_gamma=args.ga_perf_gamma if args.ga_perf_gamma is not None else args.ga_vi_gamma,
        perf_theta=args.ga_perf_theta if args.ga_perf_theta is not None else args.ga_vi_theta,
        perf_max_iterations=args.ga_perf_max_iters if args.ga_perf_max_iters is not None else args.ga_vi_max_iters,

        seed=args.ga_seed,
    )

    run.config.update({"ga_config": asdict(cfg)}, allow_val_change=True)

    print("[GA] Precomputing baseline policy & occupancy…")
    _, Q = optimal_value_iteration(mdp, gamma=cfg.vi_gamma, theta=cfg.vi_theta, max_iterations=cfg.vi_max_iterations)
    base_policy = q_table_to_policy(
        Q,
        states=list(mdp.states),
        num_actions=mdp.num_actions,
        mixing=tuple(args.ga_policy_mix),
        temperature=cfg.policy_temperature,
        tie_tol=args.ga_tie_tol,
    )
    base_occupancy = compute_occupancy_measure(
        mdp, base_policy, gamma=cfg.vi_gamma, theta=cfg.vi_theta, max_iterations=cfg.vi_max_iterations
    )

    ga = MDPEvolutionGA(base_mdp=mdp, cfg=cfg, wb_run=run)
    ga.precomputed_artifacts = [base_policy, base_occupancy]

    if args.ga_sanity_batch > 0:
        print("[GA] Sanity check evaluate_mdp_objectives…")
        batch = [mdp] + [mdp.clone() for _ in range(args.ga_sanity_batch - 1)]
        obj_vecs = evaluate_mdp_objectives(
            batch,
            score_fn_names=cfg.score_fn_names or [],
            n_workers=cfg.n_workers,
            score_args=cfg.score_args,
            score_kwargs={
                "vi_gamma": cfg.vi_gamma,
                "vi_theta": cfg.vi_theta,
                "vi_max_iterations": cfg.vi_max_iterations,
                "policy_temperature": cfg.policy_temperature,
                "policy_mixing": tuple(args.ga_policy_mix),
                "policy_tie_tol": args.ga_tie_tol,
                "perf_numpoints": cfg.perf_numpoints,
                "perf_gamma": cfg.perf_gamma,
                "perf_theta": cfg.perf_theta,
                "perf_max_iterations": cfg.perf_max_iterations,
            },
            precomputed_portables=[base_policy.to_portable(), base_occupancy.to_portable()],
        )
        print("  Batch objective vectors (head):", [[round(x, 6) for x in v] for v in obj_vecs[:3]])

    print("[GA] Running NSGA-II…")
    pareto_mdps, pareto_objs, pop, _ = ga.run()
    print(f"[GA] Pareto front size = {len(pareto_mdps)}, population size = {len(pop)}")

    saved = []
    _ensure_dir(mdp_out_dir)
    for i, m in enumerate(pareto_mdps):
        tag = "_".join(f"{v:.4f}" for v in pareto_objs[i])
        out_path = mdp_out_dir / f"pareto_{i}_objs_{tag}.json"
        m.export_to_json(str(out_path))
        saved.append(out_path)
        print(f"[GA] Saved PF[{i}] -> {out_path.name}")

    return saved


# =============================================================================
# Training Stage (Curriculum)
# =============================================================================

def stage_train(args, run, json_files: List[Path]) -> Dict[str, Any]:
    trainer_out = Path(args.outdir) / "trainer"
    _ensure_dir(trainer_out)

    target_env_spec = EnvFactorySpec(
        factory_path=TARGET_FACTORY_PATH,
        kwargs=dict(map_name=args.map, is_slippery=bool(args.slippery), max_steps=int(args.max_steps)),
    ).as_dict()

    baseline_phases: List[PhaseSpec] = [
        PhaseSpec(name=f"Phase-{i}(Target)", steps=int(steps), env_spec=target_env_spec)
        for i, steps in enumerate(args.phase_steps)
    ]

    item_phase_specs_map: Dict[str, List[PhaseSpec]] = {}
    eval_specs_map: Dict[str, List[EvalSpec]] = {}
    for p in json_files:
        label = p.stem
        source_env_spec = SourceFactorySpec(
            factory_path=SOURCE_FACTORY_PATH,
            mdp_config_path=str(p),
            kwargs=dict(max_steps=int(args.max_steps)),
        ).as_dict()
        phases = [PhaseSpec(name="Phase-0(Source)", steps=int(args.phase_steps[0]), env_spec=source_env_spec)]
        for i, steps in enumerate(args.phase_steps[1:], start=1):
            phases.append(PhaseSpec(name=f"Phase-{i}(Target)", steps=int(steps), env_spec=target_env_spec))
        item_phase_specs_map[label] = phases

        eval_specs_map[label] = [
            EvalSpec(name="Target", env_spec=target_env_spec, eval_seed_base=args.eval_seed_base_target),
            EvalSpec(name="Source-A", env_spec=source_env_spec, eval_seed_base=args.eval_seed_base_source),
        ]

    trainer = TrainerClass(
        agent_ctor_path=args.agent_ctor,
        agent_kwargs=args.agent_kwargs,
        eval_every=args.eval_every,
        n_eval_episodes=args.n_eval_episodes,
        output_dir=str(trainer_out),
        wandb_run=run,
        max_workers=args.train_workers or None,
    )

    aggregated = trainer.run(
        seeds=args.train_seeds,
        baseline_phase_specs=baseline_phases,
        item_phase_specs_map=item_phase_specs_map,
        eval_specs_map=eval_specs_map,
    )

    _save_json(Path(args.outdir) / "meta" / "trainer_meta.json", aggregated)
    return aggregated


# =============================================================================
# Visualization Stage (Overlays)
# =============================================================================

def _states_aligned(env: CustomisedFrozenLakeEnv, mdp: MDPNetwork) -> bool:
    nS = env.nrow * env.ncol
    return set(mdp.states) == set(range(nS))


def stage_visualize(args, run, json_files: List[Path]):
    vis_out = Path(args.outdir) / "vis"
    _ensure_dir(vis_out)

    env = CustomisedFrozenLakeEnv(render_mode="rgb_array", map_name=args.map, is_slippery=bool(args.slippery))
    env.reset()

    native_mdp = None
    native_occ_random = None
    native_V_opt_greedy = None

    if args.vis_include_native:
        native_mdp = env.get_mdp_network()

        native_policy_rand = create_random_policy(native_mdp)
        native_occ_random = compute_occupancy_measure(
            mdp_network=native_mdp, policy=native_policy_rand,
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )

        _, Q_star = optimal_value_iteration(
            mdp_network=native_mdp, gamma=args.vis_gamma, theta=args.vis_theta,
            max_iterations=args.vis_max_iters, verbose=False,
        )
        native_policy_opt_greedy = q_table_to_policy(
            q_table=Q_star, states=native_mdp.states, num_actions=native_mdp.num_actions,
            mixing=(1.0, 0.0, 0.0), temperature=1.0, tie_tol=args.vis_tie_tol,
        )
        native_V_opt_greedy = policy_evaluation(
            mdp_network=native_mdp, policy=native_policy_opt_greedy,
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )

        native_out = vis_out / "__native_frozenlake__"
        _ensure_dir(native_out)

        plot_frozenlake_transition_overlays(
            env=env, mdp=native_mdp, output_dir=str(native_out), filename_prefix="native_frozenlake",
            min_prob=args.vis_min_prob, alpha=args.vis_alpha, annotate=True,
            show_self_loops=args.vis_show_self_loops, dpi=args.vis_dpi,
        )

        plot_frozenlake_scalar_overlay(
            env=env, value_map=native_occ_random, output_dir=str(native_out),
            filename_prefix="native_frozenlake_occupancy_random",
            alpha=args.vis_occ_alpha, annotate=True, dpi=args.vis_dpi,
            target_cell_px=args.vis_occ_cell_px, font_scale=args.vis_occ_font_scale,
            cmap_name=args.vis_occ_cmap, gamma=args.vis_occ_gamma,
            min_abs_label=0.0, vmin=0.0, vmax=None,
            title="State Occupancy — Random", cbar_label="Occupancy measure",
            value_format=None,
        )

        V_rand = policy_evaluation(
            mdp_network=native_mdp, policy=create_random_policy(native_mdp),
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )
        plot_frozenlake_scalar_overlay(
            env=env, value_map=V_rand, output_dir=str(native_out),
            filename_prefix="native_frozenlake_VALUE_random",
            alpha=args.vis_val_alpha, annotate=True, dpi=args.vis_dpi,
            target_cell_px=args.vis_val_cell_px, font_scale=args.vis_val_font_scale,
            cmap_name=args.vis_val_cmap, gamma=args.vis_val_gamma,
            min_abs_label=0.0, vmin=None, vmax=None,
            title="State Value V(s) — Random", cbar_label="V(s)",
            value_format=None,
        )
        plot_frozenlake_scalar_overlay(
            env=env, value_map=native_V_opt_greedy, output_dir=str(native_out),
            filename_prefix="native_frozenlake_VALUE_optimal_greedy",
            alpha=args.vis_val_alpha, annotate=True, dpi=args.vis_dpi,
            target_cell_px=args.vis_val_cell_px, font_scale=args.vis_val_font_scale,
            cmap_name=args.vis_val_cmap, gamma=args.vis_val_gamma,
            min_abs_label=0.0, vmin=None, vmax=None,
            title="State Value V(s) — Optimal (greedy)", cbar_label="V(s)",
            value_format=None,
        )

        for fn in sorted(native_out.glob("*.png"))[:6]:
            _wandb_log_image(run, f"images/vis/native/{fn.stem}", fn)

    if not json_files:
        print("[VIS] No JSON files; skip visualization on loops.")
        return

    for jf in json_files:
        cfg = _load_json(jf)
        mdp = MDPNetwork(config_data=cfg)

        nS = env.nrow * env.ncol
        if getattr(mdp, "num_actions", None) != 4:
            print(f"[VIS] Skip (num_actions != 4): {jf.name}")
            continue
        if not _states_aligned(env, mdp):
            print(f"[VIS] Skip (states not aligned 0..{nS-1}): {jf.name}")
            continue

        stem = jf.stem
        out_dir = vis_out / stem
        _ensure_dir(out_dir)

        plot_frozenlake_transition_overlays(
            env=env, mdp=mdp, output_dir=str(out_dir), filename_prefix=stem,
            min_prob=args.vis_min_prob, alpha=args.vis_alpha, annotate=True,
            show_self_loops=args.vis_show_self_loops, dpi=args.vis_dpi,
        )

        policy_rand = create_random_policy(mdp)
        occ_rand = compute_occupancy_measure(
            mdp_network=mdp, policy=policy_rand,
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )

        _, Q_star = optimal_value_iteration(
            mdp_network=mdp, gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )

        policy_opt_mixed = q_table_to_policy(
            q_table=Q_star, states=mdp.states, num_actions=mdp.num_actions,
            mixing=tuple(args.vis_mix_loop), temperature=args.vis_temperature, tie_tol=args.vis_tie_tol,
        )
        occ_opt_mixed = compute_occupancy_measure(
            mdp_network=mdp, policy=policy_opt_mixed,
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )

        policy_opt_greedy = q_table_to_policy(
            q_table=Q_star, states=mdp.states, num_actions=mdp.num_actions,
            mixing=(1.0, 0.0, 0.0), temperature=1.0, tie_tol=args.vis_tie_tol,
        )
        V_opt_greedy = policy_evaluation(
            mdp_network=mdp, policy=policy_opt_greedy,
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )

        plot_frozenlake_scalar_overlay(
            env=env, value_map=occ_rand, output_dir=str(out_dir),
            filename_prefix=f"{stem}_occupancy_random",
            alpha=args.vis_occ_alpha, annotate=True, dpi=args.vis_dpi,
            target_cell_px=args.vis_occ_cell_px, font_scale=args.vis_occ_font_scale,
            cmap_name=args.vis_occ_cmap, gamma=args.vis_occ_gamma,
            min_abs_label=0.0, vmin=0.0, vmax=None,
            title="State Occupancy", cbar_label="Occupancy measure",
            value_format=None,
        )

        mix_suffix = f"mix_g{args.vis_mix_loop[0]:.2f}_s{args.vis_mix_loop[1]:.2f}_u{args.vis_mix_loop[2]:.2f}" + \
                     (f"_T{args.vis_temperature:g}" if args.vis_mix_loop[1] > 0.0 else "")
        plot_frozenlake_scalar_overlay(
            env=env, value_map=occ_opt_mixed, output_dir=str(out_dir),
            filename_prefix=f"{stem}_occupancy_optimal_{mix_suffix}",
            alpha=args.vis_occ_alpha, annotate=True, dpi=args.vis_dpi,
            target_cell_px=args.vis_occ_cell_px, font_scale=args.vis_occ_font_scale,
            cmap_name=args.vis_occ_cmap, gamma=args.vis_occ_gamma,
            min_abs_label=0.0, vmin=0.0, vmax=None,
            title="State Occupancy — Optimal (mixed)", cbar_label="Occupancy measure",
            value_format=None,
        )

        V_rand = policy_evaluation(
            mdp_network=mdp, policy=policy_rand,
            gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
        )
        plot_frozenlake_scalar_overlay(
            env=env, value_map=V_rand, output_dir=str(out_dir),
            filename_prefix=f"{stem}_VALUE_random",
            alpha=args.vis_val_alpha, annotate=True, dpi=args.vis_dpi,
            target_cell_px=args.vis_val_cell_px, font_scale=args.vis_val_font_scale,
            cmap_name=args.vis_val_cmap, gamma=args.vis_val_gamma,
            min_abs_label=0.0, vmin=None, vmax=None,
            title="State Value V(s) — Random", cbar_label="V(s)",
            value_format=None,
        )
        plot_frozenlake_scalar_overlay(
            env=env, value_map=V_opt_greedy, output_dir=str(out_dir),
            filename_prefix=f"{stem}_VALUE_optimal_greedy",
            alpha=args.vis_val_alpha, annotate=True, dpi=args.vis_dpi,
            target_cell_px=args.vis_val_cell_px, font_scale=args.vis_val_font_scale,
            cmap_name=args.vis_val_cmap, gamma=args.vis_val_gamma,
            min_abs_label=0.0, vmin=None, vmax=None,
            title="State Value V(s) — Optimal (greedy)", cbar_label="V(s)",
            value_format=None,
        )

        if native_mdp is not None:
            occ_cross_native = compute_occupancy_measure(
                mdp_network=native_mdp, policy=policy_opt_mixed,
                gamma=args.vis_gamma, theta=args.vis_theta, max_iterations=args.vis_max_iters, verbose=False,
            )
            plot_frozenlake_scalar_overlay(
                env=env, value_map=occ_cross_native, output_dir=str(out_dir),
                filename_prefix=f"{stem}_occupancy_optPolicy_on_NATIVE_{mix_suffix}",
                alpha=args.vis_occ_alpha, annotate=True, dpi=args.vis_dpi,
                target_cell_px=args.vis_occ_cell_px, font_scale=args.vis_occ_font_scale,
                cmap_name=args.vis_occ_cmap, gamma=args.vis_occ_gamma,
                min_abs_label=0.0, vmin=0.0, vmax=None,
                title="State Occupancy — Policy (learned) on NATIVE", cbar_label="Occupancy measure",
                value_format=None,
            )
            if native_occ_random is not None:
                plot_frozenlake_scalar_diff_overlay(
                    env=env,
                    values_a=occ_cross_native,
                    values_b=native_occ_random,
                    output_dir=str(out_dir),
                    filename_prefix=f"{stem}_occupancy_DIFF_optPolicyMINUS_nativeRandom_{mix_suffix}",
                    alpha=args.vis_occ_alpha, annotate=True, dpi=args.vis_dpi,
                    target_cell_px=args.vis_occ_cell_px, font_scale=args.vis_occ_font_scale,
                    cmap_name="coolwarm", min_abs_label=0.0, vmin=None, vmax=None,
                    title="Δ State Occupancy (A − B)", cbar_label="Δ occupancy (A − B)",
                    value_format="+.2e",
                )

        if native_V_opt_greedy is not None:
            plot_frozenlake_scalar_diff_overlay(
                env=env,
                values_a=V_opt_greedy,
                values_b=native_V_opt_greedy,
                output_dir=str(out_dir),
                filename_prefix=f"{stem}_VALUE_DIFF_optGreedyMINUS_nativeOptGreedy",
                alpha=args.vis_val_alpha, annotate=True, dpi=args.vis_dpi,
                target_cell_px=args.vis_val_cell_px, font_scale=args.vis_val_font_scale,
                cmap_name="coolwarm", min_abs_label=0.0, vmin=None, vmax=None,
                title="Δ State Value: optGreedy(loop) − optGreedy(native)",
                cbar_label="Δ V(s) (loop − native)",
                value_format="+.2f",
            )

        for fn in sorted(out_dir.glob("*.png"))[:8]:
            _wandb_log_image(run, f"images/vis/{stem}/{fn.stem}", fn)


# =============================================================================
# Argparse / main
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end GA → Curriculum → Visualization with W&B (images only).")

    # W&B
    p.add_argument("--outdir", type=str, default="./outputs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default="full-frozenlake")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, choices=["online", "offline"], default="online")

    # Pipeline toggles
    p.add_argument("--skip-ga", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-vis", action="store_true")

    # Env
    p.add_argument("--map", type=str, default="8x8")
    p.add_argument("--slippery", type=_str2bool, default=True)
    p.add_argument("--max-steps", type=int, default=500)

    # GA
    p.add_argument("--ga-pop-size", type=int, default=25)
    p.add_argument("--ga-generations", type=int, default=25)
    p.add_argument("--ga-tournament-k", type=int, default=2)
    p.add_argument("--ga-elitism", type=int, default=5)
    p.add_argument("--ga-crossover", type=float, default=0.5)
    p.add_argument("--ga-workers", type=int, default=0)
    p.add_argument("--ga-policy-mix", type=_parse_tuple3, default=(0.9, 0.0, 0.1))
    p.add_argument("--ga-policy-temperature", type=float, default=0.01)
    p.add_argument("--ga-tie-tol", type=float, default=1e-2)
    p.add_argument("--ga-blend-weight", type=float, default=0.8)
    p.add_argument("--ga-vi-gamma", type=float, default=0.99)
    p.add_argument("--ga-vi-theta", type=float, default=1e-3)
    p.add_argument("--ga-vi-max-iters", type=int, default=1000)
    p.add_argument("--ga-perf-numpoints", type=int, default=32)
    p.add_argument("--ga-perf-gamma", type=float, default=None)
    p.add_argument("--ga-perf-theta", type=float, default=None)
    p.add_argument("--ga-perf-max-iters", type=int, default=None)
    p.add_argument("--ga-reward-tweak-edges-per-child", type=int, default=0)
    p.add_argument("--ga-reward-k-percent", type=float, default=0.05)
    p.add_argument("--ga-sanity-batch", type=int, default=3)
    p.add_argument("--ga-seed", type=int, default=4444)

    # Training
    p.add_argument("--agent-ctor", type=str, default="simple_agents.tabular_q_agent:TabularQAgent")
    p.add_argument("--agent-kwargs", type=str, default="")
    p.add_argument("--phase-steps", type=str, default="10000,140000")
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--n-eval-episodes", type=int, default=100)
    p.add_argument("--train-seeds", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--train-workers", type=int, default=0)
    p.add_argument("--eval-seed-base-target", type=int, default=10000)
    p.add_argument("--eval-seed-base-source", type=int, default=20000)
    p.add_argument("--json-dir", type=str, default="")
    p.add_argument("--json-max", type=int, default=0)

    # Visualization
    p.add_argument("--vis-include-native", type=_str2bool, default=True)
    p.add_argument("--vis-min-prob", type=float, default=0.05)
    p.add_argument("--vis-alpha", type=float, default=0.65)
    p.add_argument("--vis-show-self-loops", type=_str2bool, default=False)
    p.add_argument("--vis-dpi", type=int, default=200)
    p.add_argument("--vis-gamma", type=float, default=0.99)
    p.add_argument("--vis-theta", type=float, default=1e-6)
    p.add_argument("--vis-max-iters", type=int, default=1000)
    p.add_argument("--vis-temperature", type=float, default=1.0)
    p.add_argument("--vis-mix-native", type=_parse_tuple3, default=(1.0, 0.0, 0.0))
    p.add_argument("--vis-mix-loop", type=_parse_tuple3, default=(0.9, 0.0, 0.1))
    p.add_argument("--vis-tie-tol", type=float, default=1e-2)
    p.add_argument("--vis-occ-alpha", type=float, default=0.65)
    p.add_argument("--vis-occ-cell-px", type=int, default=240)
    p.add_argument("--vis-occ-font-scale", type=float, default=0.16)
    p.add_argument("--vis-occ-cmap", type=str, default="magma")
    p.add_argument("--vis-occ-gamma", type=float, default=1.0)
    p.add_argument("--vis-val-alpha", type=float, default=0.65)
    p.add_argument("--vis-val-cell-px", type=int, default=240)
    p.add_argument("--vis-val-font-scale", type=float, default=0.16)
    p.add_argument("--vis-val-cmap", type=str, default="viridis")
    p.add_argument("--vis-val-gamma", type=float, default=1.0)
    return p


def _resolve_args(p: argparse.ArgumentParser) -> argparse.Namespace:
    args = p.parse_args()

    if args.agent_kwargs:
        args.agent_kwargs = json.loads(args.agent_kwargs)
    else:
        args.agent_kwargs = dict(
            learning_rate=0.1,
            gamma=0.99,
            policy_mix=(0.9, 0.0, 0.1),
            temperature=0.01,
            tie_tol=1e-2,
            verbose=0,
        )

    args.train_seeds = [int(x) for x in _parse_csv_numbers(args.train_seeds, int)]
    args.phase_steps = [int(x) for x in _parse_csv_numbers(args.phase_steps, int)]
    if len(args.phase_steps) < 2:
        raise SystemExit("--phase-steps requires at least 2 phases (Source then Target).")

    _ensure_dir(Path(args.outdir) / "meta")
    return args


def main():
    parser = build_arg_parser()
    args = _resolve_args(parser)

    run = _wandb_init(args)

    _save_json(Path(args.outdir) / "meta" / "config.json", {k: getattr(args, k) for k in vars(args)})

    # GA Stage
    if args.json_dir:
        json_dir = Path(args.json_dir)
        json_files = sorted(json_dir.glob("*.json"))
        if args.json_max > 0:
            json_files = json_files[:args.json_max]
        print(f"[MAIN] Using external JSON dir: {json_dir} ({len(json_files)} files).")
    else:
        if not args.skip_ga:
            json_files = stage_ga(args, run)
        else:
            mdp_out_dir = Path(args.outdir) / "ga" / "mdps"
            json_files = sorted(mdp_out_dir.glob("*.json"))
            if args.json_max > 0:
                json_files = json_files[:args.json_max]
            print(f"[MAIN] GA skipped; using {len(json_files)} JSON from {mdp_out_dir}.")

    if args.json_max > 0:
        json_files = json_files[:args.json_max]

    # Training Stage
    if not args.skip_train and json_files:
        _ = stage_train(args, run, json_files)
    else:
        print("[MAIN] Training skipped or no JSON files; skipping trainer stage.")

    # Visualization Stage
    if not args.skip_vis:
        stage_visualize(args, run, json_files)
    else:
        print("[MAIN] Visualization skipped.")

    print("\n[MAIN] All done.")
    run.finish()


if __name__ == "__main__":
    main()
