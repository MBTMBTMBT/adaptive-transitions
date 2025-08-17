from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from experiment_utils.utils import _ensure_dir, _save_json
from two_stage_cl.tabular_curriculum_trainer import EnvFactorySpec, PhaseSpec, EvalSpec, SourceFactorySpec, \
    TabularCurriculumTrainer as TrainerClass


def stage_train(
    args,
    run,
    json_files: List[Path],
    *,
    target_factory_path: str,
    target_factory_kwargs: Dict[str, Any],
    source_factory_path: str,
    source_env_base_kwargs: Dict[str, Any],
    phase_steps: List[int],
    eval_seed_base_target: int,
    eval_seed_base_source: int,
    target_label: str = "Target",
    source_label: str = "Source-A",
) -> Dict[str, Any]:
    """
    Environment-agnostic training stage.

    This function:
      1) Builds the target env spec from (factory_path, kwargs).
      2) Builds baseline phases that all use the target env spec.
      3) For each JSON (MDP config), builds a Source env spec, a per-item Phase-0(Source) + remaining Target phases,
         and per-item Eval specs for Target and Source.
      4) Runs the curriculum trainer and saves aggregated results.

    All environment details are passed as parameters; the function itself is agnostic.
    """
    trainer_out = Path(args.outdir) / "trainer"
    _ensure_dir(trainer_out)

    # Build target env spec (generic)
    target_env_spec = EnvFactorySpec(
        factory_path=target_factory_path,
        kwargs=dict(**target_factory_kwargs),
    ).as_dict()

    # Baseline phases (all Target)
    baseline_phase_specs: List[PhaseSpec] = [
        PhaseSpec(name=f"Phase-{i}({target_label})", steps=int(steps), env_spec=target_env_spec)
        for i, steps in enumerate(phase_steps)
    ]

    # Per-item phases and evals using Source (from each JSON) + Target
    item_phase_specs_map: Dict[str, List[PhaseSpec]] = {}
    eval_specs_map: Dict[str, List[EvalSpec]] = {}

    for p in json_files:
        label = p.stem

        source_env_spec = SourceFactorySpec(
            factory_path=source_factory_path,
            mdp_config_path=str(p),
            kwargs=dict(**source_env_base_kwargs),
        ).as_dict()

        phases = [PhaseSpec(name="Phase-0(Source)", steps=int(phase_steps[0]), env_spec=source_env_spec)]
        for i, steps in enumerate(phase_steps[1:], start=1):
            phases.append(PhaseSpec(name=f"Phase-{i}({target_label})", steps=int(steps), env_spec=target_env_spec))
        item_phase_specs_map[label] = phases

        eval_specs_map[label] = [
            EvalSpec(name=target_label, env_spec=target_env_spec, eval_seed_base=eval_seed_base_target),
            EvalSpec(name=source_label, env_spec=source_env_spec, eval_seed_base=eval_seed_base_source),
        ]

    # Trainer is environment-agnostic; it only consumes specs
    trainer = TrainerClass(
        agent_ctor_path="simple_agents.tabular_q_agent:TabularQAgent",  # fixed
        agent_kwargs=args.agent_kwargs,
        eval_every=args.eval_every,
        n_eval_episodes=args.n_eval_episodes,
        output_dir=str(trainer_out),
        wandb_run=run,
        max_workers=args.train_workers or None,
    )

    aggregated = trainer.run(
        seeds=args.train_seeds,
        baseline_phase_specs=baseline_phase_specs,
        item_phase_specs_map=item_phase_specs_map,
        eval_specs_map=eval_specs_map,
    )

    _save_json(Path(args.outdir) / "meta" / "trainer_meta.json", aggregated)
    return aggregated
