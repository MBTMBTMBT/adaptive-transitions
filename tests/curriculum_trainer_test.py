# test_curriculum_trainer_frozenlake.py
# English comments only.

import os
from pathlib import Path
from typing import List, Dict
from gymnasium.wrappers import TimeLimit
import datetime

from expetiment_utils.tabular_curriculum_trainer import (
    EnvFactorySpec,
    PhaseSpec,
    EvalSpec,
    SourceFactorySpec,
    GenericCurriculumTrainer,
)

# -----------------------------
# User knobs
# -----------------------------
OUTPUT_DIR = "./outputs/ga_cl_generic"
JSON_DIR = "./outputs/ga_test"     # folder containing multiple *.json MDPs

# Seeds and schedule
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
EVAL_EVERY = 2_500
N_EVAL_EPISODES = 100

# Two-phase schedule (A->B). You can extend to 3+ phases easily.
STEPS_JSON_PHASE = 10_000  # Phase A
STEPS_TARGET_PHASE = 140_000  # Phase B

# Agent (dotted path + kwargs). Keep SB3-like signature.
AGENT_CTOR_PATH = "simple_agents.tabular_q_agent:TabularQAgent"
AGENT_KW = dict(
    learning_rate=0.1,
    gamma=0.99,
    policy_mix=(0.9, 0.0, 0.1),
    temperature=0.01,
    tie_tol=1e-2,
    verbose=0,
)

# FrozenLake specifics (for example only)
FROZENLAKE_MAP = "8x8"
FROZENLAKE_IS_SLIPPERY = True
FROZENLAKE_MAX_STEPS = 500

# If you already have a wandb run from outside, set WANDB_RUN to it, otherwise keep None.
WANDB_RUN = None


# -----------------------------
# Env factories (dotted-path targets)
# -----------------------------
def make_frozenlake_target(seed: int, **kwargs):
    """
    Baseline FrozenLake env builder.
    Signature must be: (seed: int, **kwargs) -> gym.Env
    """
    from customised_toy_text_envs.customised_frozenlake import CustomisedFrozenLakeEnv
    env = CustomisedFrozenLakeEnv(
        render_mode=None,
        map_name=kwargs.get("map_name", "8x8"),
        is_slippery=kwargs.get("is_slippery", True),
        networkx_env=None,
    )
    env = TimeLimit(env, max_episode_steps=kwargs.get("max_steps", 500))
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_nx_env_from_mdp(mdp, seed: int, **kwargs):
    """
    Source env builder backed by an MDPNetwork.
    Signature must be: (mdp, seed: int, **kwargs) -> gym.Env
    """
    from networkx_env.networkx_env import NetworkXMDPEnvironment
    from gymnasium.wrappers import TimeLimit as TL
    env = NetworkXMDPEnvironment(mdp_network=mdp, render_mode=None, seed=seed)
    env = TL(env, max_episode_steps=kwargs.get("max_steps", 500))
    if seed is not None:
        env.reset(seed=seed)
    return env


# Expose factories via dotted paths for the trainer (module:callable).
# IMPORTANT:
# - If this file is executed as a module within a package (e.g. `python -m tests.test_curriculum_trainer_frozenlake`),
#   __name__ will be the importable module path and this dotted path will work across subprocesses.
# - If you run as a top-level script and __name__ == "__main__", the dotted path will NOT be importable in workers.
#   In that case, move these factories into a real module (e.g., expetiment_utils.env_factories) and update paths.
TARGET_FACTORY_PATH = __name__ + ":make_frozenlake_target"
SOURCE_FACTORY_PATH = __name__ + ":make_nx_env_from_mdp"


def _maybe_init_wandb():
    """
    Initialize a wandb run if WANDB_RUN is None. Uses env vars to configure project/entity/name.
    Returns a wandb run or None if wandb is unavailable.
    """
    if WANDB_RUN is not None:
        return WANDB_RUN

    try:
        import wandb  # noqa: F401
    except Exception:
        print("[W&B] wandb not available; proceeding without logging.")
        return None

    import wandb
    project = os.environ.get("WANDB_PROJECT", "curriculum-frozenlake")
    entity = os.environ.get("WANDB_ENTITY", None)  # optional
    run_name = os.environ.get("WANDB_RUN_NAME", None)
    if run_name is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"frozenlake_cl_{FROZENLAKE_MAP}_slip{int(FROZENLAKE_IS_SLIPPERY)}_" \
                   f"steps{STEPS_JSON_PHASE}-{STEPS_TARGET_PHASE}_seeds{len(SEEDS)}_{ts}"

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        job_type="curriculum-train",
        config={
            "seeds": SEEDS,
            "eval_every": EVAL_EVERY,
            "n_eval_episodes": N_EVAL_EPISODES,
            "steps_json_phase": STEPS_JSON_PHASE,
            "steps_target_phase": STEPS_TARGET_PHASE,
            "agent_ctor_path": AGENT_CTOR_PATH,
            "agent_kwargs": AGENT_KW,
            "frozenlake": {
                "map": FROZENLAKE_MAP,
                "is_slippery": FROZENLAKE_IS_SLIPPERY,
                "max_steps": FROZENLAKE_MAX_STEPS,
            },
            "json_dir": str(JSON_DIR),
            "output_dir": str(OUTPUT_DIR),
        },
        reinit=True,
        settings=wandb.Settings(start_method="fork")  # safe for Linux; use "thread" if needed.
    )
    return run


def _log_outputs_as_artifact(wandb_run, output_dir: str, boundaries: List[int]):
    """
    Log the entire output directory as a W&B artifact for reproducibility.
    """
    if wandb_run is None:
        return
    try:
        import wandb
        boundaries_str = "-".join(str(b) for b in boundaries) if boundaries else "none"
        art_name = f"curriculum_outputs_boundaries_{boundaries_str}"
        art = wandb.Artifact(name=art_name, type="results")
        art.add_dir(output_dir)
        wandb_run.log_artifact(art)
    except Exception as e:
        print(f"[W&B] artifact logging failed: {e}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover JSON MDPs
    json_paths = sorted(Path(JSON_DIR).glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No *.json found in {JSON_DIR}")

    # ---------- Build generic specs ----------
    target_env_spec = EnvFactorySpec(
        factory_path=TARGET_FACTORY_PATH,
        kwargs=dict(map_name=FROZENLAKE_MAP, is_slippery=FROZENLAKE_IS_SLIPPERY, max_steps=FROZENLAKE_MAX_STEPS),
    ).as_dict()

    # Baseline: both phases on Target (boundaries still exist and show in plots)
    baseline_phase_specs: List[PhaseSpec] = [
        PhaseSpec(name="Phase-A(Target)", steps=STEPS_JSON_PHASE, env_spec=target_env_spec),
        PhaseSpec(name="Phase-B(Target)", steps=STEPS_TARGET_PHASE, env_spec=target_env_spec),
    ]

    # Curriculum items: per JSON label, Phase-A(Source mdp) -> Phase-B(Target)
    item_phase_specs_map: Dict[str, List[PhaseSpec]] = {}
    eval_specs_map: Dict[str, List[EvalSpec]] = {}
    for p in json_paths:
        label = p.stem
        source_env_spec = SourceFactorySpec(
            factory_path=SOURCE_FACTORY_PATH,
            mdp_config_path=str(p),
            kwargs=dict(max_steps=FROZENLAKE_MAX_STEPS),
        ).as_dict()
        item_phase_specs_map[label] = [
            PhaseSpec(name="Phase-A(Source)", steps=STEPS_JSON_PHASE, env_spec=source_env_spec),
            PhaseSpec(name="Phase-B(Target)", steps=STEPS_TARGET_PHASE, env_spec=target_env_spec),
        ]
        # Two eval contexts: Target (primary), Source (aux)
        eval_specs_map[label] = [
            EvalSpec(name="Target", env_spec=target_env_spec, eval_seed_base=10_000),
            EvalSpec(name="Source-A", env_spec=source_env_spec, eval_seed_base=20_000),
        ]

    # ---------- W&B init (if not provided) ----------
    run = _maybe_init_wandb()

    # ---------- Run trainer ----------
    trainer = GenericCurriculumTrainer(
        agent_ctor_path=AGENT_CTOR_PATH,
        agent_kwargs=AGENT_KW,
        eval_every=EVAL_EVERY,
        n_eval_episodes=N_EVAL_EPISODES,
        output_dir=OUTPUT_DIR,
        wandb_run=run,
        max_workers=None,
    )

    aggregated = trainer.run(
        seeds=SEEDS,
        baseline_phase_specs=baseline_phase_specs,
        item_phase_specs_map=item_phase_specs_map,
        eval_specs_map=eval_specs_map,
    )

    # Print summary at final checkpoint
    ckpts = aggregated["checkpoints"]
    print("\n=== Summary at final checkpoint (per item) ===")
    for label, item in sorted(aggregated["items"].items()):
        b_g = aggregated["baseline"]["greedy_mean"][-1]
        b_gs = aggregated["baseline"]["greedy_std"][-1]
        b_t = aggregated["baseline"]["train_mean"][-1]
        b_ts = aggregated["baseline"]["train_std"][-1]

        tgt = item["Target"]
        print(f"[{label}]")
        print(f"  Target-only baseline — Greedy: {b_g:.3f} ± {b_gs:.3f} | TrainPol: {b_t:.3f} ± {b_ts:.3f}")
        print(f"  Curriculum → Target  — Greedy: {tgt['greedy_mean'][-1]:.3f} ± {tgt['greedy_std'][-1]:.3f} | "
              f"TrainPol: {tgt['train_mean'][-1]:.3f} ± {tgt['train_std'][-1]:.3f}")

        src_key = "Source-A"
        if src_key in item:
            src = item[src_key]
            print(f"  Curriculum (Source) — Greedy: {src['greedy_mean'][-1]:.3f} ± {src['greedy_std'][-1]:.3f} | "
                  f"TrainPol: {src['train_mean'][-1]:.3f} ± {src['train_std'][-1]:.3f}")

    # Log outputs as a W&B artifact for reproducibility
    _log_outputs_as_artifact(run, OUTPUT_DIR, aggregated.get("boundaries", []))

    print("\nDone.")

    # Finish wandb run if we created it here
    if run is not None and run is not WANDB_RUN:
        try:
            run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
