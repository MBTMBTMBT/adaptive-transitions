# generic_curriculum_trainer.py
# English comments only.

from __future__ import annotations
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from simple_agents.apis import FunctionCallback

# Optional: W&B is only used in the parent process if provided.
try:
    import wandb  # noqa: F401
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False


# =========================
# Callback
# =========================
class PeriodicEvalCallback:
    """
    Minimal SB3-like callback that evaluates on step 0, every eval_every, and at training end.
    It assumes model has attributes/methods like SB3 (num_timesteps, predict, set_env).
    Optionally logs scalars to Weights & Biases if a run is provided.
    """
    def __init__(self, eval_env, eval_every: int, n_eval_episodes: int,
                 greedy_scores_list: List[float], train_scores_list: List[float],
                 eval_seed_base: int = 10000, verbose: int = 0,
                 wandb_run: Optional[Any] = None,
                 wandb_prefix: str = "eval",
                 eval_name: str = "default"):
        self.model = None
        self.eval_env = eval_env
        self.eval_every = int(eval_every)
        self.n_eval_episodes = int(n_eval_episodes)
        self.greedy_scores_list = greedy_scores_list
        self.train_scores_list = train_scores_list
        self.eval_seed_base = int(eval_seed_base)
        self._last_eval_step = -1
        self._eval_count = 0
        self.verbose = int(verbose)

        # Optional W&B logging. Safe no-op if _WANDB_AVAILABLE is False.
        self.wandb_run = (wandb_run if (_WANDB_AVAILABLE and wandb_run is not None) else None)
        self.wandb_prefix = str(wandb_prefix)
        self.eval_name = str(eval_name)

    def _wandb_log_safe(self, step: int, greedy: float, trainpol: float, tag: str):
        """
        Log scalars to W&B if a run is available. Uses the model's global step as W&B step.
        Keys are hierarchical: <prefix>/<eval_name>/<metric>.
        """
        if self.wandb_run is None:
            return
        try:
            key_g = f"{self.wandb_prefix}/{self.eval_name}/greedy"
            key_t = f"{self.wandb_prefix}/{self.eval_name}/train"
            # Optionally include tag info (e.g., start/periodic/end) as another key if desired.
            self.wandb_run.log({key_g: float(greedy), key_t: float(trainpol)}, step=int(step))
        except Exception as e:
            # Do not break training due to logging failures.
            if self.verbose:
                print(f"[W&B] scalar log failed ({self.eval_name}, {tag}): {e}")

    def _do_eval(self, tag: str):
        # Control RNG for reproducibility
        self.eval_env.reset(seed=self.eval_seed_base + 2*self._eval_count)
        mean_greedy, _ = evaluate_policy(
            model=self.model, env=self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True, render=False, warn=False
        )
        self.eval_env.reset(seed=self.eval_seed_base + 2*self._eval_count + 1)
        mean_train, _ = evaluate_policy(
            model=self.model, env=self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=False, render=False, warn=False
        )
        self.greedy_scores_list.append(float(mean_greedy))
        self.train_scores_list.append(float(mean_train))

        # WandB scalar logging at the model's global step
        self._wandb_log_safe(step=self.model.num_timesteps,
                             greedy=mean_greedy, trainpol=mean_train, tag=tag)

        self._last_eval_step = self.model.num_timesteps
        self._eval_count += 1
        if self.verbose:
            print(f"[Eval:{tag}] step={self.model.num_timesteps}  "
                  f"Greedy={mean_greedy:.3f}  TrainPol={mean_train:.3f}")

    def on_training_start(self, model):
        self.model = model
        self._do_eval(tag="start")

    def on_step(self):
        if self.model.num_timesteps > 0 and self.model.num_timesteps % self.eval_every == 0:
            if self._last_eval_step != self.model.num_timesteps:
                self._do_eval(tag="periodic")
        return True

    def on_training_end(self):
        if self._last_eval_step != self.model.num_timesteps:
            self._do_eval(tag="end")


# =========================
# Spec dataclasses (serialized as dicts)
# =========================
@dataclass
class EnvFactorySpec:
    """
    Generic target env factory spec.
    factory_path: dotted path like "my_pkg.my_mod:make_env"
    kwargs: passed to factory when building env
    The factory signature must be: def factory(seed: int, **kwargs) -> gym.Env
    """
    factory_path: str
    kwargs: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"type": "target", "factory_path": self.factory_path, "kwargs": self.kwargs}


@dataclass
class SourceFactorySpec:
    """
    Source env factory spec that depends on an MDPNetwork loaded from config path.
    The factory signature must be: def factory(mdp, seed: int, **kwargs) -> gym.Env
    """
    factory_path: str
    mdp_config_path: str
    kwargs: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": "source",
            "factory_path": self.factory_path,
            "mdp_config_path": self.mdp_config_path,
            "kwargs": self.kwargs,
        }


@dataclass
class PhaseSpec:
    """
    One training phase.
    name: display name
    steps: training steps for this phase
    env_spec: EnvFactorySpec or SourceFactorySpec (serialized dict)
    """
    name: str
    steps: int
    env_spec: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "steps": int(self.steps), "env_spec": self.env_spec}


@dataclass
class EvalSpec:
    """
    One evaluation context.
    name: display name (e.g., "Target", "Source-A")
    env_spec: EnvFactorySpec or SourceFactorySpec (serialized dict)
    eval_seed_base: base seed for deterministic eval reproducibility
    """
    name: str
    env_spec: Dict[str, Any]
    eval_seed_base: int = 10000

    def as_dict(self) -> Dict[str, Any]:
        d = {"name": self.name, "env_spec": self.env_spec, "eval_seed_base": int(self.eval_seed_base)}
        return d


# =========================
# Utilities
# =========================
def _import_from_path(path: str) -> Callable:
    """
    Import a callable from dotted path "module.sub:callable_name".
    """
    mod, fn = path.split(":")
    m = importlib.import_module(mod)
    return getattr(m, fn)


def _make_env_from_spec(spec: Dict[str, Any], seed: int):
    """
    Build env from a serialized spec. Supports target and source envs.
    """
    spec_type = spec["type"]
    factory = _import_from_path(spec["factory_path"])
    if spec_type == "target":
        return factory(seed=seed, **spec.get("kwargs", {}))
    elif spec_type == "source":
        # Lazy import to avoid heavy deps in parent
        from mdp_network.mdp_network import MDPNetwork  # noqa
        mdp = MDPNetwork(config_path=spec["mdp_config_path"])
        return factory(mdp=mdp, seed=seed, **spec.get("kwargs", {}))
    else:
        raise ValueError(f"Unknown env spec type: {spec_type}")


def _make_agent_from_ctor(agent_ctor_path: str, env, agent_kwargs: Dict[str, Any], seed: int):
    """
    Build agent from a callable dotted path, expecting SB3-like signature: (env=..., **kwargs).
    """
    ctor = _import_from_path(agent_ctor_path)
    return ctor(env=env, seed=seed, **agent_kwargs)


def build_checkpoints_from_curve_len_multiphase(steps_per_phase: List[int], n_points: int) -> np.ndarray:
    """
    Reconstruct x-axis (global timesteps) from callback output length for multi-phase training.
    Behavior mirrors: eval at start (step 0), periodic every eval_every (unknown here), and at each training end.
    We only know how many points came out (n_points). We distribute points across phases proportionally and
    inject boundary duplicates (end of phase i and immediate start of phase i+1 at the same global step).
    """
    assert n_points >= 2, "At least start and one more eval."
    n_boundaries = max(0, len(steps_per_phase) - 1)
    total_steps = int(sum(steps_per_phase))
    # Remaining points after subtracting start and the boundary duplicates
    k_total = n_points - 1 - n_boundaries
    k_total = max(0, k_total)

    # Proportional allocation per phase
    alloc = []
    cum = 0
    for s in steps_per_phase:
        alloc.append(s / total_steps if total_steps > 0 else 0.0)
        cum += s
    k_per_phase = [int(round(k_total * a)) for a in alloc]
    # Fix rounding drift
    drift = k_total - sum(k_per_phase)
    for i in range(abs(drift)):
        idx = i % len(k_per_phase)
        k_per_phase[idx] += 1 if drift > 0 else -1
    # Ensure non-negative
    k_per_phase = [max(0, x) for x in k_per_phase]

    xs: List[int] = [0]
    cum = 0
    for i, (s, k_i) in enumerate(zip(steps_per_phase, k_per_phase)):
        cum += int(s)
        if k_i > 0:
            # Uniformly spread k_i points; last hits phase end (cum)
            for j in range(1, k_i + 1):
                xs.append(int(round((cum - (cum - s)) * j / k_i)) + (cum - s))
        else:
            xs.append(cum)
        if i < len(steps_per_phase) - 1:
            # Boundary duplicate: append cum again
            xs.append(cum)

    xs = np.asarray(xs, dtype=int)
    # Fix length if off by 1 due to rounding corner cases
    if len(xs) > n_points:
        xs = xs[:n_points]
    elif len(xs) < n_points:
        pad = [total_steps] * (n_points - len(xs))
        xs = np.concatenate([xs, np.asarray(pad, dtype=int)], axis=0)
    return xs


# =========================
# Worker (runs in subprocess)
# =========================
def _run_one_seed_worker(
    seed: int,
    agent_ctor_path: str,
    agent_kwargs: Dict[str, Any],
    eval_every: int,
    n_eval_episodes: int,
    baseline_phase_specs: List[Dict[str, Any]],
    item_phase_specs_map: Dict[str, List[Dict[str, Any]]],
    eval_specs_map: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Run one seed across:
      - Baseline plan (phases on Target only)
      - For each item label: an item plan (e.g., [Source->Target]) plus multiple eval contexts
    Returns only pickle-safe data (numpy arrays, dicts, lists, numbers).
    """
    results: Dict[str, Any] = {"seed": int(seed)}

    # --- Build eval envs per label name ---
    def _make_eval_envs(eval_specs: List[Dict[str, Any]]):
        envs = {}
        for es in eval_specs:
            envs[es["name"]] = _make_env_from_spec(es["env_spec"], seed=12345)
        return envs

    # --- Helper to run one multi-phase plan with multiple eval contexts ---
    def _run_plan(phase_specs: List[Dict[str, Any]], eval_specs: List[Dict[str, Any]]):
        # Training agent constructed on first phase env; we will switch by set_env.
        first_env = DummyVecEnv([lambda: _make_env_from_spec(phase_specs[0]["env_spec"], seed=seed)])
        agent = _make_agent_from_ctor(agent_ctor_path, env=first_env, agent_kwargs=agent_kwargs, seed=seed)

        # Build evaluation targets and callbacks
        eval_envs = {}
        greedy_lists = {}
        train_lists = {}
        callbacks = {}
        for es in eval_specs:
            name = es["name"]
            eval_envs[name] = _make_env_from_spec(es["env_spec"], seed=12345)
            greedy_lists[name] = []
            train_lists[name] = []
            callbacks[name] = PeriodicEvalCallback(
                eval_env=eval_envs[name],
                eval_every=eval_every,
                n_eval_episodes=n_eval_episodes,
                greedy_scores_list=greedy_lists[name],
                train_scores_list=train_lists[name],
                eval_seed_base=es.get("eval_seed_base", 10000),
                verbose=0,
                wandb_run=None,  # for multi process, we don't want to log to the same run
                wandb_prefix="eval",
                eval_name=name
            )

        # Training start callbacks
        for cb in callbacks.values():
            cb.on_training_start(agent)

        # Train through phases sequentially without resetting num_timesteps
        total_steps_list: List[int] = []
        for idx, ph in enumerate(phase_specs):
            steps = int(ph["steps"])
            total_steps_list.append(steps)
            if idx > 0:
                # swap env to the new phase env
                next_env = DummyVecEnv([lambda: _make_env_from_spec(ph["env_spec"], seed=seed)])
                agent.set_env(next_env)
                # Duplicate eval at boundary: start of new training call
                for cb in callbacks.values():
                    cb.on_training_start(agent)

            # Standard SB3-like training loop imitation
            # We call agent.learn with a simple loop to be callback-compatible
            remaining = steps
            # If agent has a .learn(total_timesteps, reset_num_timesteps=False, callback=callable)
            # we can call it directly and step callback via its built-in mechanism.
            # To make it generic, we rely on agent.learn calling cb.on_step internally.
            step_cb = FunctionCallback(lambda model: all(cb.on_step() for cb in callbacks.values()))
            agent.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=False, callback=step_cb)

        # Finalize
        for cb in callbacks.values():
            cb.on_training_end()

        # Build checkpoint x-axis array from lengths (same for all eval contexts)
        n_points = len(next(iter(greedy_lists.values())))
        checkpoints = build_checkpoints_from_curve_len_multiphase(total_steps_list, n_points)

        # Bundle curves per eval name
        out = {"checkpoints": checkpoints}
        for name in greedy_lists:
            out[name] = {
                "greedy": np.asarray(greedy_lists[name], dtype=float),
                "train": np.asarray(train_lists[name], dtype=float),
            }
        return out

    # --- Baseline (single plan) ---
    baseline_out = _run_plan(baseline_phase_specs, eval_specs=[{"name": "Target", "env_spec": baseline_phase_specs[-1]["env_spec"]}])
    results["baseline"] = baseline_out

    # --- Items (per label) ---
    items_out: Dict[str, Any] = {}
    for label, phase_specs in item_phase_specs_map.items():
        evs = eval_specs_map[label]  # List of EvalSpec dicts
        items_out[label] = _run_plan(phase_specs, evs)
    results["items"] = items_out
    return results


# =========================
# Aggregation + Plot + Optional W&B
# =========================
def _mean_std(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.stack(curves, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def _save_csv(path: str, steps: np.ndarray, mean: np.ndarray, std: np.ndarray, header: str):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([header])
        w.writerow(["step", "mean", "std"])
        for s, m, sd in zip(steps.tolist(), mean.tolist(), std.tolist()):
            w.writerow([int(s), float(m), float(sd)])


def _plot_pairwise(
    out_png_path: str,
    checkpoints: np.ndarray,
    phase_boundaries: List[int],
    title_prefix: str,
    baseline: Dict[str, np.ndarray],
    curves_target: Dict[str, np.ndarray],
    curves_source: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Plot greedy and train-policy curves (mean±std) with vertical phase boundary markers.
    baseline: {"greedy_mean","greedy_std","train_mean","train_std"}
    curves_target / curves_source: same keys as baseline
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Titles
    ax1.set_title(f"{title_prefix} (Greedy)")
    ax2.set_title(f"{title_prefix} (Training-policy)")

    # Greedy
    ax1.plot(checkpoints, baseline["greedy_mean"], label="Target-only baseline", linewidth=1.8)
    ax1.fill_between(checkpoints, baseline["greedy_mean"] - baseline["greedy_std"],
                     baseline["greedy_mean"] + baseline["greedy_std"], alpha=0.2)
    ax1.plot(checkpoints, curves_target["greedy_mean"], label="Curriculum → Target (primary)", linewidth=2.2)
    ax1.fill_between(checkpoints, curves_target["greedy_mean"] - curves_target["greedy_std"],
                     curves_target["greedy_mean"] + curves_target["greedy_std"], alpha=0.15)
    if curves_source is not None:
        ax1.plot(checkpoints, curves_source["greedy_mean"], label="Curriculum (eval on Source)", linewidth=1.6)
        ax1.fill_between(checkpoints, curves_source["greedy_mean"] - curves_source["greedy_std"],
                         curves_source["greedy_mean"] + curves_source["greedy_std"], alpha=0.15)
    ax1.set_xlabel("Timesteps"); ax1.set_ylabel("Mean return"); ax1.grid(True, alpha=0.3); ax1.legend()

    # Train-policy
    ax2.plot(checkpoints, baseline["train_mean"], label="Target-only baseline", linewidth=1.8)
    ax2.fill_between(checkpoints, baseline["train_mean"] - baseline["train_std"],
                     baseline["train_mean"] + baseline["train_std"], alpha=0.2)
    ax2.plot(checkpoints, curves_target["train_mean"], label="Curriculum → Target (primary)", linewidth=2.2)
    ax2.fill_between(checkpoints, curves_target["train_mean"] - curves_target["train_std"],
                     curves_target["train_mean"] + curves_target["train_std"], alpha=0.15)
    if curves_source is not None:
        ax2.plot(checkpoints, curves_source["train_mean"], label="Curriculum (eval on Source)", linewidth=1.6)
        ax2.fill_between(checkpoints, curves_source["train_mean"] - curves_source["train_std"],
                         curves_source["train_mean"] + curves_source["train_std"], alpha=0.15)
    ax2.set_xlabel("Timesteps"); ax2.set_ylabel("Mean return"); ax2.grid(True, alpha=0.3); ax2.legend()

    # Phase boundary markers
    for ax in (ax1, ax2):
        for b in phase_boundaries:
            ax.axvline(b, linestyle="--", alpha=0.7)
        ymin, ymax = ax.get_ylim()
        ytxt = ymin + 0.06 * (ymax - ymin)
        # Optional labels: show first-half vs second-half regions
        if len(phase_boundaries) >= 1:
            left_mid = phase_boundaries[0] * 0.5
            right_mid = phase_boundaries[-1] + (checkpoints[-1] - phase_boundaries[-1]) * 0.5
            ax.text(left_mid, ytxt, "Phase 1 (Source)", ha="center", va="bottom", fontsize=9, alpha=0.8)
            ax.text(right_mid, ytxt, "Later Phases (Target/others)", ha="center", va="bottom", fontsize=9, alpha=0.8)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)


# =========================
# Public Trainer API
# =========================
class GenericCurriculumTrainer:
    """
    Process-parallel generic curriculum trainer.
    - Env construction decoupled via factory specs.
    - Multi-phase schedules with arbitrary boundaries.
    - Aggregation across seeds; W&B optional logging at the parent process.
    """

    def __init__(
        self,
        agent_ctor_path: str,
        agent_kwargs: Dict[str, Any],
        eval_every: int,
        n_eval_episodes: int,
        output_dir: str,
        wandb_run: Optional[Any] = None,
        max_workers: Optional[int] = None,
    ):
        self.agent_ctor_path = agent_ctor_path
        self.agent_kwargs = dict(agent_kwargs)
        self.eval_every = int(eval_every)
        self.n_eval_episodes = int(n_eval_episodes)
        self.output_dir = str(output_dir)
        self.wandb_run = wandb_run if _WANDB_AVAILABLE else None
        self.max_workers = max_workers

        _ensure_dir(self.output_dir)

    def run(
        self,
        seeds: List[int],
        baseline_phase_specs: List[PhaseSpec],
        item_phase_specs_map: Dict[str, List[PhaseSpec]],
        eval_specs_map: Dict[str, List[EvalSpec]],
    ) -> Dict[str, Any]:
        """
        Execute parallel training across seeds.
        - baseline_phase_specs: phases for target-only baseline
        - item_phase_specs_map: per label phases (e.g., [Source->Target])
        - eval_specs_map: per label evaluation contexts (e.g., {"Target","Source"})
        Returns aggregated results with means/stds and file paths.
        """
        # Serialize specs to dicts (pickle-safe)
        baseline_phase_specs_d = [ph.as_dict() for ph in baseline_phase_specs]
        item_phase_specs_map_d = {k: [ph.as_dict() for ph in v] for k, v in item_phase_specs_map.items()}
        eval_specs_map_d = {k: [es.as_dict() for es in v] for k, v in eval_specs_map.items()}

        num_workers = self.max_workers or min(len(seeds), os.cpu_count() or 1)
        all_results: List[Dict[str, Any]] = []

        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context("spawn")) as ex:
            futs = {
                ex.submit(
                    _run_one_seed_worker,
                    seed,
                    self.agent_ctor_path,
                    self.agent_kwargs,
                    self.eval_every,
                    self.n_eval_episodes,
                    baseline_phase_specs_d,
                    item_phase_specs_map_d,
                    eval_specs_map_d,
                ): seed
                for seed in seeds
            }
            for fut in as_completed(futs):
                res = fut.result()
                all_results.append(res)
                print(f"[Seed {res['seed']}] finished.")

        # ---------- Aggregate ----------
        labels = sorted(item_phase_specs_map.keys())
        # Use first seed checkpoints as reference; assert match
        ref_checkpoints = all_results[0]["baseline"]["checkpoints"]
        for r in all_results:
            assert np.array_equal(r["baseline"]["checkpoints"], ref_checkpoints), "Baseline checkpoints mismatch."
            for lb in labels:
                assert np.array_equal(r["items"][lb]["checkpoints"], all_results[0]["items"][lb]["checkpoints"]), \
                    f"Item checkpoints mismatch for '{lb}'."

        # Compute phase boundaries (cumulative steps) for naming & markers
        baseline_boundaries = []
        s = 0
        for i, ph in enumerate(baseline_phase_specs[:-1]):
            s += int(ph.steps)
            baseline_boundaries.append(s)
        # Boundaries string for filenames
        boundaries_str = "-".join(str(b) for b in baseline_boundaries) if baseline_boundaries else "none"

        # Baseline aggregate
        base_greedy_all = []
        base_train_all = []
        for r in all_results:
            base = r["baseline"]["Target"]
            base_greedy_all.append(base["greedy"])
            base_train_all.append(base["train"])
        base_greedy_mean, base_greedy_std = _mean_std(base_greedy_all)
        base_train_mean, base_train_std = _mean_std(base_train_all)

        aggregated = {
            "checkpoints": ref_checkpoints.tolist(),
            "boundaries": baseline_boundaries,
            "baseline": {
                "greedy_mean": base_greedy_mean.tolist(),
                "greedy_std": base_greedy_std.tolist(),
                "train_mean": base_train_mean.tolist(),
                "train_std": base_train_std.tolist(),
            },
            "items": {}
        }

        # Per item aggregate and plotting/logging
        for lb in labels:
            # Collect per-eval curves
            eval_names = [es.name for es in eval_specs_map[lb]]
            eval_curves: Dict[str, Dict[str, List[np.ndarray]]] = {nm: {"greedy": [], "train": []} for nm in eval_names}
            for r in all_results:
                item = r["items"][lb]
                for nm in eval_names:
                    eval_curves[nm]["greedy"].append(item[nm]["greedy"])
                    eval_curves[nm]["train"].append(item[nm]["train"])

            # Mean/std
            agg_item: Dict[str, Dict[str, Any]] = {}
            for nm in eval_names:
                g_mean, g_std = _mean_std(eval_curves[nm]["greedy"])
                t_mean, t_std = _mean_std(eval_curves[nm]["train"])
                agg_item[nm] = {
                    "greedy_mean": g_mean, "greedy_std": g_std,
                    "train_mean": t_mean, "train_std": t_std
                }

            aggregated["items"][lb] = {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in agg_item.items()}

            # Save CSVs
            item_out_dir = os.path.join(self.output_dir, f"{lb}")
            _ensure_dir(item_out_dir)
            # Baseline CSV
            _save_csv(os.path.join(item_out_dir, f"baseline_target_phase_{boundaries_str}.csv"),
                      ref_checkpoints, base_greedy_mean, base_greedy_std, header="greedy")
            _save_csv(os.path.join(item_out_dir, f"baseline_target_train_phase_{boundaries_str}.csv"),
                      ref_checkpoints, base_train_mean, base_train_std, header="train")
            # Target eval CSV
            if "Target" in agg_item:
                _save_csv(os.path.join(item_out_dir, f"curriculum_eval_target_phase_{boundaries_str}.csv"),
                          ref_checkpoints, agg_item["Target"]["greedy_mean"], agg_item["Target"]["greedy_std"], header="greedy")
                _save_csv(os.path.join(item_out_dir, f"curriculum_eval_target_train_phase_{boundaries_str}.csv"),
                          ref_checkpoints, agg_item["Target"]["train_mean"], agg_item["Target"]["train_std"], header="train")
            # Source eval CSV (optional)
            src_key = next((nm for nm in eval_names if nm.lower().startswith("source")), None)
            if src_key is not None:
                _save_csv(os.path.join(item_out_dir, f"curriculum_eval_source_phase_{boundaries_str}.csv"),
                          ref_checkpoints, agg_item[src_key]["greedy_mean"], agg_item[src_key]["greedy_std"], header="greedy")
                _save_csv(os.path.join(item_out_dir, f"curriculum_eval_source_train_phase_{boundaries_str}.csv"),
                          ref_checkpoints, agg_item[src_key]["train_mean"], agg_item[src_key]["train_std"], header="train")

            # Plot and (optionally) log to W&B
            png_path = os.path.join(item_out_dir, f"pairwise_{lb}_phase_{boundaries_str}.png")
            _plot_pairwise(
                out_png_path=png_path,
                checkpoints=ref_checkpoints,
                phase_boundaries=baseline_boundaries,
                title_prefix=f"Pairwise for '{lb}'",
                baseline={
                    "greedy_mean": base_greedy_mean, "greedy_std": base_greedy_std,
                    "train_mean": base_train_mean, "train_std": base_train_std,
                },
                curves_target=agg_item.get("Target", None) or agg_item[eval_names[0]],
                curves_source=agg_item.get("Source", None) or agg_item.get("Source-A", None),
            )

            if self.wandb_run is not None:
                try:
                    # 1) Log the saved pairwise image as before
                    self.wandb_run.log({
                        f"images/pairwise_{lb}_phase_{boundaries_str}": wandb.Image(png_path)
                    })

                    # 2) Log curves as a proper wandb.Table (instead of a dict-of-lists)
                    # Build table schema (columns)
                    columns = [
                        "step",
                        "baseline_greedy_mean", "baseline_greedy_std",
                        "baseline_train_mean", "baseline_train_std",
                    ]
                    for nm in eval_names:
                        columns.extend([
                            f"{nm}_greedy_mean", f"{nm}_greedy_std",
                            f"{nm}_train_mean", f"{nm}_train_std",
                        ])

                    # Build table rows
                    data = []
                    L = len(ref_checkpoints)
                    for i in range(L):
                        row = [
                            int(ref_checkpoints[i]),
                            float(base_greedy_mean[i]), float(base_greedy_std[i]),
                            float(base_train_mean[i]), float(base_train_std[i]),
                        ]
                        for nm in eval_names:
                            row.extend([
                                float(agg_item[nm]["greedy_mean"][i]),
                                float(agg_item[nm]["greedy_std"][i]),
                                float(agg_item[nm]["train_mean"][i]),
                                float(agg_item[nm]["train_std"][i]),
                            ])
                        data.append(row)

                    table = wandb.Table(columns=columns, data=data)
                    self.wandb_run.log({f"tables/curves_{lb}_phase_{boundaries_str}": table})

                except Exception as e:
                    print(f"[W&B] logging failed for '{lb}': {e}")

        # Save meta
        with open(os.path.join(self.output_dir, "meta.json"), "w") as f:
            json.dump({
                "checkpoints": aggregated["checkpoints"],
                "boundaries": aggregated["boundaries"],
                "labels": labels,
            }, f, indent=2)

        return aggregated
