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

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from simple_agents.apis import FunctionCallback
from two_stage_cl.utils import plot_pairwise

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


def _dedup_by_equal_steps(
    checkpoints: np.ndarray,
    curves: Dict[str, Dict[str, np.ndarray]]
) -> tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """
    Remove consecutive duplicates in `checkpoints` (same timestep twice),
    and drop the same indices from all curves to keep them aligned.
    Keeps the FIRST of each duplicate run, drops the later one.
    """
    if checkpoints.size <= 1:
        return checkpoints, curves

    keep = [0]
    for i in range(1, len(checkpoints)):
        if checkpoints[i] != checkpoints[i - 1]:
            keep.append(i)
    keep_idx = np.asarray(keep, dtype=int)

    # new x
    new_ckpt = checkpoints[keep_idx]

    # new y for every eval name
    for name, dd in curves.items():
        dd["greedy"] = dd["greedy"][keep_idx]
        dd["train"]  = dd["train"][keep_idx]

    return new_ckpt, curves


def _rollout_and_save_media(
    make_env_fn: Callable[[], Any],
    model: Any,
    out_path: str,
    episodes: int = 3,
    max_steps: int = 200,
    fps: int = 8,
    fmt: str = "gif",
    deterministic: bool = True,
) -> Optional[str]:
    """
    Rollout with a trained model while capturing rgb frames, then save as GIF/MP4.
    - make_env_fn(): returns a fresh single env; we will set env.render_mode='rgb_array' if available
    - model: SB3-like model with .predict(obs, deterministic=...) -> (action, state)
    Returns the saved file path or None if failed/no frames.
    """
    try:
        import imageio.v2 as imageio  # lazy import
    except Exception as e:
        print(f"[media] imageio not available: {e}")
        return None

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    env = make_env_fn()
    # force rgb_array if the env supports this knob
    if hasattr(env, "render_mode"):
        try:
            env.render_mode = "rgb_array"
        except Exception:
            pass

    frames: list = []
    try:
        for ep in range(episodes):
            obs, _info = env.reset(seed=12345 + ep)
            for t in range(max_steps):
                # grab frame (if renderer returns something)
                try:
                    fr = env.render()
                    if fr is not None:
                        frames.append(fr)
                except Exception:
                    pass

                action, _state = model.predict(obs, deterministic=deterministic)
                obs, r, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    # final frame
                    try:
                        fr = env.render()
                        if fr is not None:
                            frames.append(fr)
                    except Exception:
                        pass
                    break

        if not frames:
            print("[media] No frames captured; skip saving.")
            return None

        fmt = fmt.lower()
        if fmt == "gif":
            # duration: seconds per frame
            imageio.mimsave(out_path, frames, duration=max(1e-6, 1.0 / float(fps)))
        else:
            # mp4 or other: use writer with fps
            writer = imageio.get_writer(out_path, fps=int(fps))
            for fr in frames:
                writer.append_data(fr)
            writer.close()
        print(f"[media] Saved media to {out_path}")
        return out_path
    except Exception as e:
        print(f"[media] Failed to save media: {e}")
        return None
    finally:
        try:
            env.close()
        except Exception:
            pass


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
    media_cfg: Optional[Dict[str, Any]] = None,
    media_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one seed across:
      - Baseline plan (all phases on Target)
      - Per-item plans (e.g., [Source -> Target]) with multiple eval contexts
    Adds "boundary tests": right AFTER specified phases finish (e.g., after Phase-0),
    we evaluate ON SELECTED EVAL ENVS (typically Target/baseline), and optionally record media.

    Returns only pickle-safe data (numpy arrays, dicts, lists, numbers).
    """
    results: Dict[str, Any] = {"seed": int(seed)}

    # -----------------------------
    # Helper to run one multi-phase plan
    # -----------------------------
    def _run_plan(
        phase_specs: List[Dict[str, Any]],
        eval_specs: List[Dict[str, Any]],
        label_for_media: str
    ) -> Dict[str, Any]:
        """
        Train across multiple phases without resetting num_timesteps.
        Keeps periodic/start/end evaluations via PeriodicEvalCallback,
        and additionally performs "boundary tests" right AFTER selected phases end.
        """
        # Build the agent on the first phase env; later phases call set_env(...)
        first_env = DummyVecEnv([lambda: _make_env_from_spec(phase_specs[0]["env_spec"], seed=seed)])
        agent = _make_agent_from_ctor(agent_ctor_path, env=first_env, agent_kwargs=agent_kwargs, seed=seed)

        # Build evaluation environments + callback storage
        eval_envs: Dict[str, Any] = {}
        greedy_lists: Dict[str, List[float]] = {}
        train_lists: Dict[str, List[float]] = {}
        callbacks: Dict[str, PeriodicEvalCallback] = {}

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
                wandb_run=None,
                wandb_prefix="eval",
                eval_name=name,
            )

        # Fire "training start" callbacks once before Phase-0 training
        for cb in callbacks.values():
            cb.on_training_start(agent)

        # Parse boundary-test configuration
        boundary_cfg = (media_cfg or {}).get("boundary_tests", {}) or {}
        boundary_enabled: bool = bool(boundary_cfg.get("enabled", False))
        boundary_phase_indices = set(int(i) for i in boundary_cfg.get("phase_indices", [0]))  # default: after Phase-0
        boundary_eval_names = boundary_cfg.get("eval_names", ["Target"])  # default: only Target
        boundary_eval_names_set = None if boundary_eval_names is None else set(boundary_eval_names)
        record_boundary_media: bool = bool(boundary_cfg.get("record_media", False))

        # Boundary cache: out["boundary"][f"phase_{idx}"][eval_name] = {...}
        boundary_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Train through phases
        total_steps_list: List[int] = []
        for idx, ph in enumerate(phase_specs):
            steps = int(ph["steps"])
            total_steps_list.append(steps)

            # Train current phase with callbacks being ticked by the agent
            step_cb = FunctionCallback(lambda model: all(cb.on_step() for cb in callbacks.values()))
            agent.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=False, callback=step_cb)

            # -----------------------------
            # Boundary test right AFTER this phase finishes
            # -----------------------------
            if boundary_enabled and (idx in boundary_phase_indices):
                boundary_key = f"phase_{idx}"
                for es in eval_specs:
                    name = es["name"]
                    if (boundary_eval_names_set is not None) and (name not in boundary_eval_names_set):
                        continue

                    # Fresh eval env for this boundary (isolated RNG)
                    test_env = _make_env_from_spec(es["env_spec"], seed=seed + 777 + idx)
                    try:
                        # Greedy (deterministic) policy eval
                        test_env.reset(seed=seed + 777 + idx)
                        greedy_mean, _ = evaluate_policy(
                            model=agent, env=test_env,
                            n_eval_episodes=n_eval_episodes,
                            deterministic=True, render=False, warn=False
                        )
                        # Stochastic (training-policy) eval
                        test_env.reset(seed=seed + 888 + idx)
                        train_mean, _ = evaluate_policy(
                            model=agent, env=test_env,
                            n_eval_episodes=n_eval_episodes,
                            deterministic=False, render=False, warn=False
                        )
                    finally:
                        try:
                            test_env.close()
                        except Exception:
                            pass

                    # Save numbers
                    boundary_cache.setdefault(boundary_key, {})
                    boundary_cache[boundary_key].setdefault(name, {})
                    boundary_cache[boundary_key][name]["greedy"] = float(greedy_mean)
                    boundary_cache[boundary_key][name]["train"] = float(train_mean)

                    # Optional media at the boundary
                    if record_boundary_media and media_cfg and media_dir:
                        def _make_media_env():
                            env = _make_env_from_spec(es["env_spec"], seed=seed + 42 + idx)
                            if hasattr(env, "render_mode"):
                                try:
                                    env.render_mode = "rgb_array"
                                except Exception:
                                    pass
                            return env

                        subdir = os.path.join(str(media_dir), f"seed_{seed}", label_for_media)
                        os.makedirs(subdir, exist_ok=True)
                        fmt = str(media_cfg.get("format", "gif")).lower()
                        out_path = os.path.join(subdir, f"{name}_phase{idx}.{fmt if fmt in ('gif','mp4') else 'gif'}")
                        try:
                            saved = _rollout_and_save_media(
                                make_env_fn=_make_media_env,
                                model=agent,
                                out_path=out_path,
                                episodes=int(media_cfg.get("episodes", 3)),
                                max_steps=int(media_cfg.get("max_steps", 200)),
                                fps=int(media_cfg.get("fps", 8)),
                                fmt=fmt,
                                deterministic=bool(media_cfg.get("deterministic", True)),
                            )
                        except Exception as e:
                            print(f"[media] boundary recording failed for '{label_for_media}/{name}@phase{idx}': {e}")
                            saved = None
                        boundary_cache[boundary_key][name]["media_path"] = saved

            # -----------------------------
            # Switch to the next phase's env (if any) and trigger "start" evals
            # -----------------------------
            if idx < len(phase_specs) - 1:
                next_env = DummyVecEnv([lambda: _make_env_from_spec(phase_specs[idx + 1]["env_spec"], seed=seed)])
                agent.set_env(next_env)
                for cb in callbacks.values():
                    cb.on_training_start(agent)

        # Finalize (end-of-training evals)
        for cb in callbacks.values():
            cb.on_training_end()

        # Build checkpoints array (same for all eval contexts)
        n_points = len(next(iter(greedy_lists.values())))
        checkpoints = build_checkpoints_from_curve_len_multiphase(total_steps_list, n_points)

        # Assemble curves
        curves = {
            name: {
                "greedy": np.asarray(greedy_lists[name], dtype=float),
                "train": np.asarray(train_lists[name], dtype=float),
            }
            for name in greedy_lists
        }

        # Drop boundary duplicates (same timestep back-to-back)
        checkpoints, curves = _dedup_by_equal_steps(checkpoints, curves)

        # Bundle outputs
        out: Dict[str, Any] = {"checkpoints": checkpoints}
        out.update(curves)

        # Regular media from post-training rollout
        out["media"] = {}
        if media_cfg and media_cfg.get("enabled", False) and media_dir:
            episodes = int(media_cfg.get("episodes", 3))
            max_steps = int(media_cfg.get("max_steps", 200))
            fps = int(media_cfg.get("fps", 8))
            fmt = str(media_cfg.get("format", "gif")).lower()
            deterministic = bool(media_cfg.get("deterministic", True))
            only_names = media_cfg.get("eval_names")  # None -> all eval contexts

            for es in eval_specs:
                name = es["name"]
                if (only_names is not None) and (name not in set(only_names)):
                    continue

                def _make_media_env():
                    env = _make_env_from_spec(es["env_spec"], seed=seed + 42)
                    if hasattr(env, "render_mode"):
                        try:
                            env.render_mode = "rgb_array"
                        except Exception:
                            pass
                    return env

                subdir = os.path.join(str(media_dir), f"seed_{seed}", label_for_media)
                os.makedirs(subdir, exist_ok=True)
                out_path = os.path.join(subdir, f"{name}.{fmt if fmt in ('gif', 'mp4') else 'gif'}")

                try:
                    saved = _rollout_and_save_media(
                        make_env_fn=_make_media_env,
                        model=agent,
                        out_path=out_path,
                        episodes=episodes,
                        max_steps=max_steps,
                        fps=fps,
                        fmt=fmt,
                        deterministic=deterministic,
                    )
                except Exception as e:
                    print(f"[media] recording failed for '{label_for_media}/{name}': {e}")
                    saved = None
                out["media"][name] = saved

        # Attach boundary results (if any)
        out["boundary"] = boundary_cache
        return out

    # --- Baseline (Target-only) ---
    results["baseline"] = _run_plan(
        baseline_phase_specs,
        eval_specs=[{"name": "Target", "env_spec": baseline_phase_specs[-1]["env_spec"]}],
        label_for_media="baseline",
    )

    # --- Per-item plans ---
    items_out: Dict[str, Any] = {}
    for label, phase_specs in item_phase_specs_map.items():
        evs = eval_specs_map[label]
        items_out[label] = _run_plan(phase_specs, evs, label_for_media=str(label))
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


# =========================
# Public Trainer API
# =========================
class TabularCurriculumTrainer:
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
        media_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.agent_ctor_path = agent_ctor_path
        self.agent_kwargs = dict(agent_kwargs)
        self.eval_every = int(eval_every)
        self.n_eval_episodes = int(n_eval_episodes)
        self.output_dir = str(output_dir)
        self.wandb_run = wandb_run if _WANDB_AVAILABLE else None
        self.max_workers = max_workers

        self.media_cfg = {
            "enabled": False,  # turn on to record
            "episodes": 3,
            "max_steps": 200,
            "fps": 8,
            "format": "gif",  # "gif" or "mp4"
            "deterministic": True,
            "eval_names": ["Target"],  # None for all eval contexts
            "log_first_seed_only": True,  # only log seed_?=first result to W&B
        }
        if media_cfg:
            self.media_cfg.update(media_cfg)

        _ensure_dir(self.output_dir)

    def run(
            self,
            seeds: List[int],
            baseline_phase_specs: List[PhaseSpec],
            item_phase_specs_map: Dict[str, List[PhaseSpec]],
            eval_specs_map: Dict[str, List[EvalSpec]],
    ) -> Dict[str, Any]:
        """
        Execute parallel training across seeds and aggregate results.
        Also uploads both end-of-training media and boundary-test media to W&B (if enabled).
        """
        # Serialize specs to dicts (pickle-safe)
        baseline_phase_specs_d = [ph.as_dict() for ph in baseline_phase_specs]
        item_phase_specs_map_d = {k: [ph.as_dict() for ph in v] for k, v in item_phase_specs_map.items()}
        eval_specs_map_d = {k: [es.as_dict() for es in v] for k, v in eval_specs_map.items()}

        num_workers = self.max_workers or min(len(seeds), os.cpu_count() or 1)
        all_results: List[Dict[str, Any]] = []

        media_root = os.path.join(self.output_dir, "media")
        _ensure_dir(media_root)
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
                    self.media_cfg,
                    media_root,
                ): seed
                for seed in seeds
            }
            for fut in as_completed(futs):
                res = fut.result()
                all_results.append(res)
                print(f"[Seed {res['seed']}] finished.")

        # ---------- Aggregate curves ----------
        labels = sorted(item_phase_specs_map.keys())

        # Use first seed checkpoints as reference; assert match
        ref_checkpoints = all_results[0]["baseline"]["checkpoints"]
        for r in all_results:
            assert np.array_equal(r["baseline"]["checkpoints"], ref_checkpoints), "Baseline checkpoints mismatch."
            for lb in labels:
                assert np.array_equal(r["items"][lb]["checkpoints"], all_results[0]["items"][lb]["checkpoints"]), \
                    f"Item checkpoints mismatch for '{lb}'."

        # Compute phase boundaries for baseline (for plot/filenames)
        baseline_boundaries = []
        s = 0
        for ph in baseline_phase_specs[:-1]:
            s += int(ph.steps)
            baseline_boundaries.append(s)
        boundaries_str = "-".join(str(b) for b in baseline_boundaries) if baseline_boundaries else "none"

        # Baseline aggregate
        base_greedy_all, base_train_all = [], []
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

        # Per item aggregate + save CSV + plot + (optional) W&B images/tables
        for lb in labels:
            eval_names = [es.name for es in eval_specs_map[lb]]
            eval_curves: Dict[str, Dict[str, List[np.ndarray]]] = {nm: {"greedy": [], "train": []} for nm in eval_names}
            for r in all_results:
                item = r["items"][lb]
                for nm in eval_names:
                    eval_curves[nm]["greedy"].append(item[nm]["greedy"])
                    eval_curves[nm]["train"].append(item[nm]["train"])

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

            _save_csv(os.path.join(item_out_dir, f"baseline_target_phase_{boundaries_str}.csv"),
                      ref_checkpoints, base_greedy_mean, base_greedy_std, header="greedy")
            _save_csv(os.path.join(item_out_dir, f"baseline_target_train_phase_{boundaries_str}.csv"),
                      ref_checkpoints, base_train_mean, base_train_std, header="train")

            if "Target" in agg_item:
                _save_csv(os.path.join(item_out_dir, f"curriculum_eval_target_phase_{boundaries_str}.csv"),
                          ref_checkpoints, agg_item["Target"]["greedy_mean"], agg_item["Target"]["greedy_std"],
                          header="greedy")
                _save_csv(os.path.join(item_out_dir, f"curriculum_eval_target_train_phase_{boundaries_str}.csv"),
                          ref_checkpoints, agg_item["Target"]["train_mean"], agg_item["Target"]["train_std"],
                          header="train")

            src_key = next((nm for nm in eval_names if nm.lower().startswith("source")), None)
            if src_key is not None:
                _save_csv(os.path.join(item_out_dir, f"curriculum_eval_source_phase_{boundaries_str}.csv"),
                          ref_checkpoints, agg_item[src_key]["greedy_mean"], agg_item[src_key]["greedy_std"],
                          header="greedy")
                _save_csv(os.path.join(item_out_dir, f"curriculum_eval_source_train_phase_{boundaries_str}.csv"),
                          ref_checkpoints, agg_item[src_key]["train_mean"], agg_item[src_key]["train_std"],
                          header="train")

            # Plot pairwise
            png_path = os.path.join(item_out_dir, f"pairwise_{lb}_phase_{boundaries_str}.png")
            plot_pairwise(
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
                    self.wandb_run.log({f"images/pairwise_{lb}_phase_{boundaries_str}": wandb.Image(png_path)})

                    # Log curves as a wandb.Table
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

        # --- W&B: upload media (end-of-training + boundary), if any ---
        if self.wandb_run is not None and self.media_cfg.get("enabled", False):
            to_log = []

            def _enqueue_media(seed_id: int, label: str, media_map: Dict[str, Optional[str]]):
                # media_map: {"Target": ".../Target.gif", ...}
                if not isinstance(media_map, dict):
                    return
                for name, path in media_map.items():
                    if path:
                        # e.g. media/seed_0/<label>/Target
                        to_log.append((f"media/seed_{seed_id}/{label}/{name}", path))

            def _enqueue_boundary(seed_id: int, label: str, boundary_map: Dict[str, Any]):
                # boundary_map: {"phase_0": {"Target": {"media_path": ".../Target_phase0.gif", ...}}, ...}
                if not isinstance(boundary_map, dict):
                    return
                for phase_key, per_eval in boundary_map.items():
                    if not isinstance(per_eval, dict):
                        continue
                    for eval_name, rec in per_eval.items():
                        path = (rec or {}).get("media_path", None)
                        if path:
                            # e.g. media/seed_0/<label>/Target_phase_0
                            to_log.append((f"media/seed_{seed_id}/{label}/{eval_name}_{phase_key}", path))

            for i, r in enumerate(all_results):
                if self.media_cfg.get("log_first_seed_only", True) and i > 0:
                    break
                sd = int(r.get("seed", i))

                # baseline: end-of-training media + boundary media
                base = r.get("baseline", {}) or {}
                _enqueue_media(sd, "baseline", base.get("media", {}) or {})
                _enqueue_boundary(sd, "baseline", base.get("boundary", {}) or {})

                # items: end-of-training media + boundary media
                items_map = r.get("items", {}) or {}
                for lb, d in items_map.items():
                    d = d or {}
                    _enqueue_media(sd, str(lb), d.get("media", {}) or {})
                    _enqueue_boundary(sd, str(lb), d.get("boundary", {}) or {})

            print(f"[W&B] prepared {len(to_log)} media files to upload.")
            for key, path in to_log:
                try:
                    if str(path).lower().endswith(".gif"):
                        self.wandb_run.log(
                            {key: wandb.Video(path, fps=int(self.media_cfg.get("fps", 8)), format="gif")}
                        )
                    else:
                        self.wandb_run.log(
                            {key: wandb.Video(path, fps=int(self.media_cfg.get("fps", 8)))}
                        )
                    print(f"[W&B] uploaded: {key}  -->  {path}")
                except Exception as e:
                    print(f"[W&B] video log failed for {key}: {e}")

        # Save meta
        with open(os.path.join(self.output_dir, "meta.json"), "w") as f:
            json.dump({
                "checkpoints": aggregated["checkpoints"],
                "boundaries": aggregated["boundaries"],
                "labels": labels,
            }, f, indent=2)

        return aggregated
