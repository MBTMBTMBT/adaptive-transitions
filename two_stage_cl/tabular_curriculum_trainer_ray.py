# generic_curriculum_trainer.py — Ray-based, plain dicts, local logging independent of W&B.
# English only.

from __future__ import annotations

import asyncio
import os, time, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from simple_agents.apis import FunctionCallback
from ray.actor import ActorHandle

from experiment_utils.utils import _import, ensure_dir
from experiment_utils.save_media import save_policy_media
from experiment_utils.env_factories import make_env
from two_stage_cl.metrics import (
    _mean_over,
    _ap_last_k,
    _ttt_frac,
    _interp_at,
    _ensure_curve,
    _auc_over,
    _jumpstart_fields,
)

from two_stage_cl.utils import plot_pairwise, save_csv


@ray.remote
class Collector:
    """
    Minimal & robust aggregator.

    - Baseline and CL are handled independently (separate buckets/state).
    - Aggregate by (series = label, eval_name, eval_idx) across seeds.
    - Each series binds its own x-axis to 'online/{label}/{eval}/series_step'.
    - Monotonic guard per series on eval_idx; drop if not strictly increasing.
    - 'series_step' = min(real_step) + step_base over seeds in the bucket (monotone enough).
    """

    def __init__(
        self,
        seeds: List[int],
        keep_intermediate: bool = True,
        wandb_actor: Optional[ActorHandle] = None,
        step_base: int = 0,
        flush_interval_s: float = 0.5,
        max_queue: int = 16384,
        verbose: int = 0,
    ):
        self.seeds = list(map(int, seeds))
        self.keep = bool(keep_intermediate)
        self._wb = wandb_actor
        self._step_base = int(step_base)

        # Optional raw timeline for debugging
        self._timeline: Dict[int, List[Dict[str, Any]]] = {s: [] for s in self.seeds}

        # Separate in-flight buckets for baseline vs curriculum(CL)
        # key: (label, eval_name, eval_idx) -> {seed: (greedy, train, step)}
        self._bucket_base: Dict[
            Tuple[str, str, int], Dict[int, Tuple[float, float, int]]
        ] = {}
        self._bucket_cl: Dict[
            Tuple[str, str, int], Dict[int, Tuple[float, float, int]]
        ] = {}

        # Per-series last eval_idx accepted (monotonic)
        self._last_idx_base: Dict[Tuple[str, str], int] = {}
        self._last_idx_cl: Dict[Tuple[str, str], int] = {}

        # Per-series define_metric guard
        self._defined_series: set[Tuple[str, str]] = set()

        # Async queue
        self._q: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=int(max_queue))
        self._flush_interval_s = float(flush_interval_s)

        self.verbose = int(verbose)

        # No global define_metric here; bind per series on first emit.

        # Start background consumer
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        loop.create_task(self._drain_loop())

    def _pick_tables(self, label: str):
        """Choose state tables for baseline vs CL by label string."""
        if label == "baseline":
            return self._bucket_base, self._last_idx_base
        else:
            return self._bucket_cl, self._last_idx_cl

    async def _drain_loop(self) -> None:
        """Consume queue; aggregate by (label, eval_name, eval_idx) across seeds; emit when complete."""
        last_yield = time.time()
        while True:
            e = await self._q.get()

            label = str(e["label"])
            eval_name = str(e["eval_name"])
            step = int(e["step"])
            eval_idx = int(e.get("eval_idx", -1))  # sent by PeriodicEvalCallback
            seed = int(e["seed"])
            g = float(e["greedy"])
            t = float(e["train"])

            buckets, last_idx = self._pick_tables(label)
            key = (label, eval_name, eval_idx)
            bucket = buckets.setdefault(key, {})
            bucket[seed] = (g, t, step)

            # When all seeds for this eval_idx arrived -> emit one row
            if self._wb is not None and len(bucket) == len(self.seeds):
                series_key = (label, eval_name)
                prev = last_idx.get(series_key, None)
                if prev is not None and eval_idx <= prev:
                    # Non-monotonic eval index for this series -> drop silently
                    if self.verbose > 0:
                        print(
                            f"[Collector] drop non-monotonic idx for online/{label}/{eval_name}: "
                            f"{eval_idx} <= {prev}"
                        )
                    del buckets[key]
                else:
                    # First time for this series: bind axis to series_step
                    if series_key not in self._defined_series:
                        try:
                            self._wb.define_metric.remote(
                                f"online/{label}/{eval_name}/*",
                                step_metric=f"online/{label}/{eval_name}/series_step",
                            )
                        except Exception:
                            pass
                        self._defined_series.add(series_key)

                    # Aggregate across seeds
                    g_arr = np.array([v[0] for v in bucket.values()], dtype=np.float64)
                    t_arr = np.array([v[1] for v in bucket.values()], dtype=np.float64)
                    min_step = int(min(v[2] for v in bucket.values()))
                    out_step = int(min_step + self._step_base)

                    series = f"online/{label}/{eval_name}"
                    payload = {
                        f"{series}/greedy_mean": float(g_arr.mean()),
                        f"{series}/greedy_std": float(g_arr.std()),
                        f"{series}/train_mean": float(t_arr.mean()),
                        f"{series}/train_std": float(t_arr.std()),
                        f"{series}/series_step": out_step,
                        f"{series}/series_idx": int(eval_idx),
                    }

                    del buckets[key]
                    try:
                        self._wb.log.remote(payload)  # no explicit step=
                        last_idx[series_key] = eval_idx
                    except Exception:
                        pass

            # Cooperative yield
            if (time.time() - last_yield) >= self._flush_interval_s:
                await asyncio.sleep(0)
                last_yield = time.time()

    async def report(self, e: Dict[str, Any]) -> None:
        """Non-blocking producer; drops on full queue."""
        if self.keep:
            self._timeline[int(e["seed"])].append(e)
        try:
            self._q.put_nowait(e)
        except asyncio.QueueFull:
            return

    def timeline(self) -> Dict[int, List[Dict[str, Any]]]:
        return self._timeline if self.keep else {}


class PeriodicEvalCallback:
    """
    Lightweight callback (not SB3 BaseCallback).
    - Evaluate at step 0, every `eval_every` steps, and at training end.
    - Keeps per-eval logs and optionally reports to a Collector actor.
    - Must be driven by FunctionCallback: call on_training_start(model) once per phase,
      then repeatedly call on_step() inside learn(), and finally on_training_end().
    """

    def __init__(
        self,
        *,
        eval_env,
        eval_every: int,
        n_eval_episodes: int,
        steps_log: List[int],
        greedy_log: List[float],
        train_log: List[float],
        seed_base: int,
        label: str,
        eval_name: str,
        collector: Optional[ActorHandle] = None,
        seed_id: Optional[int] = None,
    ):
        self.model = None
        self.eval_env = eval_env
        self.eval_every = int(eval_every)
        assert self.eval_every > 0, "eval_every must be > 0"
        self.n_eval = int(n_eval_episodes)
        self.steps_log = steps_log
        self.greedy_log = greedy_log
        self.train_log = train_log
        self.seed_base = int(seed_base)
        self.label = str(label)
        self.eval_name = str(eval_name)
        self.collector = collector
        self.seed_id = None if seed_id is None else int(seed_id)
        self._last_eval_step = -1
        self._eval_count = 0

    def _do_eval(self, tag: str):
        # greedy
        self.eval_env.reset(seed=self.seed_base + 2 * self._eval_count)
        g_mean, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval,
            deterministic=True,
            render=False,
            warn=False,
        )
        # train-policy
        self.eval_env.reset(seed=self.seed_base + 2 * self._eval_count + 1)
        t_mean, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval,
            deterministic=False,
            render=False,
            warn=False,
        )

        step = int(self.model.num_timesteps)
        self.steps_log.append(step)
        self.greedy_log.append(float(g_mean))
        self.train_log.append(float(t_mean))

        if self.collector is not None and self.seed_id is not None:
            try:
                self.collector.report.remote(
                    {
                        "seed": self.seed_id,
                        "label": self.label,
                        "eval_name": self.eval_name,
                        "eval_idx": int(self._eval_count),  # <<< add this line
                        "step": step,
                        "greedy": float(g_mean),
                        "train": float(t_mean),
                        "wall_time": time.time(),
                    }
                )
            except Exception as ex:
                print(f"[Collector] report failed: {ex}")

        self._last_eval_step = step
        self._eval_count += 1

    def on_training_start(self, model):
        self.model = model
        s = int(self.model.num_timesteps)
        # If previous phase ended at step s, don't double-log at the start of this phase.
        if self._last_eval_step != s:
            self._do_eval(tag="start")

    def on_step(self) -> bool:
        s = int(self.model.num_timesteps)
        if s > 0 and s % self.eval_every == 0 and self._last_eval_step != s:
            self._do_eval(tag="periodic")
        return True

    def on_training_end(self) -> None:
        s = int(self.model.num_timesteps)
        if self._last_eval_step != s:
            self._do_eval(tag="end")


@ray.remote
class SeedTrainer:
    def __init__(
        self,
        seed: int,
        envs: Dict[str, Dict[str, Any]],
        baseline_phases: List[Dict[str, Any]],
        baseline_evals: List[Dict[str, Any]],
        item_phases_map: Dict[str, List[Dict[str, Any]]],
        evals_map: Dict[str, List[Dict[str, Any]]],
        agent_ctor_path: str,
        agent_kwargs: Dict[str, Any],
        eval_every: int,
        n_eval_episodes: int,
        collector: Optional[Any] = None,
        collect_intermediate: bool = True,
        media_root: Optional[str] = None,
        media_opts: Optional[Dict[str, Any]] = None,
        mode: str = "both",  # "baseline" | "items" | "both" | "item"
        single_item_label: Optional[
            str
        ] = None,  # NEW: run only one item when mode=="item"
    ):
        # Keep original assignments
        self.seed = int(seed)
        self.envs = envs
        self.base_ph = baseline_phases
        self.base_evals = baseline_evals
        self.items_ph = item_phases_map
        self.evals_map = evals_map
        self.agent_ctor_path = agent_ctor_path
        self.agent_kwargs = dict(agent_kwargs)
        self.eval_every = int(eval_every)
        self.n_eval = int(n_eval_episodes)
        self.collector = collector
        self.collect_on = bool(collect_intermediate)
        self.mode = str(mode)
        self.single_item_label = single_item_label  # NEW

        # Media settings (unchanged)
        self.media_root = media_root
        self.media_on = media_root is not None
        mo = dict(media_opts or {})
        self.mo_episodes = int(mo.get("episodes", 3))
        self.mo_max_steps = int(mo.get("max_steps", 200))
        self.mo_fps = int(mo.get("fps", 8))
        self.mo_fmt = str(mo.get("fmt", "gif"))
        self.mo_deterministic = bool(mo.get("deterministic", True))
        self.mo_start_seed_base = int(mo.get("start_seed_base", 10_000))
        self.mo_target_size = mo.get("target_size", None)
        self.mo_bgr_to_rgb = bool(mo.get("bgr_to_rgb", False))

    # Resolve env spec from registry (supports nested envs["items"])
    def _env_spec(self, key: str) -> Dict[str, Any]:
        if key == "target" and "target" in self.envs:
            return self.envs["target"]
        if "items" in self.envs and key in self.envs["items"]:
            return self.envs["items"][key]
        raise KeyError(f"env key '{key}' not found in target/items")

    # --- eval helper ---
    def _eval_once(self, model, env, det: bool, seed_base: int) -> float:
        env.reset(seed=seed_base)
        m, _ = evaluate_policy(
            model, env, self.n_eval, deterministic=det, render=False, warn=False
        )
        return float(m)

    # --- media helpers ---
    def _save_media(
        self, label: str, eval_name: str, env_key: str, suffix: str, agent
    ) -> Optional[str]:
        if not self.media_on:
            return None
        env = make_env(
            self._env_spec(env_key), seed=self.seed + self.mo_start_seed_base
        )
        # subdir: <media_root>/seed_<id>/<label>/
        subdir = os.path.join(self.media_root, f"seed_{self.seed}", label)
        ensure_dir(Path(subdir))
        fmt = self.mo_fmt.lower()
        out_path = os.path.join(
            subdir, f"{eval_name}{suffix}.{fmt if fmt in ('gif','mp4') else 'gif'}"
        )
        try:
            saved = save_policy_media(
                model=agent,
                env=env,
                out_path=out_path,
                episodes=self.mo_episodes,
                start_seed=self.seed + self.mo_start_seed_base,
                max_steps=self.mo_max_steps,
                deterministic=self.mo_deterministic,
                fps=self.mo_fps,
                fmt=fmt,
                fix_render_mode=True,
                target_size=self.mo_target_size,
                bgr_to_rgb=self.mo_bgr_to_rgb,
                close_env=True,
            )
            return saved
        except Exception as e:
            print(f"[media] save failed for '{label}/{eval_name}{suffix}': {e}")
            return None

    def _boundary_media_all(
        self, label: str, eval_specs: List[Dict[str, Any]], agent, phase_idx: int
    ) -> Dict[str, Optional[str]]:
        """Record media for all eval envs at a phase boundary (if media_on)."""
        out: Dict[str, Optional[str]] = {}
        if not self.media_on:
            return out
        for es in eval_specs:
            nm = es["name"]
            p = self._save_media(
                label, nm, es["env"], suffix=f"_phase{phase_idx}", agent=agent
            )
            out[nm] = p
        return out

    def _final_media_all(
        self, label: str, eval_specs: List[Dict[str, Any]], agent
    ) -> Dict[str, Optional[str]]:
        """Record media for all eval envs at training end (if media_on)."""
        out: Dict[str, Optional[str]] = {}
        if not self.media_on:
            return out
        for es in eval_specs:
            nm = es["name"]
            p = self._save_media(label, nm, es["env"], suffix="", agent=agent)
            out[nm] = p
        return out

    # --- core schedule ---
    def _run_schedule(
        self, label: str, phases: List[Dict[str, Any]], evals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # Build training env & agent on phase-0 env
        train_env = DummyVecEnv(
            [lambda: make_env(self._env_spec(phases[0]["env"]), seed=self.seed)]
        )
        agent_cls = _import(self.agent_ctor_path)
        agent = agent_cls(env=train_env, seed=self.seed, **self.agent_kwargs)

        # Fixed eval envs + storages
        eval_envs: Dict[str, Any] = {
            es["name"]: make_env(self._env_spec(es["env"]), seed=12345) for es in evals
        }
        steps_log: Dict[str, List[int]] = {nm: [] for nm in eval_envs}
        greedy_log: Dict[str, List[float]] = {nm: [] for nm in eval_envs}
        trainp_log: Dict[str, List[float]] = {nm: [] for nm in eval_envs}

        # One callback per eval env
        per_eval_cbs: List[PeriodicEvalCallback] = []
        for es in evals:
            nm = es["name"]
            per_eval_cbs.append(
                PeriodicEvalCallback(
                    eval_env=eval_envs[nm],
                    eval_every=self.eval_every,
                    n_eval_episodes=self.n_eval,
                    steps_log=steps_log[nm],
                    greedy_log=greedy_log[nm],
                    train_log=trainp_log[nm],
                    seed_base=int(es.get("seed_base", self.mo_start_seed_base)),
                    label=label,
                    eval_name=nm,
                    collector=self.collector if self.collect_on else None,
                    seed_id=self.seed,
                )
            )

        boundary_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # --- Start-of-schedule evals at global step 0 (only once) ---
        for cb in per_eval_cbs:
            cb.on_training_start(agent)

        # --- Iterate phases; keep global timesteps continuous ---
        for i, ph in enumerate(phases):
            # Periodic evals during learning
            step_cb = FunctionCallback(
                lambda model: all(cb.on_step() for cb in per_eval_cbs)
            )
            if not hasattr(step_cb, "n_episodes"):
                step_cb.n_episodes = 0

            phase_steps = int(ph["steps"])
            s0 = int(agent.num_timesteps)
            target_total = s0 + phase_steps

            agent.learn(
                total_timesteps=target_total,
                reset_num_timesteps=False,
                callback=step_cb,
                progress_bar=False,
            )

            s1 = int(agent.num_timesteps)
            assert s1 - s0 == phase_steps, (
                f"[schedule] steps advanced {s1 - s0} (expect {phase_steps}) "
                f"at phase {i} label={label}"
            )

            # End-of-phase evals at the boundary step
            for cb in per_eval_cbs:
                cb.on_training_end()

            # Boundary tests immediately AFTER phase i (not sent to W&B)
            bkey = f"phase_{i}"
            boundary_cache.setdefault(bkey, {})
            for es in evals:
                name = es["name"]
                test_env = make_env(
                    self._env_spec(es["env"]), seed=self.seed + 1000 + i
                )

                test_env.reset(seed=self.seed + 1000 + i)
                g_mean, _ = evaluate_policy(
                    agent,
                    test_env,
                    self.n_eval,
                    deterministic=True,
                    render=False,
                    warn=False,
                )
                test_env.reset(seed=self.seed + 1001 + i)
                t_mean, _ = evaluate_policy(
                    agent,
                    test_env,
                    self.n_eval,
                    deterministic=False,
                    render=False,
                    warn=False,
                )
                test_env.close()

                boundary_cache[bkey].setdefault(name, {})
                boundary_cache[bkey][name]["greedy"] = float(g_mean)
                boundary_cache[bkey][name]["train"] = float(t_mean)

            # Optional boundary media
            if self.media_on:
                media_map = self._boundary_media_all(label, evals, agent, phase_idx=i)
                for nm, path in media_map.items():
                    boundary_cache[bkey][nm]["media_path"] = path

            # Switch env for next phase (do NOT call on_training_start again)
            if i < len(phases) - 1:
                next_env = DummyVecEnv(
                    [
                        lambda: make_env(
                            self._env_spec(phases[i + 1]["env"]), seed=self.seed
                        )
                    ]
                )
                agent.set_env(next_env)

        # End-of-training media
        media_paths = (
            self._final_media_all(label, evals, agent) if self.media_on else {}
        )

        # Pack outputs
        out: Dict[str, Any] = {
            "steps": steps_log,
            "boundary": boundary_cache,
            "media": media_paths,
        }
        for nm in eval_envs:
            out[nm] = {"greedy": greedy_log[nm], "train": trainp_log[nm]}

        # Close eval envs
        for v in eval_envs.values():
            v.close()

        return out

    def run(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"seed": self.seed, "baseline": None, "items": {}}

        # Run baseline schedule if requested
        if self.mode in ("baseline", "both"):
            out["baseline"] = self._run_schedule(
                "baseline", self.base_ph, self.base_evals
            )

        # NEW: per-item mode (run exactly one item label)
        if self.mode == "item":
            lb = self.single_item_label
            assert (
                lb is not None and lb in self.items_ph
            ), f"invalid single_item_label={lb}"
            out["items"][lb] = self._run_schedule(
                str(lb), self.items_ph[lb], self.evals_map[lb]
            )
            return out  # Important: do not run other items in this mode

        # Original multi-item mode: iterate all items sequentially
        if self.mode in ("items", "both"):
            for label, phases in self.items_ph.items():
                out["items"][label] = self._run_schedule(
                    str(label), phases, self.evals_map[label]
                )
        return out


# ------------------------------
# Driver
# ------------------------------
class RayCurriculumTrainer:
    def __init__(
        self,
        agent_ctor_path: str,
        agent_kwargs: Dict[str, Any],
        eval_every: int,
        n_eval_episodes: int,
        output_dir: Optional[str],
        *,
        wandb_step_base: int = 0,
        save_intermediate: bool = True,
        wandb_actor: Optional[ActorHandle] = None,
        media_opts: Optional[Dict[str, Any]] = None,
        run_baseline: bool = True,
        run_items: bool = True,
        metrics_opts: Optional[Dict[str, Any]] = None,
    ):
        """
        Slimmed constructor: removed manual concurrency knobs.
        Ray will naturally limit parallelism via num_cpus on each actor.
        """
        self.agent_ctor_path = agent_ctor_path
        self.agent_kwargs = dict(agent_kwargs)
        self.eval_every = int(eval_every)
        self.n_eval = int(n_eval_episodes)
        self.outdir = output_dir
        self.save_intermediate = bool(save_intermediate)
        self.wb = wandb_actor
        self.media_opts = dict(media_opts or {})
        self.wandb_step_base = int(wandb_step_base)
        self.run_baseline = bool(run_baseline)
        self.run_items = bool(run_items)
        if self.outdir is not None:
            ensure_dir(Path(self.outdir))
        mo = dict(metrics_opts or {})
        self.metrics_opts = {
            "enabled": bool(mo.get("enabled", True)),
            "ttt_fraction": float(mo.get("ttt_fraction", 0.90)),
            "ap_last_k": int(mo.get("ap_last_k", 10)),
            # it will be treated as baseline cap for backward convenience.
            "cap_steps": {
                "baseline": (mo.get("cap_steps", {}) or {}).get(
                    "baseline", mo.get("use_max_step", None)
                ),
                "target": (mo.get("cap_steps", {}) or {}).get("target", None),
                "item": (mo.get("cap_steps", {}) or {}).get("item", None),
            },
            "compute_greedy": bool(mo.get("compute_greedy", True)),
            "compute_train": bool(mo.get("compute_train", True)),
            "js_first_n": max(1, int(mo.get("js_first_n", 1))),
        }

        self.verbose = int(self.agent_kwargs.get("verbose", 0))

    # --- compute metrics independent of CSV/plots ---
    def _compute_metrics(
        self,
        summary: Dict[str, Any],
        item_phases_map: Dict[str, List[Dict[str, Any]]],
        evals_map: Dict[str, List[Dict[str, Any]]],
        baseline_evals: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Produce a stable metrics dict with float|None leaves.
        - No eval-name fallback.
        - None only when the metric is not definable by configuration or sampling.
        - Raise if configuration declares an eval that is missing in summary.

        Changes:
        - Per-scope cap steps: cfg['cap_steps'] with keys 'baseline'/'target'/'item'.
        - Remove all compatibility aliases.
        - Add real AUC metrics: auc_total / auc_p1 / auc_p2 (trapezoidal).
        - Redefine jumpstart to absolute levels:
            * target_start    : value at the start of the target curve.
            * p2_head         : mean of the first N points of the P2 segment
                                (N = cfg['js_first_n']; if N=1, it's value at boundary B).
            * baseline_B      : baseline Target value interpolated at B (if available).
          No baseline_start, no difference terms.
        """

        # ---- config snapshot ----
        cap = dict(self.metrics_opts.get("cap_steps", {}) or {})
        cfg = {
            "ttt_fraction": float(self.metrics_opts["ttt_fraction"]),
            "ap_last_k": int(self.metrics_opts["ap_last_k"]),
            "cap_steps": {
                "baseline": cap.get("baseline", None),
                "target": cap.get("target", None),
                "item": cap.get("item", None),
            },
            "compute_greedy": bool(self.metrics_opts["compute_greedy"]),
            "compute_train": bool(self.metrics_opts["compute_train"]),
            "js_first_n": int(self.metrics_opts["js_first_n"]),
        }
        out: Dict[str, Any] = {"config": cfg, "baseline": {}, "items": {}}

        def _cap_for(scope: str) -> Optional[int]:
            return cfg["cap_steps"].get(scope, None)

        # ---- baseline metrics (only if configuration declares Target and run_baseline=True) ----
        baseline_declares_target = (
            self.run_baseline
            and isinstance(baseline_evals, (list, tuple))
            and any(str(e.get("name")) == "Target" for e in baseline_evals)
        )

        baseline_block = summary.get("baseline", {}) or {}
        baseline_target = None
        if baseline_declares_target:
            if "Target" not in baseline_block:
                raise ValueError(
                    "baseline config declares 'Target' but summary.baseline['Target'] missing"
                )
            baseline_target = baseline_block["Target"]

            def _fill_baseline(chan: str):
                xs, ys = _ensure_curve(baseline_target, f"{chan}_mean")
                mean_total = _mean_over(
                    xs, ys, lo=None, hi=None, clamp_hi=_cap_for("baseline")
                )
                auc_total = _auc_over(
                    xs, ys, lo=None, hi=None, clamp_hi=_cap_for("baseline")
                )
                return {
                    "mean_total": mean_total,
                    "auc_total": auc_total,
                    "ap_last_k": _ap_last_k(
                        xs, ys, cfg["ap_last_k"], _cap_for("baseline")
                    ),
                    "ttt_fraction": _ttt_frac(
                        xs, ys, cfg["ttt_fraction"], _cap_for("baseline")
                    ),
                }

            out["baseline"]["greedy"] = (
                _fill_baseline("greedy")
                if cfg["compute_greedy"]
                else {
                    "mean_total": None,
                    "auc_total": None,
                    "ap_last_k": None,
                    "ttt_fraction": None,
                }
            )
            out["baseline"]["train"] = (
                _fill_baseline("train")
                if cfg["compute_train"]
                else {
                    "mean_total": None,
                    "auc_total": None,
                    "ap_last_k": None,
                    "ttt_fraction": None,
                }
            )
        else:
            out["baseline"]["greedy"] = {
                "mean_total": None,
                "auc_total": None,
                "ap_last_k": None,
                "ttt_fraction": None,
            }
            out["baseline"]["train"] = {
                "mean_total": None,
                "auc_total": None,
                "ap_last_k": None,
                "ttt_fraction": None,
            }

        # ---- per-item metrics ----
        for lb, eval_dict in (summary.get("items", {}) or {}).items():
            item_out: Dict[str, Any] = {
                "target": {
                    "greedy": {
                        "mean_total": None,
                        "mean_p1": None,
                        "mean_p2": None,
                        "auc_total": None,
                        "auc_p1": None,
                        "auc_p2": None,
                        "ap_last_k": None,
                        "ttt_fraction": None,
                    },
                    "train": {
                        "mean_total": None,
                        "mean_p1": None,
                        "mean_p2": None,
                        "auc_total": None,
                        "auc_p1": None,
                        "auc_p2": None,
                        "ap_last_k": None,
                        "ttt_fraction": None,
                    },
                },
                "item": {
                    "greedy": {
                        "mean_total": None,
                        "mean_p1": None,
                        "mean_p2": None,
                        "auc_total": None,
                        "auc_p1": None,
                        "auc_p2": None,
                        "ap_last_k": None,
                        "ttt_fraction": None,
                    },
                    "train": {
                        "mean_total": None,
                        "mean_p1": None,
                        "mean_p2": None,
                        "auc_total": None,
                        "auc_p1": None,
                        "auc_p2": None,
                        "ap_last_k": None,
                        "ttt_fraction": None,
                    },
                },
                "jumpstart": {
                    "greedy": {
                        "target_start": None,
                        "p2_head": None,
                        "baseline_B": None,
                    },
                    "train": {
                        "target_start": None,
                        "p2_head": None,
                        "baseline_B": None,
                    },
                },
            }

            # configuration gates
            item_cfg = evals_map.get(lb, []) or []
            has_target_eval = any(str(e.get("name")) == "Target" for e in item_cfg)
            has_self_eval = any(str(e.get("name")) == str(lb) for e in item_cfg)

            # required blocks
            tgt_block = None
            if has_target_eval:
                if "Target" not in eval_dict:
                    raise ValueError(
                        f"item '{lb}' config declares Target eval but summary missing it"
                    )
                tgt_block = eval_dict["Target"]

            self_block = None
            if has_self_eval:
                if lb not in eval_dict:
                    raise ValueError(
                        f"item '{lb}' config declares self eval '{lb}' but summary missing it"
                    )
                self_block = eval_dict[lb]

            # phase boundary (for p1/p2/jumpstart)
            phs = item_phases_map.get(lb, []) or []
            if len(phs) >= 2:
                B = int(phs[0].get("steps", 0))
                if B < 0:
                    raise ValueError(f"invalid phase boundary for item '{lb}': {B}")
            else:
                B = None

            # --- pack one env/channel ---
            def _pack_env(block, chan: str, scope: str):
                """
                scope: 'target' or 'item' -> choose its own cap for *_total & last-k/ttt.
                p1/p2 segments are computed on the natural segments (no cap).
                """
                if block is None:
                    return {
                        "mean_total": None,
                        "mean_p1": None,
                        "mean_p2": None,
                        "auc_total": None,
                        "auc_p1": None,
                        "auc_p2": None,
                        "ap_last_k": None,
                        "ttt_fraction": None,
                    }

                xs, ys = _ensure_curve(block, f"{chan}_mean")
                mean_total = _mean_over(
                    xs, ys, lo=None, hi=None, clamp_hi=_cap_for(scope)
                )
                auc_total = _auc_over(
                    xs, ys, lo=None, hi=None, clamp_hi=_cap_for(scope)
                )

                outp = {
                    "mean_total": mean_total,
                    "auc_total": auc_total,
                    "ap_last_k": _ap_last_k(xs, ys, cfg["ap_last_k"], _cap_for(scope)),
                    "ttt_fraction": _ttt_frac(
                        xs, ys, cfg["ttt_fraction"], _cap_for(scope)
                    ),
                    "mean_p1": None,
                    "mean_p2": None,
                    "auc_p1": None,
                    "auc_p2": None,
                }

                if B is not None and xs.size:
                    start_s = float(xs[0])
                    end_s = float(xs[-1])
                    if start_s <= B <= end_s:
                        outp["mean_p1"] = _mean_over(
                            xs, ys, lo=start_s, hi=float(B), clamp_hi=None
                        )
                        outp["mean_p2"] = _mean_over(
                            xs, ys, lo=float(B), hi=end_s, clamp_hi=None
                        )
                        outp["auc_p1"] = _auc_over(
                            xs, ys, lo=start_s, hi=float(B), clamp_hi=None
                        )
                        outp["auc_p2"] = _auc_over(
                            xs, ys, lo=float(B), hi=end_s, clamp_hi=None
                        )
                return outp

            # target/item metrics
            if cfg["compute_greedy"]:
                item_out["target"]["greedy"] = _pack_env(
                    tgt_block, "greedy", scope="target"
                )
                item_out["item"]["greedy"] = _pack_env(
                    self_block, "greedy", scope="item"
                )
            if cfg["compute_train"]:
                item_out["target"]["train"] = _pack_env(
                    tgt_block, "train", scope="target"
                )
                item_out["item"]["train"] = _pack_env(self_block, "train", scope="item")

            # ---- jumpstart: absolute levels (target_start, p2_head, baseline_B) ----
            if B is not None and tgt_block is not None:
                baseline_for_js = baseline_target if baseline_declares_target else None
                if cfg["compute_greedy"]:
                    item_out["jumpstart"]["greedy"] = _jumpstart_fields(
                        tgt_block=tgt_block,
                        baseline_target=baseline_for_js,
                        B=B,
                        chan="greedy",
                        first_n=cfg["js_first_n"],
                    )
                if cfg["compute_train"]:
                    item_out["jumpstart"]["train"] = _jumpstart_fields(
                        tgt_block=tgt_block,
                        baseline_target=baseline_for_js,
                        B=B,
                        chan="train",
                        first_n=cfg["js_first_n"],
                    )

            out["items"][str(lb)] = item_out

        return out

    def run(
        self,
        seeds: List[int],
        envs: Dict[str, Dict[str, Any]],
        baseline_phases: List[Dict[str, Any]],
        baseline_evals: List[Dict[str, Any]],
        item_phases_map: Dict[str, List[Dict[str, Any]]],
        evals_map: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Submit all per-(seed, task) actors at once (fire-and-forget).
        Print on submit and on completion; no manual throttling.
        Collector actor consumes 0 CPU, serializes aggregation with max_concurrency=1.
        Only the first seed produces media.
        """
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        seeds = sorted(map(int, seeds))

        # ===== Normalize baseline total steps to match curriculum total (unchanged) =====
        def _sum_steps(phs: List[Dict[str, Any]]) -> int:
            return sum(int(p.get("steps", 0)) for p in phs)

        max_item_total = 0
        if self.run_items:
            for _, phs in item_phases_map.items():
                max_item_total = max(max_item_total, _sum_steps(phs))

        if self.run_baseline and max_item_total > 0:
            base_total = _sum_steps(baseline_phases)
            if base_total != max_item_total:
                delta = max_item_total - base_total
                new_phases = [dict(p) for p in baseline_phases]
                if not new_phases:
                    raise ValueError(
                        "baseline_phases is empty; cannot normalize steps."
                    )
                if delta > 0:
                    new_phases[-1]["steps"] = (
                        int(new_phases[-1].get("steps", 0)) + delta
                    )
                    if self.verbose > 0:
                        print(
                            f"[info] baseline steps extended by {delta} to {max_item_total}."
                        )
                else:
                    need_cut = -delta
                    idx = len(new_phases) - 1
                    while need_cut > 0 and idx >= 0:
                        cur = int(new_phases[idx].get("steps", 0))
                        take = min(cur, need_cut)
                        new_phases[idx]["steps"] = cur - take
                        need_cut -= take
                        if new_phases[idx]["steps"] == 0 and idx > 0:
                            new_phases.pop(idx)
                        idx -= 1
                    if self.verbose > 0:
                        print(
                            f"[info] baseline steps trimmed by {-delta} to {max_item_total}."
                        )
                baseline_phases = new_phases

        # ===== Shared collector (0 CPU; serial queue) =====
        collector = Collector.options(num_cpus=0, max_concurrency=1).remote(
            seeds=seeds,
            keep_intermediate=self.save_intermediate,
            wandb_actor=self.wb,
            step_base=int(self.wandb_step_base),
            flush_interval_s=0.2,
            max_queue=16384,
            verbose=self.verbose,
        )

        # ===== Prepare all tasks (baseline + per-item) and submit all =====
        item_labels = list(item_phases_map.keys()) if self.run_items else []

        futures2meta: Dict[Any, Tuple[int, str, Optional[str]]] = {}
        media_root = (
            os.path.join(self.outdir, "media") if self.outdir is not None else None
        )
        if media_root is not None:
            ensure_dir(Path(media_root))

        first_seed = seeds[0] if seeds else None

        def submit(seed: int, mode: str, item_label: Optional[str], save_media: bool):
            """Create one SeedTrainer actor and submit its run() call."""
            actor = SeedTrainer.options(num_cpus=1).remote(
                seed=seed,
                envs=envs,
                baseline_phases=baseline_phases,
                baseline_evals=baseline_evals,
                item_phases_map=item_phases_map,
                evals_map=evals_map,
                agent_ctor_path=self.agent_ctor_path,
                agent_kwargs=self.agent_kwargs,
                eval_every=self.eval_every,
                n_eval_episodes=self.n_eval,
                collector=collector if self.save_intermediate else None,
                collect_intermediate=self.save_intermediate,
                media_root=(media_root if save_media else None),
                media_opts=self.media_opts,
                mode=mode,  # "baseline" or "item"
                single_item_label=item_label,  # None for baseline; label for item
            )
            fut = actor.run.remote()
            futures2meta[fut] = (seed, mode, item_label)

        # Submit all baseline tasks (one per seed)
        if self.run_baseline:
            for s in seeds:
                save_media = s == first_seed  # only the first seed records media
                if self.verbose > 0:
                    print(
                        f"[seed {s}] submit (baseline) [media={'on' if save_media else 'off'}]."
                    )
                submit(s, "baseline", None, save_media)

        # Submit all per-item tasks (one task per (seed, item))
        if self.run_items:
            for s in seeds:
                for lb in item_labels:
                    save_media = s == first_seed  # only the first seed records media
                    if self.verbose > 0:
                        print(
                            f"[seed {s}] submit (item/{lb}) [media={'on' if save_media else 'off'}]."
                        )
                    submit(s, "item", lb, save_media)

        # ===== Wait for completion and print who finished =====
        results: List[Dict[str, Any]] = []
        while futures2meta:
            done, _ = ray.wait(list(futures2meta.keys()), timeout=None, num_returns=1)
            for fut in done:
                seed, mode, lb = futures2meta.pop(fut)
                res = ray.get(fut)
                results.append(res)
                tag = f"{mode}{'/' + lb if lb else ''}"
                if self.verbose > 0:
                    print(f"[seed {seed}] done ({tag}).")

        # ===== Merge baseline/items per seed (unchanged) =====
        by_seed: Dict[int, Dict[str, Any]] = {}
        for r in results:
            sid = int(r.get("seed", -1))
            assert sid != -1, "invalid seed in result"
            tgt = by_seed.setdefault(sid, {"seed": sid, "baseline": None, "items": {}})
            if r.get("baseline") is not None:
                assert tgt["baseline"] is None, f"duplicate baseline for seed {sid}"
                tgt["baseline"] = r["baseline"]
            if "items" in r and isinstance(r["items"], dict):
                for lb, dd in r["items"].items():
                    assert (
                        lb not in tgt["items"]
                    ), f"duplicate items[{lb}] for seed {sid}"
                    tgt["items"][lb] = dd

        if self.run_baseline:
            missing = [s for s, v in by_seed.items() if v["baseline"] is None]
            assert not missing, f"baseline missing for seeds: {missing}"
        if self.run_items:
            missing = [s for s, v in by_seed.items() if not v["items"]]
            assert not missing, f"items missing for seeds: {missing}"

        merged_results: List[Dict[str, Any]] = [
            by_seed[s] for s in sorted(by_seed.keys())
        ]

        # ===== Optional timeline dump =====
        if self.outdir is not None and self.save_intermediate:
            try:
                tl = ray.get(collector.timeline.remote())
                with open(os.path.join(self.outdir, "online_timeline.json"), "w") as f:
                    json.dump(tl, f)
            except Exception as e:
                print(f"[driver] dump timeline failed: {e}")

        # ===== Aggregate & metrics =====
        summary = self._aggregate(merged_results, item_phases_map)
        if self.metrics_opts.get("enabled", True):
            metrics = self._compute_metrics(
                summary,
                item_phases_map=item_phases_map,
                evals_map=evals_map,
                baseline_evals=baseline_evals,
            )
            summary["metrics"] = metrics

        # ===== Save per-seed JSONs =====
        if self.outdir is not None:
            ensure_dir(self.outdir)
            with open(os.path.join(self.outdir, "final_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            seeds_dir = os.path.join(self.outdir, "seeds")
            ensure_dir(seeds_dir)
            for r in merged_results:
                sid = int(r.get("seed", -1))
                with open(os.path.join(seeds_dir, f"seed_{sid}.json"), "w") as f:
                    json.dump(r, f)

        # ===== Upload videos (kept; W&B IO guarded separately) =====
        if self.wb is not None and self.outdir is not None:
            media_root_dir = os.path.join(self.outdir, "media")
            if os.path.isdir(media_root_dir):
                to_log = []

                def _enqueue(
                    seed_id: int, label: str, media_map: Dict[str, Optional[str]]
                ):
                    if not isinstance(media_map, dict):
                        return
                    for name, path in media_map.items():
                        if path:
                            to_log.append(
                                (f"media/seed_{seed_id}/{label}/{name}", path)
                            )

                def _enqueue_boundary(
                    seed_id: int, label: str, boundary_map: Dict[str, Any]
                ):
                    if not isinstance(boundary_map, dict):
                        return
                    for phase_key, per_eval in boundary_map.items():
                        if not isinstance(per_eval, dict):
                            continue
                        for eval_name, rec in per_eval.items():
                            p = (rec or {}).get("media_path", None)
                            if p:
                                to_log.append(
                                    (
                                        f"media/seed_{seed_id}/{label}/{eval_name}_{phase_key}",
                                        p,
                                    )
                                )

                for r in merged_results:
                    sd = int(r.get("seed", 0))
                    base = r.get("baseline", None)
                    if base and self.run_baseline:
                        _enqueue(sd, "baseline", base.get("media", {}) or {})
                        _enqueue_boundary(
                            sd, "baseline", base.get("boundary", {}) or {}
                        )
                    items = r.get("items", {}) or {}
                    for lb, d in items.items():
                        _enqueue(sd, str(lb), (d or {}).get("media", {}) or {})
                        _enqueue_boundary(
                            sd, str(lb), (d or {}).get("boundary", {}) or {}
                        )

                fps = int(self.media_opts.get("fps", 8))
                for key, path in to_log:
                    if not (path and os.path.isfile(path)):
                        if self.verbose > 0:
                            print(f"[W&B] skip missing video: {path}")
                        continue
                    try:
                        fmt = "gif" if str(path).lower().endswith(".gif") else None
                        self.wb.log_video.remote(key, path, fps=fps, fmt=fmt)
                    except Exception as e:
                        print(f"[W&B] schedule video upload failed for {key}: {e}")

        # ===== CSV & plots (gated strictly by configuration; no eval-name fallback) =====
        if self.outdir is not None:
            # Does configuration declare baseline Target?
            baseline_declares_target = (
                self.run_baseline
                and isinstance(baseline_evals, (list, tuple))
                and any(str(e.get("name")) == "Target" for e in baseline_evals)
            )

            steps_base = None
            base_curves = None
            if baseline_declares_target:
                if "baseline" not in summary or "Target" not in summary["baseline"]:
                    raise ValueError(
                        "baseline config declares 'Target' but summary.baseline['Target'] missing"
                    )
                steps_base = np.asarray(
                    summary["baseline"]["Target"]["steps"], dtype=int
                )
                base = summary["baseline"]["Target"]
                base_curves = {
                    "greedy_mean": np.asarray(base["greedy_mean"], dtype=float),
                    "greedy_std": np.asarray(base["greedy_std"], dtype=float),
                    "train_mean": np.asarray(base["train_mean"], dtype=float),
                    "train_std": np.asarray(base["train_std"], dtype=float),
                }

            for lb, dd in summary["items"].items():
                item_dir = os.path.join(self.outdir, str(lb))
                ensure_dir(Path(item_dir))

                phs = item_phases_map.get(lb, []) or []
                # phase boundaries for plotting labels
                item_boundaries: List[int] = []
                acc = 0
                for ph in phs[:-1]:
                    acc += int(ph.get("steps", 0))
                    item_boundaries.append(acc)
                boundaries_str = (
                    "-".join(str(b) for b in item_boundaries)
                    if item_boundaries
                    else "none"
                )

                # What evals are declared for this item?
                item_cfg = evals_map.get(lb, []) or []
                has_target_eval = any(str(e.get("name")) == "Target" for e in item_cfg)
                has_self_eval = any(str(e.get("name")) == str(lb) for e in item_cfg)

                # ---- CSV dumps ----
                if has_target_eval:
                    if "Target" not in dd:
                        raise ValueError(
                            f"item '{lb}' config declares Target eval but summary missing it"
                        )
                    tgt = dd["Target"]
                    steps_tgt = np.asarray(tgt["steps"], dtype=int)
                    tgt_g_mean = np.asarray(tgt["greedy_mean"], dtype=float)
                    tgt_g_std = np.asarray(tgt["greedy_std"], dtype=float)
                    tgt_t_mean = np.asarray(tgt["train_mean"], dtype=float)
                    tgt_t_std = np.asarray(tgt["train_std"], dtype=float)

                    save_csv(
                        os.path.join(
                            item_dir,
                            f"curriculum_eval_target_phase_{boundaries_str}.csv",
                        ),
                        steps_tgt,
                        tgt_g_mean,
                        tgt_g_std,
                        header="greedy",
                    )
                    save_csv(
                        os.path.join(
                            item_dir,
                            f"curriculum_eval_target_train_phase_{boundaries_str}.csv",
                        ),
                        steps_tgt,
                        tgt_t_mean,
                        tgt_t_std,
                        header="train",
                    )

                if has_self_eval:
                    if lb not in dd:
                        raise ValueError(
                            f"item '{lb}' config declares self eval '{lb}' but summary missing it"
                        )
                    selfb = dd[lb]
                    steps_self = np.asarray(selfb["steps"], dtype=int)
                    self_g_mean = np.asarray(selfb["greedy_mean"], dtype=float)
                    self_g_std = np.asarray(selfb["greedy_std"], dtype=float)
                    self_t_mean = np.asarray(selfb["train_mean"], dtype=float)
                    self_t_std = np.asarray(selfb["train_std"], dtype=float)

                    save_csv(
                        os.path.join(
                            item_dir, f"curriculum_eval_item_phase_{boundaries_str}.csv"
                        ),
                        steps_self,
                        self_g_mean,
                        self_g_std,
                        header="greedy",
                    )
                    save_csv(
                        os.path.join(
                            item_dir,
                            f"curriculum_eval_item_train_phase_{boundaries_str}.csv",
                        ),
                        steps_self,
                        self_t_mean,
                        self_t_std,
                        header="train",
                    )

                # ---- Pairwise plot (baseline Target vs item Target) ----
                if baseline_declares_target and has_target_eval:
                    if base_curves is None or steps_base is None:
                        raise ValueError(
                            "baseline Target curves are required but missing"
                        )
                    if "Target" not in dd:
                        raise ValueError(
                            f"item '{lb}' Target curves required for plotting but missing"
                        )

                    item_total_steps = sum(int(p.get("steps", 0)) for p in phs)
                    assert item_total_steps > 0, f"empty phases for item {lb}"

                    tgt = dd["Target"]
                    steps_tgt = np.asarray(tgt["steps"], dtype=int)
                    tgt_g_mean = np.asarray(tgt["greedy_mean"], dtype=float)
                    tgt_g_std = np.asarray(tgt["greedy_std"], dtype=float)
                    tgt_t_mean = np.asarray(tgt["train_mean"], dtype=float)
                    tgt_t_std = np.asarray(tgt["train_std"], dtype=float)

                    def _tail(x):
                        return int(x[-1]) if len(x) else None

                    msg_base = f"[plot-check] baseline tail={_tail(steps_base)} len={len(steps_base)}"
                    msg_tgt = f"[plot-check] tgt({lb}) tail={_tail(steps_tgt)} len={len(steps_tgt)}"
                    assert (
                        _tail(steps_base) == item_total_steps
                    ), f"baseline total({_tail(steps_base)}) != item_total({item_total_steps}). {msg_base}"
                    assert (
                        _tail(steps_tgt) == item_total_steps
                    ), f"target total({_tail(steps_tgt)}) != item_total({item_total_steps}). {msg_tgt}"

                    X = steps_base

                    def _interp_strict(x_old, mean, std, x_new, name):
                        assert x_old[0] <= x_new[0], f"{name}: x_old starts after x_new"
                        assert (
                            x_old[-1] >= x_new[-1]
                        ), f"{name}: x_old ends before x_new"
                        return (
                            np.interp(x_new, x_old, mean),
                            np.interp(x_new, x_old, std),
                        )

                    base_g_mean_i, base_g_std_i = _interp_strict(
                        steps_base,
                        base_curves["greedy_mean"],
                        base_curves["greedy_std"],
                        X,
                        "baseline/greedy",
                    )
                    base_t_mean_i, base_t_std_i = _interp_strict(
                        steps_base,
                        base_curves["train_mean"],
                        base_curves["train_std"],
                        X,
                        "baseline/train",
                    )

                    tgt_g_mean_i, tgt_g_std_i = _interp_strict(
                        steps_tgt, tgt_g_mean, tgt_g_std, X, f"tgt({lb})/greedy"
                    )
                    tgt_t_mean_i, tgt_t_std_i = _interp_strict(
                        steps_tgt, tgt_t_mean, tgt_t_std, X, f"tgt({lb})/train"
                    )

                    curves_item_i = None
                    if has_self_eval:
                        selfb = dd[lb]
                        steps_self = np.asarray(selfb["steps"], dtype=int)
                        assert (
                            steps_self[-1] == item_total_steps
                        ), f"self total({steps_self[-1]}) != item_total({item_total_steps}) for {lb}"
                        self_g_mean_i, self_g_std_i = _interp_strict(
                            steps_self,
                            np.asarray(selfb["greedy_mean"], float),
                            np.asarray(selfb["greedy_std"], float),
                            X,
                            f"item({lb})/greedy",
                        )
                        self_t_mean_i, self_t_std_i = _interp_strict(
                            steps_self,
                            np.asarray(selfb["train_mean"], float),
                            np.asarray(selfb["train_std"], float),
                            X,
                            f"item({lb})/train",
                        )
                        curves_item_i = {
                            "greedy_mean": self_g_mean_i,
                            "greedy_std": self_g_std_i,
                            "train_mean": self_t_mean_i,
                            "train_std": self_t_std_i,
                        }

                    png_path = os.path.join(
                        item_dir, f"pairwise_{lb}_phase_{boundaries_str}.png"
                    )
                    plot_pairwise(
                        out_png_path=png_path,
                        checkpoints=X,
                        phase_boundaries=item_boundaries,
                        title_prefix=f"Pairwise for '{lb}'",
                        baseline={
                            "greedy_mean": base_g_mean_i,
                            "greedy_std": base_g_std_i,
                            "train_mean": base_t_mean_i,
                            "train_std": base_t_std_i,
                        },
                        curves_target={
                            "greedy_mean": tgt_g_mean_i,
                            "greedy_std": tgt_g_std_i,
                            "train_mean": tgt_t_mean_i,
                            "train_std": tgt_t_std_i,
                        },
                        curves_source=curves_item_i,  # optional (self-eval)
                    )

                    if self.wb is not None:
                        self.wb.log_image.remote(
                            key=f"images/pairwise_{lb}_phase_{boundaries_str}",
                            path=png_path,
                            caption=f"pairwise {lb} (phase boundaries: {boundaries_str})",
                        )

        return summary

    # --- offline aggregation ---
    def _aggregate(
        self,
        per_seed: List[Dict[str, Any]],
        item_phases_map: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Aggregate per-seed traces. No eval-name fallback; keep all evals explicitly."""
        assert per_seed, "no results"
        labels = sorted(item_phases_map.keys())

        def _stack(ll: List[List[float]]) -> Tuple[List[float], List[float]]:
            arr = np.asarray([np.asarray(x, float) for x in ll])
            assert arr.ndim == 2, "invalid series shape"
            return arr.mean(0).tolist(), arr.std(0).tolist()

        out: Dict[str, Any] = {"steps": None, "baseline": {}, "items": {}}

        # -------- baseline: aggregate all eval names if present (no fallback) --------
        first_base = per_seed[0].get("baseline", None)
        has_baseline = (
            isinstance(first_base, dict)
            and "steps" in first_base
            and isinstance(first_base["steps"], dict)
            and len(first_base["steps"]) > 0
        )
        if has_baseline:
            base_names = list(first_base["steps"].keys())
            assert base_names, "no baseline evals"

            for eval_name in base_names:
                ref_steps = per_seed[0]["baseline"]["steps"][eval_name]
                for r in per_seed:
                    assert (
                        r["baseline"]["steps"][eval_name] == ref_steps
                    ), f"baseline steps mismatch on eval '{eval_name}'"

                g_mean, g_std = _stack(
                    [r["baseline"][eval_name]["greedy"] for r in per_seed]
                )
                t_mean, t_std = _stack(
                    [r["baseline"][eval_name]["train"] for r in per_seed]
                )
                out["baseline"][eval_name] = {
                    "steps": ref_steps,
                    "greedy_mean": g_mean,
                    "greedy_std": g_std,
                    "train_mean": t_mean,
                    "train_std": t_std,
                }

            # Reference steps only if baseline contains "Target"
            out["steps"] = (
                out["baseline"]["Target"]["steps"]
                if "Target" in out["baseline"]
                else None
            )

        # -------- items: aggregate each configured eval name for each item --------
        for lb in labels:
            assert (
                "items" in per_seed[0] and lb in per_seed[0]["items"]
            ), f"missing item '{lb}'"
            # available eval names for this item (skip helper keys)
            eval_names = [
                k
                for k in per_seed[0]["items"][lb].keys()
                if k not in ("steps", "boundary", "media")
            ]
            steps_ref_map = per_seed[0]["items"][lb]["steps"]
            agg: Dict[str, Any] = {}

            for nm in eval_names:
                ref_steps = steps_ref_map[nm]
                for r in per_seed:
                    assert (
                        r["items"][lb]["steps"][nm] == ref_steps
                    ), f"steps mismatch for item '{lb}' eval '{nm}'"

                gm, gs = _stack([r["items"][lb][nm]["greedy"] for r in per_seed])
                tm, ts = _stack([r["items"][lb][nm]["train"] for r in per_seed])
                agg[nm] = {
                    "steps": ref_steps,
                    "greedy_mean": gm,
                    "greedy_std": gs,
                    "train_mean": tm,
                    "train_std": ts,
                }
            out["items"][lb] = agg

        return out


# ------------------------------
# Convenience API
# ------------------------------
def run_curriculum(
    *,
    seeds: List[int],
    envs: Dict[str, Dict[str, Any]],
    baseline_phases: List[Dict[str, Any]],
    baseline_evals: List[Dict[str, Any]],
    item_phases_map: Dict[str, List[Dict[str, Any]]],
    evals_map: Dict[str, List[Dict[str, Any]]],
    agent_ctor_path: str,
    agent_kwargs: Dict[str, Any],
    eval_every: int,
    n_eval_episodes: int,
    output_dir: Optional[str] = None,  # keep default
    save_intermediate: bool = True,
    wandb_actor: Optional[ActorHandle] = None,
    media_opts: Optional[Dict[str, Any]] = None,
    wandb_step_base: int = 0,
    run_baseline: bool = True,
    run_items: bool = True,
    metrics_opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Slim convenience API: no manual concurrency controls anymore.
    """
    trainer = RayCurriculumTrainer(
        agent_ctor_path=agent_ctor_path,
        agent_kwargs=agent_kwargs,
        eval_every=eval_every,
        n_eval_episodes=n_eval_episodes,
        output_dir=output_dir,
        save_intermediate=save_intermediate,
        wandb_actor=wandb_actor,
        media_opts=media_opts,
        wandb_step_base=wandb_step_base,
        run_baseline=run_baseline,
        run_items=run_items,
        metrics_opts=metrics_opts,
    )
    return trainer.run(
        seeds=seeds,
        envs=envs,
        baseline_phases=baseline_phases,
        baseline_evals=baseline_evals,
        item_phases_map=item_phases_map,
        evals_map=evals_map,
    )
