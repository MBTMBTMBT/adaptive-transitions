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
    def __init__(self,
                 seeds: List[int],
                 keep_intermediate: bool = True,
                 wandb_actor: Optional[ActorHandle] = None,
                 step_base: int = 0,
                 flush_interval_s: float = 0.5,
                 max_queue: int = 16384):
        self.seeds = list(map(int, seeds))
        self.keep = bool(keep_intermediate)
        self._wb = wandb_actor
        self._step_base = int(step_base)

        # Optional raw timeline for debugging
        self._timeline: Dict[int, List[Dict[str, Any]]] = {s: [] for s in self.seeds}

        # Separate in-flight buckets for baseline vs curriculum(CL)
        # key: (label, eval_name, eval_idx) -> {seed: (greedy, train, step)}
        self._bucket_base: Dict[Tuple[str, str, int], Dict[int, Tuple[float, float, int]]] = {}
        self._bucket_cl:   Dict[Tuple[str, str, int], Dict[int, Tuple[float, float, int]]] = {}

        # Per-series last eval_idx accepted (monotonic)
        self._last_idx_base: Dict[Tuple[str, str], int] = {}
        self._last_idx_cl:   Dict[Tuple[str, str], int] = {}

        # Per-series define_metric guard
        self._defined_series: set[Tuple[str, str]] = set()

        # Async queue
        self._q: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=int(max_queue))
        self._flush_interval_s = float(flush_interval_s)

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
                    try:
                        print(f"[Collector] drop non-monotonic idx for online/{label}/{eval_name}: "
                              f"{eval_idx} <= {prev}")
                    except Exception:
                        pass
                    del buckets[key]
                else:
                    # First time for this series: bind axis to series_step
                    if series_key not in self._defined_series:
                        try:
                            self._wb.define_metric.remote(
                                f"online/{label}/{eval_name}/*",
                                step_metric=f"online/{label}/{eval_name}/series_step"
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
                        f"{series}/greedy_std":  float(g_arr.std()),
                        f"{series}/train_mean":  float(t_arr.mean()),
                        f"{series}/train_std":   float(t_arr.std()),
                        f"{series}/series_step": out_step,
                        f"{series}/series_idx":  int(eval_idx),
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
    def __init__(self, *,
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
                 seed_id: Optional[int] = None):
        self.model = None
        self.eval_env = eval_env
        self.eval_every = int(eval_every)
        assert self.eval_every > 0, "eval_every must be > 0"
        self.n_eval = int(n_eval_episodes)
        self.steps_log = steps_log
        self.greedy_log = greedy_log
        self.train_log  = train_log
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
        g_mean, _ = evaluate_policy(self.model, self.eval_env,
                                    n_eval_episodes=self.n_eval,
                                    deterministic=True, render=False, warn=False)
        # train-policy
        self.eval_env.reset(seed=self.seed_base + 2 * self._eval_count + 1)
        t_mean, _ = evaluate_policy(self.model, self.eval_env,
                                    n_eval_episodes=self.n_eval,
                                    deterministic=False, render=False, warn=False)

        step = int(self.model.num_timesteps)
        self.steps_log.append(step)
        self.greedy_log.append(float(g_mean))
        self.train_log.append(float(t_mean))

        if self.collector is not None and self.seed_id is not None:
            try:
                self.collector.report.remote({
                    "seed": self.seed_id,
                    "label": self.label,
                    "eval_name": self.eval_name,
                    "eval_idx": int(self._eval_count),  # <<< add this line
                    "step": step,
                    "greedy": float(g_mean),
                    "train": float(t_mean),
                    "wall_time": time.time(),
                })
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
    def __init__(self,
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
                 mode: str = "both"):  # "baseline" | "items" | "both"
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

        # media settings
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
        return self.envs[key]  # fallback if you ever pass a flat registry

    # --- eval helper ---
    def _eval_once(self, model, env, det: bool, seed_base: int) -> float:
        env.reset(seed=seed_base)
        m, _ = evaluate_policy(model, env, self.n_eval, deterministic=det, render=False, warn=False)
        return float(m)

    # --- media helpers ---
    def _save_media(self, label: str, eval_name: str, env_key: str, suffix: str, agent) -> Optional[str]:
        if not self.media_on:
            return None
        env = make_env(self._env_spec(env_key), seed=self.seed + self.mo_start_seed_base)
        # subdir: <media_root>/seed_<id>/<label>/
        subdir = os.path.join(self.media_root, f"seed_{self.seed}", label)
        ensure_dir(Path(subdir))
        fmt = self.mo_fmt.lower()
        out_path = os.path.join(subdir, f"{eval_name}{suffix}.{fmt if fmt in ('gif','mp4') else 'gif'}")
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

    def _boundary_media_all(self, label: str, eval_specs: List[Dict[str, Any]], agent, phase_idx: int) -> Dict[str, Optional[str]]:
        """Record media for all eval envs at a phase boundary (if media_on)."""
        out: Dict[str, Optional[str]] = {}
        if not self.media_on:
            return out
        for es in eval_specs:
            nm = es["name"]
            p = self._save_media(label, nm, es["env"], suffix=f"_phase{phase_idx}", agent=agent)
            out[nm] = p
        return out

    def _final_media_all(self, label: str, eval_specs: List[Dict[str, Any]], agent) -> Dict[str, Optional[str]]:
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
    def _run_schedule(self, label: str,
                      phases: List[Dict[str, Any]],
                      evals: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Build training env & agent on phase-0 env
        train_env = DummyVecEnv([lambda: make_env(self._env_spec(phases[0]["env"]), seed=self.seed)])
        agent_cls = _import(self.agent_ctor_path)
        agent = agent_cls(env=train_env, seed=self.seed, **self.agent_kwargs)

        # Fixed eval envs + storages
        eval_envs: Dict[str, Any] = {es["name"]: make_env(self._env_spec(es["env"]), seed=12345) for es in evals}
        steps_log: Dict[str, List[int]] = {nm: [] for nm in eval_envs}
        greedy_log: Dict[str, List[float]] = {nm: [] for nm in eval_envs}
        trainp_log: Dict[str, List[float]] = {nm: [] for nm in eval_envs}

        # One callback per eval env
        per_eval_cbs: List[PeriodicEvalCallback] = []
        for es in evals:
            nm = es["name"]
            per_eval_cbs.append(PeriodicEvalCallback(
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
            ))

        boundary_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # --- Start-of-schedule evals at global step 0 (only once) ---
        for cb in per_eval_cbs:
            cb.on_training_start(agent)

        # --- Iterate phases; keep global timesteps continuous ---
        for i, ph in enumerate(phases):
            # Periodic evals during learning
            step_cb = FunctionCallback(lambda model: all(cb.on_step() for cb in per_eval_cbs))
            if not hasattr(step_cb, "n_episodes"):
                step_cb.n_episodes = 0

            phase_steps = int(ph["steps"])
            # s0 = int(agent.num_timesteps)
            agent.learn(total_timesteps=phase_steps,
                        reset_num_timesteps=False,
                        callback=step_cb,
                        progress_bar=False)
            # s1 = int(agent.num_timesteps)
            # if s1 - s0 != phase_steps:
            #     print(f"[warn] num_timesteps advanced {s1 - s0} (expect {phase_steps}) at phase {i}.")

            # End-of-phase evals at the boundary step
            for cb in per_eval_cbs:
                cb.on_training_end()

            # Boundary tests immediately AFTER phase i (not sent to W&B)
            bkey = f"phase_{i}"
            boundary_cache.setdefault(bkey, {})
            for es in evals:
                name = es["name"]
                test_env = make_env(self._env_spec(es["env"]), seed=self.seed + 1000 + i)

                test_env.reset(seed=self.seed + 1000 + i)
                g_mean, _ = evaluate_policy(agent, test_env, self.n_eval,
                                            deterministic=True, render=False, warn=False)
                test_env.reset(seed=self.seed + 1001 + i)
                t_mean, _ = evaluate_policy(agent, test_env, self.n_eval,
                                            deterministic=False, render=False, warn=False)
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
                next_env = DummyVecEnv([lambda: make_env(self._env_spec(phases[i + 1]["env"]), seed=self.seed)])
                agent.set_env(next_env)

        # End-of-training media
        media_paths = self._final_media_all(label, evals, agent) if self.media_on else {}

        # Pack outputs
        out: Dict[str, Any] = {"steps": steps_log, "boundary": boundary_cache, "media": media_paths}
        for nm in eval_envs:
            out[nm] = {"greedy": greedy_log[nm], "train": trainp_log[nm]}

        # Close eval envs
        for v in eval_envs.values():
            v.close()

        return out

    def run(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"seed": self.seed, "baseline": None, "items": {}}
        if self.mode in ("baseline", "both"):
            out["baseline"] = self._run_schedule("baseline", self.base_ph, self.base_evals)
        if self.mode in ("items", "both"):
            for label, phases in self.items_ph.items():
                out["items"][label] = self._run_schedule(str(label), phases, self.evals_map[label])
        return out


# ------------------------------
# Driver
# ------------------------------
class RayCurriculumTrainer:
    def __init__(self,
                 agent_ctor_path: str,
                 agent_kwargs: Dict[str, Any],
                 eval_every: int,
                 n_eval_episodes: int,
                 output_dir: Optional[str],
                 *,
                 wandb_step_base: int = 0,
                 max_concurrency: Optional[int] = None,
                 save_intermediate: bool = True,
                 wandb_actor: Optional[ActorHandle] = None,
                 media_opts: Optional[Dict[str, Any]] = None,
                 run_baseline: bool = True,   # NEW
                 run_items: bool = True):     # NEW
        self.agent_ctor_path = agent_ctor_path
        self.agent_kwargs = dict(agent_kwargs)
        self.eval_every = int(eval_every)
        self.n_eval = int(n_eval_episodes)
        self.outdir = output_dir
        self.max_conc = max_concurrency
        self.save_intermediate = bool(save_intermediate)
        self.wb = wandb_actor
        self.media_opts = dict(media_opts or {})
        self.wandb_step_base = int(wandb_step_base)
        self.run_baseline = bool(run_baseline)
        self.run_items = bool(run_items)
        if self.outdir is not None:
            ensure_dir(Path(self.outdir))

    def run(self,
            seeds: List[int],
            envs: Dict[str, Dict[str, Any]],
            baseline_phases: List[Dict[str, Any]],
            baseline_evals: List[Dict[str, Any]],
            item_phases_map: Dict[str, List[Dict[str, Any]]],
            evals_map: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        seeds = sorted(map(int, seeds))

        # ===== Normalize baseline total steps to match curriculum total =====
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
                    raise ValueError("baseline_phases is empty; cannot normalize steps.")
                if delta > 0:
                    new_phases[-1]["steps"] = int(new_phases[-1].get("steps", 0)) + delta
                    print(f"[info] baseline steps extended by {delta} to {max_item_total}.")
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
                    print(f"[info] baseline steps trimmed by {-delta} to {max_item_total}.")
                baseline_phases = new_phases

        # ===== Shared collector =====
        collector = Collector.options(max_concurrency=1).remote(
            seeds=seeds,
            keep_intermediate=self.save_intermediate,
            wandb_actor=self.wb,
            step_base=int(self.wandb_step_base),
            flush_interval_s=0.2,
            max_queue=16384,
        )

        # ===== Submit tasks =====
        maxc = self.max_conc or min(len(seeds) * (1 + int(self.run_baseline and self.run_items)), os.cpu_count() or 1)
        pending: List[Tuple[int, str]] = []  # (seed, mode)
        if self.run_baseline:
            pending += [(s, "baseline") for s in seeds]
        if self.run_items:
            pending += [(s, "items") for s in seeds]
        in_flight: Dict[Any, Tuple[int, str]] = {}
        results: List[Dict[str, Any]] = []

        media_root = os.path.join(self.outdir, "media") if self.outdir is not None else None
        if media_root is not None:
            ensure_dir(Path(media_root))

        def submit(seed: int, mode: str, save_media: bool):
            actor = SeedTrainer.options(num_cpus=1).remote(
                seed=seed, envs=envs,
                baseline_phases=baseline_phases, baseline_evals=baseline_evals,
                item_phases_map=item_phases_map, evals_map=evals_map,
                agent_ctor_path=self.agent_ctor_path, agent_kwargs=self.agent_kwargs,
                eval_every=self.eval_every, n_eval_episodes=self.n_eval,
                collector=collector if self.save_intermediate else None,
                collect_intermediate=self.save_intermediate,
                media_root=(media_root if save_media else None),
                media_opts=self.media_opts,
                mode=mode,
            )
            fut = actor.run.remote()
            in_flight[fut] = (seed, mode)

        # Prime queue
        while pending and len(in_flight) < maxc:
            seed, mode = pending.pop(0)
            print(f"[seed {seed}] submit ({mode}).")
            save_media = (len(in_flight) == 0)
            submit(seed, mode, save_media)

        # Drain
        while in_flight:
            done, _ = ray.wait(list(in_flight.keys()), timeout=None, num_returns=1)
            for fut in done:
                seed, mode = in_flight.pop(fut)
                res = ray.get(fut)
                results.append(res)
                print(f"[seed {seed}] done ({mode}).")
                if pending and len(in_flight) < maxc:
                    seed2, mode2 = pending.pop(0)
                    print(f"[seed {seed2}] submit ({mode2}).")
                    submit(seed2, mode2, save_media=False)

        # ===== Merge baseline/items per seed =====
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
                    assert lb not in tgt["items"], f"duplicate items[{lb}] for seed {sid}"
                    tgt["items"][lb] = dd

        if self.run_baseline:
            missing = [s for s, v in by_seed.items() if v["baseline"] is None]
            assert not missing, f"baseline missing for seeds: {missing}"
        if self.run_items:
            missing = [s for s, v in by_seed.items() if not v["items"]]
            assert not missing, f"items missing for seeds: {missing}"

        merged_results: List[Dict[str, Any]] = [by_seed[s] for s in sorted(by_seed.keys())]

        # ===== Optional timeline dump =====
        if self.outdir is not None and self.save_intermediate:
            try:
                tl = ray.get(collector.timeline.remote())
                with open(os.path.join(self.outdir, "online_timeline.json"), "w") as f:
                    json.dump(tl, f)
            except Exception as e:
                print(f"[driver] dump timeline failed: {e}")

        # ===== Aggregate on merged results =====
        summary = self._aggregate(merged_results, item_phases_map)

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

        # ===== Upload videos =====
        if self.wb is not None and media_root is not None:
            to_log = []

            def _enqueue(seed_id: int, label: str, media_map: Dict[str, Optional[str]]):
                if not isinstance(media_map, dict):
                    return
                for name, path in media_map.items():
                    if path:
                        to_log.append((f"media/seed_{seed_id}/{label}/{name}", path))

            def _enqueue_boundary(seed_id: int, label: str, boundary_map: Dict[str, Any]):
                if not isinstance(boundary_map, dict):
                    return
                for phase_key, per_eval in boundary_map.items():
                    if not isinstance(per_eval, dict):
                        continue
                    for eval_name, rec in per_eval.items():
                        p = (rec or {}).get("media_path", None)
                        if p:
                            to_log.append((f"media/seed_{seed_id}/{label}/{eval_name}_{phase_key}", p))

            for r in merged_results:
                sd = int(r.get("seed", 0))
                base = r.get("baseline", None)
                if base and self.run_baseline:
                    _enqueue(sd, "baseline", base.get("media", {}) or {})
                    _enqueue_boundary(sd, "baseline", base.get("boundary", {}) or {})
                items = r.get("items", {}) or {}
                for lb, d in items.items():
                    _enqueue(sd, str(lb), (d or {}).get("media", {}) or {})
                    _enqueue_boundary(sd, str(lb), (d or {}).get("boundary", {}) or {})

            fps = int(self.media_opts.get("fps", 8))
            for key, path in to_log:
                if not (path and os.path.isfile(path)):
                    print(f"[W&B] skip missing video: {path}")
                    continue
                try:
                    fmt = "gif" if str(path).lower().endswith(".gif") else None
                    self.wb.log_video.remote(key, path, fps=fps, fmt=fmt)
                except Exception as e:
                    print(f"[W&B] schedule video upload failed for {key}: {e}")

        # ===== CSV & plots =====
        if self.outdir is not None:
            steps_base = None
            base_curves = None

            if self.run_baseline and ("baseline" in summary and summary["baseline"]):
                base_key = next(iter(summary["baseline"].keys()))
                steps_base = np.asarray(summary["steps"], dtype=int)
                base = summary["baseline"][base_key]
                base_curves = {
                    "greedy_mean": np.asarray(base["greedy_mean"], dtype=float),
                    "greedy_std": np.asarray(base["greedy_std"], dtype=float),
                    "train_mean": np.asarray(base["train_mean"], dtype=float),
                    "train_std": np.asarray(base["train_std"], dtype=float),
                }

            for lb, dd in summary["items"].items():
                item_dir = os.path.join(self.outdir, str(lb))
                ensure_dir(Path(item_dir))

                phs = item_phases_map.get(lb, [])
                item_boundaries: List[int] = []
                acc = 0
                for ph in phs[:-1]:
                    acc += int(ph.get("steps", 0))
                    item_boundaries.append(acc)
                boundaries_str = "-".join(str(b) for b in item_boundaries) if item_boundaries else "none"

                if "Target" in dd:
                    tgt = dd["Target"]
                else:
                    tgt = dd[next(iter(dd.keys()))]

                steps_tgt = np.asarray(
                    tgt.get("steps", steps_base if steps_base is not None else tgt["steps"]),
                    dtype=int
                )

                tgt_g_mean = np.asarray(tgt["greedy_mean"], dtype=float)
                tgt_g_std = np.asarray(tgt["greedy_std"], dtype=float)
                tgt_t_mean = np.asarray(tgt["train_mean"], dtype=float)
                tgt_t_std = np.asarray(tgt["train_std"], dtype=float)

                save_csv(os.path.join(item_dir, f"curriculum_eval_target_phase_{boundaries_str}.csv"),
                         steps_tgt, tgt_g_mean, tgt_g_std, header="greedy")
                save_csv(os.path.join(item_dir, f"curriculum_eval_target_train_phase_{boundaries_str}.csv"),
                         steps_tgt, tgt_t_mean, tgt_t_std, header="train")

                src_name = next((k for k in dd.keys() if k.lower().startswith("source")), None)
                if src_name is not None:
                    src = dd[src_name]
                    steps_src = np.asarray(src.get("steps", steps_tgt), dtype=int)
                    src_g_mean = np.asarray(src["greedy_mean"], dtype=float)
                    src_g_std = np.asarray(src["greedy_std"], dtype=float)
                    src_t_mean = np.asarray(src["train_mean"], dtype=float)
                    src_t_std = np.asarray(src["train_std"], dtype=float)
                    save_csv(os.path.join(item_dir, f"curriculum_eval_source_phase_{boundaries_str}.csv"),
                             steps_src, src_g_mean, src_g_std, header="greedy")
                    save_csv(os.path.join(item_dir, f"curriculum_eval_source_train_phase_{boundaries_str}.csv"),
                             steps_src, src_t_mean, src_t_std, header="train")

                if self.run_baseline and base_curves is not None and steps_base is not None:
                    X = np.union1d(steps_base, steps_tgt)

                    def _interp(x_old, mean, std, x_new):
                        return (np.interp(x_new, x_old, mean),
                                np.interp(x_new, x_old, std))

                    base_g_mean_i, base_g_std_i = _interp(steps_base, base_curves["greedy_mean"],
                                                          base_curves["greedy_std"], X)
                    base_t_mean_i, base_t_std_i = _interp(steps_base, base_curves["train_mean"],
                                                          base_curves["train_std"], X)
                    tgt_g_mean_i, tgt_g_std_i = _interp(steps_tgt, tgt_g_mean, tgt_g_std, X)
                    tgt_t_mean_i, tgt_t_std_i = _interp(steps_tgt, tgt_t_mean, tgt_t_std, X)

                    curves_source_i = None
                    if src_name is not None:
                        src = dd[src_name]
                        steps_src = np.asarray(src.get("steps", X), dtype=int)
                        src_g_mean = np.asarray(src["greedy_mean"], dtype=float)
                        src_g_std = np.asarray(src["greedy_std"], dtype=float)
                        src_t_mean = np.asarray(src["train_mean"], dtype=float)
                        src_t_std = np.asarray(src["train_std"], dtype=float)
                        src_g_mean_i, src_g_std_i = _interp(steps_src, src_g_mean, src_g_std, X)
                        src_t_mean_i, src_t_std_i = _interp(steps_src, src_t_mean, src_t_std, X)
                        curves_source_i = {
                            "greedy_mean": src_g_mean_i, "greedy_std": src_g_std_i,
                            "train_mean": src_t_mean_i, "train_std": src_t_std_i,
                        }

                    png_path = os.path.join(item_dir, f"pairwise_{lb}_phase_{boundaries_str}.png")
                    plot_pairwise(
                        out_png_path=png_path,
                        checkpoints=X,
                        phase_boundaries=item_boundaries,
                        title_prefix=f"Pairwise for '{lb}'",
                        baseline={
                            "greedy_mean": base_g_mean_i, "greedy_std": base_g_std_i,
                            "train_mean": base_t_mean_i, "train_std": base_t_std_i,
                        },
                        curves_target={
                            "greedy_mean": tgt_g_mean_i, "greedy_std": tgt_g_std_i,
                            "train_mean": tgt_t_mean_i, "train_std": tgt_t_std_i,
                        },
                        curves_source=curves_source_i,
                    )

                    if self.wb is not None:
                        self.wb.log_image.remote(
                            key=f"images/pairwise_{lb}_phase_{boundaries_str}",
                            path=png_path,
                            caption=f"pairwise {lb} (phase boundaries: {boundaries_str})"
                        )

        return summary

    # --- offline aggregation (simple, readable) ---
    def _aggregate(self, per_seed: List[Dict[str, Any]],
                   item_phases_map: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        assert per_seed, "no results"
        labels = sorted(item_phases_map.keys())

        def _stack(ll: List[List[float]]) -> Tuple[List[float], List[float]]:
            arr = np.asarray([np.asarray(x, float) for x in ll])
            return arr.mean(0).tolist(), arr.std(0).tolist()

        # baseline: choose the first eval name as primary
        base_names = list(per_seed[0]["baseline"]["steps"].keys())
        assert base_names, "no baseline evals"
        base_primary = base_names[0]
        ref_steps = per_seed[0]["baseline"]["steps"][base_primary]
        for r in per_seed:
            assert r["baseline"]["steps"][base_primary] == ref_steps, "baseline steps mismatch"

        g_mean, g_std = _stack([r["baseline"][base_primary]["greedy"] for r in per_seed])
        t_mean, t_std = _stack([r["baseline"][base_primary]["train"]  for r in per_seed])

        out = {
            "steps": ref_steps,
            "baseline": {
                base_primary: {
                    "greedy_mean": g_mean, "greedy_std": g_std,
                    "train_mean":  t_mean, "train_std":  t_std,
                }
            },
            "items": {}
        }

        for lb in labels:
            eval_names = [k for k in per_seed[0]["items"][lb].keys() if k not in ("steps", "boundary", "media")]
            steps_ref = per_seed[0]["items"][lb]["steps"]
            for r in per_seed:
                for nm in eval_names:
                    assert r["items"][lb]["steps"][nm] == steps_ref[nm], f"steps mismatch for {lb}/{nm}"
            agg = {}
            for nm in eval_names:
                gm, gs = _stack([r["items"][lb][nm]["greedy"] for r in per_seed])
                tm, ts = _stack([r["items"][lb][nm]["train"]  for r in per_seed])
                agg[nm] = {"steps": steps_ref[nm], "greedy_mean": gm, "greedy_std": gs,
                           "train_mean": tm, "train_std": ts}
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
    output_dir: Optional[str] = None,            # give default or place before wandb_step_base
    max_concurrency: Optional[int] = None,
    save_intermediate: bool = True,
    wandb_actor: Optional[ActorHandle] = None,
    media_opts: Optional[Dict[str, Any]] = None,
    wandb_step_base: int = 0,                    # put this at the end
) -> Dict[str, Any]:
    """
    - Pure dicts for phases/evals/envs.
    - Local writes happen iff output_dir is not None.
    - W&B uploads happen iff you pass a WandbWriter actor (independent of local writes).
    """
    trainer = RayCurriculumTrainer(
        agent_ctor_path=agent_ctor_path,
        agent_kwargs=agent_kwargs,
        eval_every=eval_every,
        n_eval_episodes=n_eval_episodes,
        output_dir=output_dir,
        max_concurrency=max_concurrency,
        save_intermediate=save_intermediate,
        wandb_actor=wandb_actor,
        media_opts=media_opts,
        wandb_step_base=wandb_step_base,         # <<< IMPORTANT: forward the base here
    )
    return trainer.run(
        seeds=seeds,
        envs=envs,
        baseline_phases=baseline_phases,
        baseline_evals=baseline_evals,
        item_phases_map=item_phases_map,
        evals_map=evals_map,
    )
