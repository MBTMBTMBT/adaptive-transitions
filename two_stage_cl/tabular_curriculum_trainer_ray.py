# generic_curriculum_trainer.py — Ray-based, plain dicts, local logging independent of W&B.
# English only.

from __future__ import annotations
import os, time, json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from ray.actor import ActorHandle

from experiment_utils.utils import _import, ensure_dir
from experiment_utils.save_media import save_policy_media
from experiment_utils.env_factories import make_env


@ray.remote
class Collector:
    def __init__(self,
                 seeds: List[int],
                 keep_intermediate: bool = True,
                 wandb_actor: Optional[ActorHandle] = None):
        """
        - keep_intermediate=False: no timeline snapshotting.
        - keep_intermediate=True: keep per-seed timeline (driver may dump locally if output_dir exists).
        - wandb_actor: external WandbWriter actor handle; if provided, Collector pushes online scalars to it.
        """
        self.seeds = list(map(int, seeds))
        self.keep = bool(keep_intermediate)
        self._bucket: Dict[Tuple[str, str, int], Dict[int, Tuple[float, float]]] = {}
        self._timeline: Dict[int, List[Dict[str, Any]]] = {s: [] for s in self.seeds}
        self._wb = wandb_actor

    def report(self, e: Dict[str, Any]):
        """
        e = {seed, label, eval_name, step, greedy, train, wall_time}
        """
        key = (str(e["label"]), str(e["eval_name"]), int(e["step"]))
        b = self._bucket.setdefault(key, {})
        b[int(e["seed"])] = (float(e["greedy"]), float(e["train"]))
        if self.keep:
            self._timeline[int(e["seed"])].append(e)

        # Aggregate when all seeds reached the same (label, eval, step)
        if len(b) == len(self.seeds):
            g = np.array([v[0] for v in b.values()], dtype=np.float64)
            t = np.array([v[1] for v in b.values()], dtype=np.float64)
            row = {
                "label": key[0], "eval_name": key[1], "step": key[2],
                "greedy_mean": float(g.mean()), "greedy_std": float(g.std()),
                "train_mean":  float(t.mean()), "train_std":  float(t.std()),
            }
            if self._wb is not None:
                prefix = f"online/{row['label']}/{row['eval_name']}"
                try:
                    self._wb.log.remote({
                        f"{prefix}/greedy_mean": row["greedy_mean"],
                        f"{prefix}/greedy_std":  row["greedy_std"],
                        f"{prefix}/train_mean":  row["train_mean"],
                        f"{prefix}/train_std":   row["train_std"],
                    }, step=int(row["step"]))
                except Exception as ex:
                    print(f"[Collector] wandb.log failed: {ex}")

            # free memory
            try:
                del self._bucket[key]
            except Exception:
                pass

    def timeline(self) -> Dict[int, List[Dict[str, Any]]]:
        return self._timeline if self.keep else {}


class PeriodicEvalCallbackSB3(BaseCallback):
    """
    SB3 BaseCallback that evaluates at step 0, every eval_every, and training end.
    It appends results into the shared logs and (optionally) reports to Collector.
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
        super().__init__()
        self.eval_env = eval_env
        self.eval_every = int(eval_every)
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
                    "step": step,
                    "greedy": float(g_mean),
                    "train": float(t_mean),
                    "wall_time": time.time(),
                })
            except Exception as ex:
                print(f"[Collector] report failed: {ex}")

        self._last_eval_step = step
        self._eval_count += 1

    def _on_training_start(self) -> None:
        self._do_eval(tag="start")

    def _on_step(self) -> bool:
        s = int(self.model.num_timesteps)
        if s > 0 and s % self.eval_every == 0 and self._last_eval_step != s:
            self._do_eval(tag="periodic")
        return True

    def _on_training_end(self) -> None:
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
                 media_opts: Optional[Dict[str, Any]] = None):
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

        # media: always on if media_root is provided (no extra switches)
        self.media_root = media_root  # None => do not record anything
        self.media_on = media_root is not None

        # minimal, flexible options (no booleans to toggle features)
        mo = dict(media_opts or {})
        self.mo_episodes = int(mo.get("episodes", 3))
        self.mo_max_steps = int(mo.get("max_steps", 200))
        self.mo_fps = int(mo.get("fps", 8))
        self.mo_fmt = str(mo.get("fmt", "gif"))
        self.mo_deterministic = bool(mo.get("deterministic", True))
        self.mo_start_seed_base = int(mo.get("start_seed_base", 10_000))
        self.mo_target_size = mo.get("target_size", None)      # e.g. (256,256)
        self.mo_bgr_to_rgb = bool(mo.get("bgr_to_rgb", False)) # set True if renderer outputs BGR

    # --- eval helper ---
    def _eval_once(self, model, env, det: bool, seed_base: int) -> float:
        env.reset(seed=seed_base)
        m, _ = evaluate_policy(model, env, self.n_eval, deterministic=det, render=False, warn=False)
        return float(m)

    # --- media helpers ---
    def _save_media(self, label: str, eval_name: str, env_key: str, suffix: str, agent) -> Optional[str]:
        if not self.media_on:
            return None
        env = make_env(self.envs[env_key], seed=self.seed + self.mo_start_seed_base)
        # subdir: <media_root>/seed_<id>/<label>/
        subdir = os.path.join(self.media_root, f"seed_{self.seed}", label)
        ensure_dir(subdir)
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
        # train env + agent
        train_env = DummyVecEnv([lambda: make_env(self.envs[phases[0]["env"]], seed=self.seed)])
        agent_cls = _import(self.agent_ctor_path)
        agent = agent_cls(env=train_env, seed=self.seed, **self.agent_kwargs)

        # fixed eval envs (for periodic evals inside callback)
        eval_envs = {e["name"]: make_env(self.envs[e["env"]], seed=12345) for e in evals}
        steps_log = {nm: [] for nm in eval_envs}
        greedy_log = {nm: [] for nm in eval_envs}
        trainp_log = {nm: [] for nm in eval_envs}

        boundary_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for i, ph in enumerate(phases):
            # build a fresh CallbackList for this phase (so step-0 eval happens per phase)
            cbs = []
            for e in evals:
                nm = e["name"]
                cb = PeriodicEvalCallbackSB3(
                    eval_env=eval_envs[nm],
                    eval_every=self.eval_every,
                    n_eval_episodes=self.n_eval,
                    steps_log=steps_log[nm],
                    greedy_log=greedy_log[nm],
                    train_log=trainp_log[nm],
                    seed_base=int(e.get("seed_base", self.mo_start_seed_base)),
                    label=label,
                    eval_name=nm,
                    collector=self.collector if self.collect_on else None,
                    seed_id=self.seed,
                )
                cbs.append(cb)
            cb_list = CallbackList(cbs)

            # single learn call for this phase (NO chunking)
            phase_steps = int(ph["steps"])
            agent.learn(total_timesteps=phase_steps, reset_num_timesteps=False, callback=cb_list, progress_bar=False)

            # ---- phase boundary evals (ALWAYS) right AFTER this phase ----
            bkey = f"phase_{i}"
            boundary_cache.setdefault(bkey, {})
            for e in evals:
                name = e["name"]
                test_env = make_env(self.envs[e["env"]], seed=self.seed + 1000 + i)
                try:
                    test_env.reset(seed=self.seed + 1000 + i)
                    g_mean, _ = evaluate_policy(agent, test_env, self.n_eval, deterministic=True, render=False,
                                                warn=False)
                    test_env.reset(seed=self.seed + 1001 + i)
                    t_mean, _ = evaluate_policy(agent, test_env, self.n_eval, deterministic=False, render=False,
                                                warn=False)
                finally:
                    try:
                        test_env.close()
                    except Exception:
                        pass

                boundary_cache[bkey].setdefault(name, {})
                boundary_cache[bkey][name]["greedy"] = float(g_mean)
                boundary_cache[bkey][name]["train"] = float(t_mean)

            # boundary media (ALWAYS if media_root is set)
            if self.media_on:
                media_map = self._boundary_media_all(label, evals, agent, phase_idx=i)
                for nm, path in media_map.items():
                    boundary_cache[bkey][nm]["media_path"] = path

            # switch to next phase env
            if i < len(phases) - 1:
                next_env = DummyVecEnv([lambda: make_env(self.envs[phases[i + 1]["env"]], seed=self.seed)])
                agent.set_env(next_env)

        # final media (ALWAYS if media_root is set)
        media_paths = self._final_media_all(label, evals, agent) if self.media_on else {}

        # pack
        out = {"steps": steps_log, "boundary": boundary_cache, "media": media_paths}
        for nm in eval_envs:
            out[nm] = {"greedy": greedy_log[nm], "train": trainp_log[nm]}
        for v in eval_envs.values():
            try:
                v.close()
            except Exception:
                pass
        return out

    def run(self) -> Dict[str, Any]:
        res = {"seed": self.seed, "baseline": self._run_schedule("baseline", self.base_ph, self.base_evals),
               "items": {}}
        # baseline uses baseline_evals directly (often [{"name": "Target", "env": "..."}])
        for label, phases in self.items_ph.items():
            res["items"][label] = self._run_schedule(str(label), phases, self.evals_map[label])
        return res


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
                 max_concurrency: Optional[int] = None,
                 save_intermediate: bool = True,
                 wandb_actor: Optional[ActorHandle] = None,
                 media_opts: Optional[Dict[str, Any]] = None):
        """
        - output_dir governs ALL local writes; None => write nothing locally (including media).
        - wandb_actor governs ALL W&B uploads; None => upload nothing to W&B.
        - save_intermediate controls whether the Collector keeps the online timeline (dumped only if output_dir exists).
        - media_opts provides lightweight numeric/format knobs for media recording; if omitted, defaults are used.
        """
        self.agent_ctor_path = agent_ctor_path
        self.agent_kwargs = dict(agent_kwargs)
        self.eval_every = int(eval_every)
        self.n_eval = int(n_eval_episodes)
        self.outdir = output_dir  # may be None
        self.max_conc = max_concurrency
        self.save_intermediate = bool(save_intermediate)
        self.wb = wandb_actor  # may be None
        self.media_opts = dict(media_opts or {})
        if self.outdir is not None:
            ensure_dir(self.outdir)

    def run(self,
            seeds: List[int],
            envs: Dict[str, Dict[str, Any]],
            baseline_phases: List[Dict[str, Any]],
            baseline_evals: List[Dict[str, Any]],
            item_phases_map: Dict[str, List[Dict[str, Any]]],
            evals_map: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        collector = Collector.remote(
            seeds=list(map(int, seeds)),
            keep_intermediate=self.save_intermediate,
            wandb_actor=self.wb,
        )

        seeds = list(map(int, seeds))
        maxc = self.max_conc or min(len(seeds), os.cpu_count() or 1)
        pending = list(seeds)
        in_flight: Dict[Any, int] = {}
        results: List[Dict[str, Any]] = []

        # media_root is simply under output_dir (if any). No extra switches.
        media_root = os.path.join(self.outdir, "media") if self.outdir is not None else None
        if media_root is not None:
            ensure_dir(media_root)

        def submit():
            s = pending.pop(0)
            actor = SeedTrainer.options(num_cpus=1).remote(
                seed=s, envs=envs,
                baseline_phases=baseline_phases, baseline_evals=baseline_evals,
                item_phases_map=item_phases_map, evals_map=evals_map,
                agent_ctor_path=self.agent_ctor_path, agent_kwargs=self.agent_kwargs,
                eval_every=self.eval_every, n_eval_episodes=self.n_eval,
                collector=collector if self.save_intermediate else None,
                collect_intermediate=self.save_intermediate,
                media_root=media_root,
                media_opts=self.media_opts,
            )
            fut = actor.run.remote()
            in_flight[fut] = s

        while pending and len(in_flight) < maxc:
            submit()

        while in_flight:
            done, _ = ray.wait(list(in_flight.keys()), timeout=None, num_returns=1)
            for fut in done:
                s = in_flight.pop(fut)
                res = ray.get(fut)
                results.append(res)
                print(f"[seed {s}] done.")
                if pending and len(in_flight) < maxc:
                    submit()

        # optional local dump of Collector timeline (only if output_dir exists)
        if self.outdir is not None and self.save_intermediate:
            try:
                tl = ray.get(collector.timeline.remote())
                with open(os.path.join(self.outdir, "online_timeline.json"), "w") as f:
                    json.dump(tl, f)
            except Exception as e:
                print(f"[driver] dump timeline failed: {e}")

        # aggregate
        summary = self._aggregate(results, item_phases_map)

        # local save: per-seed raw + final summary (only if output_dir exists)
        if self.outdir is not None:
            ensure_dir(self.outdir)
            with open(os.path.join(self.outdir, "final_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            seeds_dir = os.path.join(self.outdir, "seeds")
            ensure_dir(seeds_dir)
            for r in results:
                sid = int(r.get("seed", -1))
                with open(os.path.join(seeds_dir, f"seed_{sid}.json"), "w") as f:
                    json.dump(r, f)

        # W&B: upload ALL media if a WandbWriter is provided (independent of local writes)
        if self.wb is not None and media_root is not None:
            to_log = []

            def _enqueue(seed_id: int, label: str, media_map: Dict[str, Optional[str]]):
                if not isinstance(media_map, dict): return
                for name, path in media_map.items():
                    if path:
                        to_log.append((f"media/seed_{seed_id}/{label}/{name}", path))

            def _enqueue_boundary(seed_id: int, label: str, boundary_map: Dict[str, Any]):
                if not isinstance(boundary_map, dict): return
                for phase_key, per_eval in boundary_map.items():
                    if not isinstance(per_eval, dict): continue
                    for eval_name, rec in per_eval.items():
                        p = (rec or {}).get("media_path", None)
                        if p:
                            to_log.append((f"media/seed_{seed_id}/{label}/{eval_name}_{phase_key}", p))

            for r in results:
                sd = int(r.get("seed", 0))
                base = r.get("baseline", {}) or {}
                _enqueue(sd, "baseline", base.get("media", {}) or {})
                _enqueue_boundary(sd, "baseline", base.get("boundary", {}) or {})
                items = r.get("items", {}) or {}
                for lb, d in items.items():
                    _enqueue(sd, str(lb), (d or {}).get("media", {}) or {})
                    _enqueue_boundary(sd, str(lb), (d or {}).get("boundary", {}) or {})

            fps = int(self.media_opts.get("fps", 8))
            for key, path in to_log:
                try:
                    fmt = "gif" if str(path).lower().endswith(".gif") else None
                    self.wb.log_video.remote(key, path, fps=fps, fmt=fmt)
                except Exception as e:
                    print(f"[W&B] schedule video upload failed for {key}: {e}")

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
    output_dir: Optional[str],
    max_concurrency: Optional[int] = None,
    save_intermediate: bool = True,
    wandb_actor: Optional[ActorHandle] = None,
    media_opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    - Pure dicts for phases/evals/envs.
    - Local writes happen iff output_dir is not None.
    - W&B uploads happen iff you pass a WandbWriter actor (independent of local writes).
    - Example WandbWriter creation:
        wb = WandbWriter.remote({'project': 'my-proj', 'name': 'exp-1', 'config': {...}})
      Then pass wandb_actor=wb.
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
    )
    return trainer.run(
        seeds=seeds,
        envs=envs,
        baseline_phases=baseline_phases,
        baseline_evals=baseline_evals,
        item_phases_map=item_phases_map,
        evals_map=evals_map,
    )
