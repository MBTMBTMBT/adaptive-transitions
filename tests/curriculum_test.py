# test_q_learning_frozenlake_json_envs_two_phase_cb.py
# English comments only. Two-phase training using PeriodicEvalCallback with pairwise comparisons.
# Parallelized across seeds via process pool (Linux / Python 3.10).

import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from simple_agents.tabular_q_agent import TabularQAgent
from customised_toy_text_envs.customised_frozenlake import CustomisedFrozenLakeEnv
from networkx_env.networkx_env import NetworkXMDPEnvironment
from mdp_network.mdp_network import MDPNetwork
from simple_agents.apis import BaseCallback  # required by the callback


# ========= Testing callback: evaluate at start, periodic, end =========
class PeriodicEvalCallback(BaseCallback):
    """
    Evaluate the model at step 0, every eval_every steps, and at the end.
    Records greedy (det=True) and training-policy (det=False) scores into provided lists.
    Does not interrupt training (always returns True).
    """
    def __init__(self, eval_env, eval_every: int, n_eval_episodes: int,
                 greedy_scores_list, train_scores_list,
                 eval_seed_base: int = 10000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_every = int(eval_every)
        self.n_eval_episodes = int(n_eval_episodes)
        self.greedy_scores_list = greedy_scores_list
        self.train_scores_list = train_scores_list
        self.eval_seed_base = int(eval_seed_base)
        self._last_eval_step = -1
        self._eval_count = 0

    def _do_eval(self, tag: str):
        # control RNG for reproducibility
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
        self._last_eval_step = self.model.num_timesteps
        self._eval_count += 1
        if self.verbose:
            print(f"[Eval:{tag}] step={self.model.num_timesteps}  "
                  f"Greedy={mean_greedy:.3f}  TrainPol={mean_train:.3f}")

    def _on_training_start(self):
        self._do_eval(tag="start")

    def _on_step(self) -> bool:
        if self.model.num_timesteps > 0 and self.model.num_timesteps % self.eval_every == 0:
            if self._last_eval_step != self.model.num_timesteps:
                self._do_eval(tag="periodic")
        return True

    def _on_training_end(self):
        if self._last_eval_step != self.model.num_timesteps:
            self._do_eval(tag="end")


# =========================
# Config
# =========================
OUTPUT_DIR = "./outputs/ga_cl"
JSON_DIR = "./outputs/ga_test"   # folder containing multiple *.json MDPs

# Two-phase schedule (no cycles)
STEPS_JSON_PHASE = 25_000        # Phase A: train on JSON-backed env
STEPS_BASE_PHASE = 125_000       # Phase B: continue training on baseline env
EVAL_EVERY = 2_500               # desired evaluation cadence (actual cadence may be coarser)
N_EVAL_EPISODES = 100            # episodes per evaluation point

# Agent hyperparams
LEARNING_RATE = 0.1
GAMMA = 0.99
POLICY_MIX = (0.9, 0.0, 0.1)
TEMPERATURE = 0.01
ITE_TOL = 1e-2

# Env specifics
FROZENLAKE_MAP = "8x8"
FROZENLAKE_IS_SLIPPERY = True
FROZENLAKE_MAX_STEPS = 500

# Seeds
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7,]

# =========================
# Naming (short & consistent)
# =========================
# We call the baseline env the Target environment.
TARGET_NAME = "Target"
LINE_TARGET_BASELINE   = f"{TARGET_NAME}-only training (baseline)"        # train+eval on Target
LINE_CURR_EVAL_TARGET  = f"Curriculum → {TARGET_NAME} (primary)"          # Phase A on Source, Phase B on Target, eval on Target
LINE_CURR_EVAL_SOURCE  = "Curriculum (eval on Source-A)"                  # same training, eval on Source-A

# Plot style knobs (kept modest; no explicit colors)
PHASE_LINE_LS = "--"
PHASE_LINE_ALPHA = 0.7
PRIMARY_LW = 2.2
OTHER_LW = 1.5


# =========================
# Helpers
# =========================
def make_baseline_env(seed: int | None = None):
    """Create the native FrozenLake environment (baseline=Target)."""
    env = CustomisedFrozenLakeEnv(
        render_mode=None,
        map_name=FROZENLAKE_MAP,
        is_slippery=FROZENLAKE_IS_SLIPPERY,
        networkx_env=None,
    )
    env = TimeLimit(env, max_episode_steps=FROZENLAKE_MAX_STEPS)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_nx_env_from_mdp(mdp: MDPNetwork, seed: int | None = None):
    """Create a NetworkX-backed FrozenLake environment using the given MDP (Source-A)."""
    nx_backend = NetworkXMDPEnvironment(mdp_network=mdp, render_mode=None, seed=seed)
    env = TimeLimit(nx_backend, max_episode_steps=FROZENLAKE_MAX_STEPS)
    if seed is not None:
        env.reset(seed=seed)
    return env


def build_checkpoints_from_curve_len(steps_a: int, steps_b: int, n_points: int) -> np.ndarray:
    """
    Infer checkpoints from the ACTUAL number of callback evaluations:
      N = 1 (start) + k_a + 1 (boundary duplicate) + k_b
    We distribute k_a points uniformly in Phase A, and k_b points uniformly in Phase B.
    This matches the callback's emitted count even if effective cadence != EVAL_EVERY.
    """
    assert n_points >= 2, "Curve must contain at least start and one more eval."
    total = steps_a + steps_b
    k_total = n_points - 2  # exclude start and the boundary duplicate
    if k_total <= 0:
        return np.array([0, steps_a], dtype=int)

    # split proportionally by phase duration
    k_a = int(round(k_total * (steps_a / total)))
    k_b = k_total - k_a

    xs: List[int] = [0]

    # Phase A: k_a points uniformly (last hits steps_a)
    if k_a > 0:
        for i in range(1, k_a + 1):
            xs.append(int(round(i * steps_a / k_a)))
    else:
        xs.append(steps_a)

    # Boundary duplicate at steps_a
    xs.append(steps_a)

    # Phase B: k_b points uniformly (last hits total)
    if k_b > 0:
        for i in range(1, k_b + 1):
            xs.append(steps_a + int(round(i * steps_b / k_b)))
    else:
        xs.append(steps_a + steps_b)

    return np.array(xs, dtype=int)


def mean_std(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.stack(curves, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


# =========================
# One-seed worker (runs in subprocess)
# =========================
def _run_one_seed(seed: int, json_files: List[str]) -> Dict:
    """
    Runs the full two-phase schedule for a single seed inside a subprocess.
    Returns numpy arrays and dicts only (pickle-safe).
    """
    json_info = [(Path(p).stem, p) for p in json_files]

    # Each process builds its own eval env (no sharing across processes)
    baseline_eval_env = make_baseline_env(seed=12345)

    # ---- Target-only (baseline) ----
    base_train_env = DummyVecEnv([lambda: make_baseline_env(seed=seed)])
    base_agent = TabularQAgent(
        env=base_train_env,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        policy_mix=POLICY_MIX,
        temperature=TEMPERATURE,
        seed=seed,
        verbose=0,
    )

    base_greedy_curve: List[float] = []
    base_train_curve: List[float] = []
    base_cb = PeriodicEvalCallback(
        eval_env=baseline_eval_env,
        eval_every=EVAL_EVERY,
        n_eval_episodes=N_EVAL_EPISODES,
        greedy_scores_list=base_greedy_curve,
        train_scores_list=base_train_curve,
        eval_seed_base=10_000 + seed,
        verbose=0,
    )

    base_agent.learn(total_timesteps=STEPS_JSON_PHASE, reset_num_timesteps=False,
                     progress_bar=False, callback=base_cb)
    base_agent.learn(total_timesteps=STEPS_BASE_PHASE, reset_num_timesteps=False,
                     progress_bar=False, callback=base_cb)

    seed_checkpoints = build_checkpoints_from_curve_len(
        STEPS_JSON_PHASE, STEPS_BASE_PHASE, len(base_greedy_curve)
    )

    # ---- Curriculum results per JSON ----
    mixed_base: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    mixed_phase: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for label, jp in json_info:
        mdp = MDPNetwork(config_path=jp)
        nx_train_env = DummyVecEnv([lambda: make_nx_env_from_mdp(mdp=mdp, seed=seed)])
        base_train_env_for_mixed = DummyVecEnv([lambda: make_baseline_env(seed=seed)])

        # Curriculum @ Target (primary eval)
        agent = TabularQAgent(
            env=base_train_env_for_mixed,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            policy_mix=POLICY_MIX,
            temperature=TEMPERATURE,
            tie_tol=ITE_TOL,
            seed=seed,
            verbose=0,
        )
        g1: List[float] = []; t1: List[float] = []
        cb1 = PeriodicEvalCallback(
            eval_env=baseline_eval_env,
            eval_every=EVAL_EVERY,
            n_eval_episodes=N_EVAL_EPISODES,
            greedy_scores_list=g1,
            train_scores_list=t1,
            eval_seed_base=42 + 1000*seed + (hash(label) % 100),
            verbose=0,
        )
        agent.set_env(nx_train_env)
        agent.learn(total_timesteps=STEPS_JSON_PHASE, reset_num_timesteps=False,
                    progress_bar=False, callback=cb1)
        agent.set_env(base_train_env_for_mixed)
        agent.learn(total_timesteps=STEPS_BASE_PHASE, reset_num_timesteps=False,
                    progress_bar=False, callback=cb1)
        mixed_base[label] = (np.asarray(g1, dtype=float), np.asarray(t1, dtype=float))

        # Curriculum @ Source-A (aux eval)
        agent2 = TabularQAgent(
            env=base_train_env_for_mixed,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            policy_mix=POLICY_MIX,
            temperature=TEMPERATURE,
            tie_tol=ITE_TOL,
            seed=seed,
            verbose=0,
        )
        g2: List[float] = []; t2: List[float] = []
        nx_eval_env = make_nx_env_from_mdp(mdp=mdp, seed=12345)
        cb2 = PeriodicEvalCallback(
            eval_env=nx_eval_env,
            eval_every=EVAL_EVERY,
            n_eval_episodes=N_EVAL_EPISODES,
            greedy_scores_list=g2,
            train_scores_list=t2,
            eval_seed_base=84 + 1000*seed + (hash(label) % 100),
            verbose=0,
        )
        agent2.set_env(nx_train_env)
        agent2.learn(total_timesteps=STEPS_JSON_PHASE, reset_num_timesteps=False,
                     progress_bar=False, callback=cb2)
        agent2.set_env(base_train_env_for_mixed)
        agent2.learn(total_timesteps=STEPS_BASE_PHASE, reset_num_timesteps=False,
                     progress_bar=False, callback=cb2)
        mixed_phase[label] = (np.asarray(g2, dtype=float), np.asarray(t2, dtype=float))

    return {
        "seed": seed,
        "checkpoints": np.asarray(seed_checkpoints, dtype=int),
        "base_greedy": np.asarray(base_greedy_curve, dtype=float),
        "base_train": np.asarray(base_train_curve, dtype=float),
        "mixed_base": mixed_base,   # {label: (greedy_arr, train_arr)}
        "mixed_phase": mixed_phase, # {label: (greedy_arr, train_arr)}
    }


# =========================
# Main
# =========================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover JSON files
    json_paths = sorted(Path(JSON_DIR).glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No *.json found in {JSON_DIR}")

    print(f"Found {len(json_paths)} JSON MDPs:")
    for p in json_paths:
        print(" -", p.name)

    # -------- Parallel loop over seeds --------
    json_files = [str(p) for p in json_paths]
    NUM_WORKERS = min(len(SEEDS), os.cpu_count() or 1)

    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=get_context("spawn")) as ex:
        futures = {ex.submit(_run_one_seed, seed, json_files): seed for seed in SEEDS}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"[Seed {res['seed']}] finished in subprocess.")

    # Use the first seed’s checkpoints as the global x-axis; assert all match
    checkpoints = results[0]["checkpoints"]
    for res in results:
        assert len(res["checkpoints"]) == len(checkpoints), \
            f"Seed {res['seed']} produced different eval count."

    # Prepare containers as before
    baseonly_greedy_all: List[np.ndarray] = []
    baseonly_train_all: List[np.ndarray] = []
    mixed_base_greedy_by_json: Dict[str, List[np.ndarray]] = {Path(p).stem: [] for p in json_paths}
    mixed_base_train_by_json:  Dict[str, List[np.ndarray]] = {Path(p).stem: [] for p in json_paths}
    mixed_phase_greedy_by_json: Dict[str, List[np.ndarray]] = {Path(p).stem: [] for p in json_paths}
    mixed_phase_train_by_json:  Dict[str, List[np.ndarray]] = {Path(p).stem: [] for p in json_paths}

    # Aggregate per-seed outputs into the above containers
    for res in results:
        baseonly_greedy_all.append(res["base_greedy"])
        baseonly_train_all.append(res["base_train"])
        for label in mixed_base_greedy_by_json.keys():
            g_b, t_b = res["mixed_base"][label]
            g_p, t_p = res["mixed_phase"][label]
            mixed_base_greedy_by_json[label].append(g_b)
            mixed_base_train_by_json[label].append(t_b)
            mixed_phase_greedy_by_json[label].append(g_p)
            mixed_phase_train_by_json[label].append(t_p)

    # =========================
    # Aggregate across seeds
    # =========================
    def _agg(dct_lists: Dict[str, List[np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for label, lst in dct_lists.items():
            arr = np.stack(lst, axis=0)
            out[label] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0)}
        return out

    base_greedy_mean, base_greedy_std = mean_std(baseonly_greedy_all)
    base_train_mean, base_train_std = mean_std(baseonly_train_all)

    mixed_base_g = _agg(mixed_base_greedy_by_json)
    mixed_base_t = _agg(mixed_base_train_by_json)
    mixed_phase_g = _agg(mixed_phase_greedy_by_json)
    mixed_phase_t = _agg(mixed_phase_train_by_json)

    # --- Drop the first point of Phase B (duplicate of Phase A end) ---
    dup_idx = np.where(np.diff(checkpoints) == 0)[0]
    if dup_idx.size > 0:
        rm = int(dup_idx[0] + 1)  # remove the second occurrence (start of Phase B)
        # x-axis
        checkpoints = np.delete(checkpoints, rm)
        # baseline aggregates
        base_greedy_mean = np.delete(base_greedy_mean, rm)
        base_greedy_std  = np.delete(base_greedy_std,  rm)
        base_train_mean  = np.delete(base_train_mean,  rm)
        base_train_std   = np.delete(base_train_std,   rm)
        # per-JSON aggregates (both eval contexts)
        for d in (mixed_base_g, mixed_base_t, mixed_phase_g, mixed_phase_t):
            for k in d:
                d[k]["mean"] = np.delete(d[k]["mean"], rm)
                d[k]["std"]  = np.delete(d[k]["std"],  rm)

    # =========================
    # Plot — pairwise per JSON (three lines each) + vertical phase marker
    # =========================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    phase_boundary = STEPS_JSON_PHASE  # x position of the A→B boundary

    for label in sorted(mixed_base_g.keys()):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Titles
        ax1.set_title(f"FrozenLake 8x8 — Pairwise for '{label}' (Greedy)")
        ax2.set_title(f"Pairwise for '{label}' (Training-policy)")

        # Lines (Greedy)
        ax1.plot(checkpoints, base_greedy_mean, label=LINE_TARGET_BASELINE, linewidth=OTHER_LW)
        ax1.fill_between(checkpoints, base_greedy_mean - base_greedy_std, base_greedy_mean + base_greedy_std, alpha=0.2)

        mb_m, mb_s = mixed_base_g[label]["mean"], mixed_base_g[label]["std"]
        mp_m, mp_s = mixed_phase_g[label]["mean"], mixed_phase_g[label]["std"]
        ax1.plot(checkpoints, mb_m, label=LINE_CURR_EVAL_TARGET, linewidth=PRIMARY_LW)
        ax1.fill_between(checkpoints, mb_m - mb_s, mb_m + mb_s, alpha=0.15)
        ax1.plot(checkpoints, mp_m, label=LINE_CURR_EVAL_SOURCE, linewidth=OTHER_LW)
        ax1.fill_between(checkpoints, mp_m - mp_s, mp_m + mp_s, alpha=0.15)

        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel(f"Mean return over {N_EVAL_EPISODES} eps")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Lines (Train-policy)
        ax2.plot(checkpoints, base_train_mean, label=LINE_TARGET_BASELINE, linewidth=OTHER_LW)
        ax2.fill_between(checkpoints, base_train_mean - base_train_std, base_train_mean + base_train_std, alpha=0.2)

        mb_m, mb_s = mixed_base_t[label]["mean"], mixed_base_t[label]["std"]
        mp_m, mp_s = mixed_phase_t[label]["mean"], mixed_phase_t[label]["std"]
        ax2.plot(checkpoints, mb_m, label=LINE_CURR_EVAL_TARGET, linewidth=PRIMARY_LW)
        ax2.fill_between(checkpoints, mb_m - mb_s, mb_m + mb_s, alpha=0.15)
        ax2.plot(checkpoints, mp_m, label=LINE_CURR_EVAL_SOURCE, linewidth=OTHER_LW)
        ax2.fill_between(checkpoints, mp_m - mp_s, mp_m + mp_s, alpha=0.15)

        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel(f"Mean return over {N_EVAL_EPISODES} eps")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # --- Phase boundary marker on BOTH subplots ---
        for ax in (ax1, ax2):
            ax.axvline(phase_boundary, linestyle=PHASE_LINE_LS, alpha=PHASE_LINE_ALPHA)
            ymin, ymax = ax.get_ylim()
            ytxt = ymin + 0.06 * (ymax - ymin)
            ax.text(phase_boundary * 0.5, ytxt, "Phase A (Source)", ha="center", va="bottom", fontsize=9, alpha=0.8)
            ax.text(phase_boundary + (checkpoints[-1] - phase_boundary) * 0.5, ytxt, f"Phase B ({TARGET_NAME})",
                    ha="center", va="bottom", fontsize=9, alpha=0.8)
            ax.text(phase_boundary, ymax, "A → B", ha="right", va="top", rotation=90, fontsize=9, alpha=0.8)

        fig.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, f"pairwise_{label}.png")
        fig.savefig(fig_path, dpi=150)
        print(f"Saved figure -> {fig_path}")

    # =========================
    # Summary at last checkpoint
    # =========================
    print("\n=== Summary at final checkpoint (per JSON) ===")
    for label in sorted(mixed_base_g.keys()):
        print(f"[{label}]")
        print(f"  {LINE_TARGET_BASELINE} — Greedy: {base_greedy_mean[-1]:.3f} ± {base_greedy_std[-1]:.3f} | "
              f"TrainPol: {base_train_mean[-1]:.3f} ± {base_train_std[-1]:.3f}")
        print(f"  {LINE_CURR_EVAL_TARGET} — Greedy: {mixed_base_g[label]['mean'][-1]:.3f} ± {mixed_base_g[label]['std'][-1]:.3f} | "
              f"TrainPol: {mixed_base_t[label]['mean'][-1]:.3f} ± {mixed_base_t[label]['std'][-1]:.3f}")
        print(f"  {LINE_CURR_EVAL_SOURCE} — Greedy: {mixed_phase_g[label]['mean'][-1]:.3f} ± {mixed_phase_g[label]['std'][-1]:.3f} | "
              f"TrainPol: {mixed_phase_t[label]['mean'][-1]:.3f} ± {mixed_phase_t[label]['std'][-1]:.3f}")

    print("\nDone.")
