# test_q_learning_stoch_envs.py
# English comments only. Flat script with a small callback class.

import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit

from customised_minigrid_env.simple_agents.tabular_q_agent import TabularQAgent
from customised_toy_text_envs.customised_taxi import CustomisedTaxiEnv
from customised_toy_text_envs.customised_frozenlake import CustomisedFrozenLakeEnv
from networkx_env.networkx_env import NetworkXMDPEnvironment
from customised_minigrid_env.simple_agents.apis import BaseCallback


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
        # control RNG for reproducibility demo
        self.eval_env.reset(seed=self.eval_seed_base + 2*self._eval_count)
        mean_greedy, std_greedy = evaluate_policy(
            model=self.model, env=self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True, render=False, warn=False
        )
        self.eval_env.reset(seed=self.eval_seed_base + 2*self._eval_count + 1)
        mean_train, std_train = evaluate_policy(
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


# ========= Helpers =========
def mean_std(curves_list):
    arr = np.stack(curves_list, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


if __name__ == '__main__':
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gymnasium.wrappers import TimeLimit

    # runtime imports (avoid touching the header)
    from mdp_network.mdp_tables import q_table_to_policy
    from tests.helpers import record_policy_gif

    output_dir = "./outputs/tabular_q_learning"
    os.makedirs(output_dir, exist_ok=True)

    total_timesteps = 250_000
    eval_every = 10_000
    n_eval_episodes = 100

    learning_rate = 0.1
    gamma = 0.99
    policy_mix = (0.5, 0.4, 0.1)
    temperature = 1.0

    # action names for GIF overlays
    taxi_action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
    fl_action_names = ["Left", "Down", "Right", "Up"]

    # seeds
    seeds = [0, 1, 2]

    # --- Stochastic Taxi (rainy) — build MDP + a base backend with TimeLimit
    print("\n=== Build stochastic Taxi (rainy) MDP and NetworkX backend ===")
    taxi_src_env = CustomisedTaxiEnv(render_mode=None, is_rainy=True)  # rainy confirmed
    taxi_src_env.reset()
    taxi_mdp = taxi_src_env.get_mdp_network()
    taxi_nx_env_base = TimeLimit(
        NetworkXMDPEnvironment(mdp_network=taxi_mdp, render_mode=None, seed=None),
        max_episode_steps=200  # Taxi-v3 default
    )
    print(f"Taxi rainy MDP: |S|={len(taxi_mdp.states)}, |A|={taxi_mdp.num_actions}")

    # --- FrozenLake 8x8 (slippery) — build MDP + a base backend with TimeLimit
    print("\n=== Build stochastic FrozenLake (8x8, slippery) MDP and NetworkX backend ===")
    fl_src_env = CustomisedFrozenLakeEnv(render_mode=None, map_name="8x8", is_slippery=True)  # slippery confirmed
    fl_src_env.reset()
    fl_mdp = fl_src_env.get_mdp_network()
    fl_nx_env_base = TimeLimit(
        NetworkXMDPEnvironment(mdp_network=fl_mdp, render_mode=None, seed=None),
        max_episode_steps=100  # FrozenLake-v1 default
    )
    print(f"FrozenLake slippery MDP: |S|={len(fl_mdp.states)}, |A|={fl_mdp.num_actions}")

    taxi_curve_greedy_all, taxi_curve_train_all = [], []
    fl_curve_greedy_all, fl_curve_train_all = [], []

    # ============================
    # Taxi training (rainy)
    # ============================
    print("\n=== Q-learning on Taxi (stochastic/rainy via NetworkX) ===")
    for seed in seeds:
        print(f"\n[Taxi] Seed = {seed}")
        taxi_train_env = TimeLimit(
            NetworkXMDPEnvironment(mdp_network=taxi_mdp, render_mode=None, seed=seed),
            max_episode_steps=200
        )
        taxi_vec_env = DummyVecEnv([lambda: taxi_train_env])

        agent = TabularQAgent(
            env=taxi_vec_env,
            learning_rate=learning_rate,
            gamma=gamma,
            policy_mix=policy_mix,
            temperature=temperature,
            seed=seed,
            verbose=1,
        )

        taxi_eval_env = TimeLimit(
            NetworkXMDPEnvironment(mdp_network=taxi_mdp, render_mode=None, seed=10_000 + seed),
            max_episode_steps=200
        )

        taxi_curve_greedy, taxi_curve_train = [], []
        eval_cb = PeriodicEvalCallback(
            eval_env=taxi_eval_env,
            eval_every=eval_every,
            n_eval_episodes=n_eval_episodes,
            greedy_scores_list=taxi_curve_greedy,
            train_scores_list=taxi_curve_train,
            eval_seed_base=42 + 1000*seed,
            verbose=1,
        )

        agent.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=True,
            progress_bar=True,
            callback=eval_cb
        )

        taxi_curve_greedy_all.append(np.array(taxi_curve_greedy, dtype=float))
        taxi_curve_train_all.append(np.array(taxi_curve_train, dtype=float))

        # Save agent per seed
        agent_path = os.path.join(output_dir, f"tabq_taxi_seed{seed}.zip")
        agent.save(agent_path)
        print(f"[Taxi] Saved agent -> {agent_path}")

        # ===== After training: export Q -> policy, then record GIF =====
        taxi_policy = q_table_to_policy(
            agent.q,             # QTable
            taxi_mdp.states,     # list of states from MDP
            taxi_mdp.num_actions,
            temperature=0.0      # greedy policy
        )

        # Renderer env; backend transitions are from NetworkX (TimeLimit applied)
        taxi_gif_backend = TimeLimit(
            NetworkXMDPEnvironment(mdp_network=taxi_mdp, render_mode=None, seed=50_000 + seed),
            max_episode_steps=200
        )
        taxi_rgb_env = CustomisedTaxiEnv(
            render_mode="rgb_array",
            is_rainy=True,                  # rainy visuals
            networkx_env=taxi_gif_backend   # backend dynamics
        )

        taxi_frames = record_policy_gif(
            taxi_rgb_env, taxi_policy, taxi_action_names,
            seed=500 + seed*10, episodes=3, max_steps=200
        )
        if taxi_frames:
            taxi_gif_path = os.path.join(output_dir, f"taxi_qlearn_policy_seed{seed}.gif")
            taxi_frames[0].save(taxi_gif_path, save_all=True, append_images=taxi_frames[1:], duration=500, loop=0)
            print(f"[Taxi] Saved policy GIF -> {taxi_gif_path}")
        else:
            print("[Taxi] Warning: no frames for GIF")

    # ============================
    # FrozenLake training (slippery)
    # ============================
    print("\n=== Q-learning on FrozenLake (8x8, slippery via NetworkX) ===")
    for seed in seeds:
        print(f"\n[FrozenLake] Seed = {seed}")
        fl_train_env = TimeLimit(
            NetworkXMDPEnvironment(mdp_network=fl_mdp, render_mode=None, seed=seed),
            max_episode_steps=100
        )
        fl_vec_env = DummyVecEnv([lambda: fl_train_env])

        agent = TabularQAgent(
            env=fl_vec_env,
            learning_rate=learning_rate,
            gamma=gamma,
            policy_mix=policy_mix,
            temperature=temperature,
            seed=seed,
            verbose=1,
        )

        fl_eval_env = TimeLimit(
            NetworkXMDPEnvironment(mdp_network=fl_mdp, render_mode=None, seed=20_000 + seed),
            max_episode_steps=100
        )

        fl_curve_greedy, fl_curve_train = [], []
        eval_cb = PeriodicEvalCallback(
            eval_env=fl_eval_env,
            eval_every=eval_every,
            n_eval_episodes=n_eval_episodes,
            greedy_scores_list=fl_curve_greedy,
            train_scores_list=fl_curve_train,
            eval_seed_base=142 + 1000*seed,
            verbose=1,
        )

        agent.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=True,
            progress_bar=True,
            callback=eval_cb
        )

        fl_curve_greedy_all.append(np.array(fl_curve_greedy, dtype=float))
        fl_curve_train_all.append(np.array(fl_curve_train, dtype=float))

        agent_path = os.path.join(output_dir, f"tabq_frozenlake_seed{seed}.zip")
        agent.save(agent_path)
        print(f"[FrozenLake] Saved agent -> {agent_path}")

        # ===== After training: export Q -> policy, then record GIF =====
        fl_policy = q_table_to_policy(
            agent.q,          # QTable
            fl_mdp.states,    # list of states
            fl_mdp.num_actions,
            temperature=0.0
        )

        fl_gif_backend = TimeLimit(
            NetworkXMDPEnvironment(mdp_network=fl_mdp, render_mode=None, seed=60_000 + seed),
            max_episode_steps=100
        )
        fl_rgb_env = CustomisedFrozenLakeEnv(
            render_mode="rgb_array",
            map_name="8x8",
            is_slippery=True,              # slippery visuals
            networkx_env=fl_gif_backend
        )

        fl_frames = record_policy_gif(
            fl_rgb_env, fl_policy, fl_action_names,
            seed=600 + seed*10, episodes=3, max_steps=100
        )
        if fl_frames:
            fl_gif_path = os.path.join(output_dir, f"frozenlake_qlearn_policy_seed{seed}.gif")
            fl_frames[0].save(fl_gif_path, save_all=True, append_images=fl_frames[1:], duration=500, loop=0)
            print(f"[FrozenLake] Saved policy GIF -> {fl_gif_path}")
        else:
            print("[FrozenLake] Warning: no frames for GIF")

    # ============================
    # Aggregate curves (mean ± std)
    # ============================
    def mean_std(curves_list):
        arr = np.stack(curves_list, axis=0)
        return arr.mean(axis=0), arr.std(axis=0)

    checkpoints = [0] + list(range(eval_every, total_timesteps + 1, eval_every))
    x = np.array(checkpoints, dtype=int)

    taxi_greedy_mean, taxi_greedy_std = mean_std(taxi_curve_greedy_all)
    taxi_train_mean, taxi_train_std = mean_std(taxi_curve_train_all)
    fl_greedy_mean, fl_greedy_std = mean_std(fl_curve_greedy_all)
    fl_train_mean, fl_train_std = mean_std(fl_curve_train_all)

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.title("Stochastic Taxi (rainy via NetworkX) — Q-learning curves")
    plt.plot(x, taxi_greedy_mean, label="Greedy policy (deterministic=True)")
    plt.fill_between(x, taxi_greedy_mean - taxi_greedy_std, taxi_greedy_mean + taxi_greedy_std, alpha=0.2)
    plt.plot(x, taxi_train_mean, label="Training policy (deterministic=False)")
    plt.fill_between(x, taxi_train_mean - taxi_train_std, taxi_train_mean + taxi_train_std, alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel(f"Mean return over {n_eval_episodes} eps")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Stochastic FrozenLake 8x8 (slippery via NetworkX) — Q-learning curves")
    plt.plot(x, fl_greedy_mean, label="Greedy policy (deterministic=True)")
    plt.fill_between(x, fl_greedy_mean - fl_greedy_std, fl_greedy_mean + fl_greedy_std, alpha=0.2)
    plt.plot(x, fl_train_mean, label="Training policy (deterministic=False)")
    plt.fill_between(x, fl_train_mean - fl_train_std, fl_train_mean + fl_train_std, alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel(f"Mean return over {n_eval_episodes} eps")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "qlearning_stoch_taxi_frozenlake_curves.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved curve figure: {fig_path}")

    print("\n=== Summary (mean return at last checkpoint) ===")
    print(f"Taxi — Greedy: {taxi_greedy_mean[-1]:.3f} ± {taxi_greedy_std[-1]:.3f} | "
          f"TrainPol: {taxi_train_mean[-1]:.3f} ± {taxi_train_std[-1]:.3f}")
    print(f"FrozenLake — Greedy: {fl_greedy_mean[-1]:.3f} ± {fl_greedy_std[-1]:.3f} | "
          f"TrainPol: {fl_train_mean[-1]:.3f} ± {fl_train_std[-1]:.3f}")

    print("\nDone.")
