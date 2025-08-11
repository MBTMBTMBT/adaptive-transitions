"""
Unit tests for Taxi & MiniGrid x NetworkX pipeline with R(s,a,s').

Flows:
  A) Taxi (dry)    -> MDPNetwork (deterministic) -> DP -> GIF
  B) Taxi (rainy)  -> MDPNetwork (stochastic)    -> NetworkX env -> DP -> GIF
  C) MiniGrid (det)-> MDPNetwork (deterministic) -> NetworkX env -> DP -> GIF
"""

import os
from typing import Dict, List, Callable, Optional
from PIL import Image, ImageDraw

import numpy as np

from customised_toy_text_envs.customised_taxi import CustomisedTaxiEnv
from customised_minigrid_env.customised_minigrid_env import CustomMiniGridEnv
from networkx_env.networkx_env import NetworkXMDPEnvironment
from mdp_network.solvers import optimal_value_iteration, policy_evaluation
from mdp_network.mdp_tables import create_random_policy, q_table_to_policy
from mdp_network.mdp_network import MDPNetwork


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(d: str):
    if not os.path.exists(d):
        os.makedirs(d)


def overlay_label(frame_ndarray, text: str) -> Optional[Image.Image]:
    """Overlay a small black bar with text on top of an RGB frame."""
    if frame_ndarray is None:
        return None
    img = Image.fromarray(frame_ndarray)
    draw = ImageDraw.Draw(img)
    bar_h = 22
    draw.rectangle([(0, 0), (img.width, bar_h)], fill=(0, 0, 0))
    draw.text((8, 4), text, fill=(255, 255, 255))
    return img


def record_policy_gif(
    env,
    policy,
    action_names: List[str],
    seed: int,
    episodes: int,
    max_steps: int,
    *,
    use_encode: bool = False,
    state_fn: Optional[Callable] = None,
) -> List[Image.Image]:
    """
    Run episodes following a (probabilistic) policy and collect frames.

    Notes:
    - For Taxi: obs is already an integer state id -> keep use_encode=False (default).
    - For MiniGrid (dict obs): pass use_encode=True or provide state_fn to return an int id
      (e.g., lambda env, obs: env.encode_state()).
    """
    frames: List[Image.Image] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)

        # Derive the state id for the policy
        if state_fn is not None:
            state = state_fn(env, obs)
        elif use_encode and hasattr(env, "encode_state"):
            state = env.encode_state()
        else:
            state = obs  # works for Taxi; MiniGrid should not rely on this path

        # First frame
        frame = env.render()
        img = overlay_label(frame, f"state={state}, action=-")
        if img:
            frames.append(img)

        ep_reward = 0.0
        for t in range(max_steps):
            # Greedy action from policy distribution
            action_probs = policy.get_action_probabilities(state)
            action = max(action_probs.items(), key=lambda x: x[1])[0] if action_probs else env.action_space.sample()

            obs_next, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            frame = env.render()
            an = action_names[action] if 0 <= action < len(action_names) else str(action)
            img = overlay_label(frame, f"state={state}, action={an}")
            if img:
                frames.append(img)

            # Update state id for the policy
            if state_fn is not None:
                state = state_fn(env, obs_next)
            elif use_encode and hasattr(env, "encode_state"):
                state = env.encode_state()
            else:
                state = obs_next

            if terminated or truncated:
                # Pad a few frames for nicer loops
                for _ in range(5):
                    if frames:
                        frames.append(frames[-1])
                break
    return frames


def summarize_stochasticity(mdp: MDPNetwork, sample_states: int = 400):
    """Quick sanity-check: count (s,a) pairs having >1 successors."""
    import random
    rng = random.Random(0)
    states = mdp.states if len(mdp.states) <= sample_states else rng.sample(mdp.states, sample_states)
    total_pairs, multi_succ, examples = 0, 0, []
    for s in states:
        for a in range(mdp.num_actions):
            trans = mdp.get_transition_probabilities(s, a)
            if not trans:
                continue
            total_pairs += 1
            if len(trans) > 1:
                multi_succ += 1
                if len(examples) < 5:
                    examples.append((s, a, trans))
    ratio = (multi_succ / total_pairs) if total_pairs else 0.0
    print(f"[Stochasticity] (s,a) with >1 successors: {multi_succ}/{total_pairs} ({ratio:.1%})")
    for s, a, trans in examples:
        print(f"  e.g. s={s}, a={a} -> {trans}")


# -----------------------------
# Main test
# -----------------------------
if __name__ == "__main__":
    output_dir = "./outputs"
    ensure_dir(output_dir)

    print("=== Taxi & MiniGrid x NetworkX Unit Test (R(s,a,s')) ===")
    taxi_action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
    gamma, theta, max_iter = 0.99, 1e-6, 1000

    # ---------------------------------------------------------
    # Group A: Deterministic (dry Taxi) => MDPNetwork => DP => GIF
    # ---------------------------------------------------------
    print("\n=== Group A: Deterministic (dry) ===")
    taxi_env = CustomisedTaxiEnv(render_mode=None, is_rainy=False)
    taxi_env.reset(seed=42)
    print(f"Taxi (dry): {taxi_env.observation_space.n} states, {taxi_env.action_space.n} actions")

    det_mdp: MDPNetwork = taxi_env.get_mdp_network()
    print(f"MDP (dry): |S|={len(det_mdp.states)}, |T|={len(det_mdp.terminal_states)}")
    print(f"Start states: {len(det_mdp.start_states)}")

    det_mdp_path = os.path.join(output_dir, "det_taxi_mdp.json")
    det_mdp.export_to_json(det_mdp_path)
    print(f"Exported deterministic MDP: {det_mdp_path}")

    rand_pol_A = create_random_policy(det_mdp)
    print("\n--- Policy Evaluation (A) ---")
    _ = policy_evaluation(det_mdp, rand_pol_A, gamma, theta, max_iter)
    print("--- Optimal Value Iteration (A) ---")
    vA, qA = optimal_value_iteration(det_mdp, gamma, theta, max_iter)

    rand_pol_A.export_to_csv(os.path.join(output_dir, "det_taxi_random_policy.csv"))
    vA.export_to_csv(os.path.join(output_dir, "det_taxi_optimal_values.csv"))
    qA.export_to_csv(os.path.join(output_dir, "det_taxi_optimal_q_values.csv"))
    print("Saved deterministic CSVs (policy/value/Q).")

    opt_pol_A = q_table_to_policy(qA, det_mdp.states, det_mdp.num_actions, temperature=0.0)

    print("\nCreating deterministic GIF (Taxi)...")
    taxi_rgb = CustomisedTaxiEnv(render_mode="rgb_array", is_rainy=False)
    frames_A = record_policy_gif(taxi_rgb, opt_pol_A, taxi_action_names, seed=100, episodes=3, max_steps=100)
    if frames_A:
        gif_A = os.path.join(output_dir, "det_taxi_optimal_policy_demo.gif")
        frames_A[0].save(gif_A, save_all=True, append_images=frames_A[1:], duration=500, loop=0)
        print(f"Saved GIF: {gif_A}")
    else:
        print("Warning: no frames for deterministic Taxi GIF")

    # ---------------------------------------------------------
    # Group B: Stochastic (rainy Taxi) => MDPNetwork => NetworkX => DP => GIF
    # ---------------------------------------------------------
    print("\n=== Group B: Stochastic (rainy) via NetworkX ===")
    rainy_env = CustomisedTaxiEnv(render_mode=None, is_rainy=True)
    rainy_env.reset(seed=123)
    print(f"Taxi (rainy): {rainy_env.observation_space.n} states, {rainy_env.action_space.n} actions")

    stoch_mdp: MDPNetwork = rainy_env.get_mdp_network()
    print(f"MDP (rainy): |S|={len(stoch_mdp.states)}, |T|={len(stoch_mdp.terminal_states)}")
    summarize_stochasticity(stoch_mdp)

    stoch_mdp_path = os.path.join(output_dir, "stoch_taxi_mdp.json")
    stoch_mdp.export_to_json(stoch_mdp_path)
    print(f"Exported stochastic MDP: {stoch_mdp_path}")

    nx_env = NetworkXMDPEnvironment(mdp_network=stoch_mdp, render_mode=None)
    print(f"NetworkX env (rainy MDP): {nx_env.observation_space.n} states")

    print("Random walk (10 steps) on NetworkX env:")
    obs, info = nx_env.reset(seed=42)
    total_r = 0.0
    for t in range(10):
        a = nx_env.action_space.sample()
        nxt, r, term, trunc, _info = nx_env.step(a)
        total_r += r
        print(f"  t={t+1}: a={a}, {obs}->{nxt}, r={r:.3f}")
        obs = nxt
        if term or trunc:
            print("  Episode ended early.")
            break
    print(f"Total reward: {total_r:.3f}")

    rand_pol_B = create_random_policy(stoch_mdp)
    print("\n--- Policy Evaluation (B) ---")
    _ = policy_evaluation(stoch_mdp, rand_pol_B, gamma, theta, max_iter)
    print("--- Optimal Value Iteration (B) ---")
    vB, qB = optimal_value_iteration(stoch_mdp, gamma, theta, max_iter)

    rand_pol_B.export_to_csv(os.path.join(output_dir, "stoch_taxi_random_policy.csv"))
    vB.export_to_csv(os.path.join(output_dir, "stoch_taxi_optimal_values.csv"))
    qB.export_to_csv(os.path.join(output_dir, "stoch_taxi_optimal_q_values.csv"))
    print("Saved stochastic CSVs (policy/value/Q).")

    opt_pol_B = q_table_to_policy(qB, stoch_mdp.states, stoch_mdp.num_actions, temperature=0.0)

    print("\nCreating stochastic GIF (NetworkX-backed with rainy MDP, Taxi renderer)...")
    taxi_nx_rgb = CustomisedTaxiEnv(render_mode="rgb_array", networkx_env=nx_env)
    frames_B = record_policy_gif(taxi_nx_rgb, opt_pol_B, taxi_action_names, seed=200, episodes=3, max_steps=100)
    if frames_B:
        gif_B = os.path.join(output_dir, "stoch_taxi_optimal_policy_demo.gif")
        frames_B[0].save(gif_B, save_all=True, append_images=frames_B[1:], duration=500, loop=0)
        print(f"Saved GIF: {gif_B}")
    else:
        print("Warning: no frames for stochastic Taxi GIF")

    # ---------------------------------------------------------
    # Group C: MiniGrid (deterministic) => MDPNetwork => NetworkX => DP => GIF
    # ---------------------------------------------------------
    print("\n=== Group C: MiniGrid (deterministic) via NetworkX ===")
    # NOTE: choose your layout file; deterministic here just means we export dynamics as-is
    mg_map_path = "../customised_minigrid_env/maps/three-rooms.json"

    # Build MDP from MiniGrid configuration
    mg_env_for_mdp = CustomMiniGridEnv(
        json_file_path=mg_map_path,
        config=None,
        display_size=None,          # let env choose
        display_mode="middle",      # fixed placement for reproducibility
        random_rotate=False,        # keep fixed
        random_flip=False,          # keep fixed
        render_carried_objs=True,
        render_mode=None,
    )
    # Some MiniGrid configs may randomize within allowed regions; if your implementation enforces "det only",
    # the constructor or get_mdp_network should raise on randomness flags as agreed earlier.
    mg_env_for_mdp.reset(seed=7)

    mg_mdp: MDPNetwork = mg_env_for_mdp.get_mdp_network()
    print(f"MiniGrid MDP: |S|={len(mg_mdp.states)}, |T|={len(mg_mdp.terminal_states)}")

    mg_mdp_path = os.path.join(output_dir, "det_minigrid_mdp.json")
    mg_mdp.export_to_json(mg_mdp_path)
    print(f"Exported MiniGrid deterministic MDP: {mg_mdp_path}")

    # DP on MiniGrid MDP
    mg_rand_pol = create_random_policy(mg_mdp)
    print("\n--- Policy Evaluation (C) ---")
    _ = policy_evaluation(mg_mdp, mg_rand_pol, gamma, theta, max_iter)
    print("--- Optimal Value Iteration (C) ---")
    mg_V, mg_Q = optimal_value_iteration(mg_mdp, gamma, theta, max_iter)

    mg_rand_pol.export_to_csv(os.path.join(output_dir, "det_minigrid_random_policy.csv"))
    mg_V.export_to_csv(os.path.join(output_dir, "det_minigrid_optimal_values.csv"))
    mg_Q.export_to_csv(os.path.join(output_dir, "det_minigrid_optimal_q_values.csv"))
    print("Saved MiniGrid deterministic CSVs (policy/value/Q).")

    mg_opt_pol = q_table_to_policy(mg_Q, mg_mdp.states, mg_mdp.num_actions, temperature=0.0)

    # Drive MiniGrid rendering with a NetworkX-backed env created from mg_mdp
    mg_nx_env = NetworkXMDPEnvironment(mdp_network=mg_mdp, render_mode=None)

    # Action labels for MiniGrid (adjust if your SimpleActions differs)
    mg_action_names = ["left", "right", "forward", "toggle"]

    print("\nCreating deterministic GIF (MiniGrid, NetworkX-backed)...")
    mg_rgb_env = CustomMiniGridEnv(
        json_file_path=mg_map_path,
        config=None,
        display_size=None,
        display_mode="middle",
        random_rotate=False,
        random_flip=False,
        render_carried_objs=True,
        render_mode="rgb_array",
        networkx_env=mg_nx_env,  # backend: NetworkXMDPEnvironment
    )

    # For MiniGrid, use encode_state() to get an integer state id
    mg_frames = record_policy_gif(
        mg_rgb_env, mg_opt_pol, mg_action_names,
        seed=300, episodes=3, max_steps=200,
        use_encode=True,  # or pass state_fn=lambda env, obs: env.encode_state()
    )
    if mg_frames:
        mg_gif = os.path.join(output_dir, "det_minigrid_optimal_policy_demo.gif")
        mg_frames[0].save(mg_gif, save_all=True, append_images=mg_frames[1:], duration=500, loop=0)
        print(f"Saved GIF: {mg_gif}")
    else:
        print("Warning: no frames for MiniGrid GIF")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n=== Summary ===")
    print(f"Deterministic Taxi MDP JSON: {det_mdp_path}")
    print(f"Stochastic   Taxi MDP JSON: {stoch_mdp_path}")
    print(f"Deterministic MiniGrid MDP JSON: {mg_mdp_path}")
    print("Deterministic CSVs: det_taxi_random_policy.csv, det_taxi_optimal_values.csv, det_taxi_optimal_q_values.csv")
    print("Stochastic   CSVs: stoch_taxi_random_policy.csv, stoch_taxi_optimal_values.csv, stoch_taxi_optimal_q_values.csv")
    print("MiniGrid     CSVs: det_minigrid_random_policy.csv, det_minigrid_optimal_values.csv, det_minigrid_optimal_q_values.csv")
    print("GIFs: det_taxi_optimal_policy_demo.gif, stoch_taxi_optimal_policy_demo.gif, det_minigrid_optimal_policy_demo.gif")
