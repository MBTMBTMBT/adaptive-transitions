"""
Unit tests for Taxi x NetworkX pipeline with R(s,a,s').
Flow:
  A) Taxi (dry) -> MDPNetwork (deterministic) -> DP -> GIF
  B) Taxi (rainy) -> MDPNetwork (stochastic) -> NetworkX env -> DP -> GIF
"""

import os
from typing import Dict, List
from PIL import Image, ImageDraw
import numpy as np

from customised_toy_text_envs.customised_taxi import CustomisedTaxiEnv
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


def overlay_label(frame_ndarray, text: str) -> Image.Image:
    """Overlay a small black bar with text on top of an RGB frame."""
    if frame_ndarray is None:
        return None
    img = Image.fromarray(frame_ndarray)
    draw = ImageDraw.Draw(img)
    bar_h = 22
    draw.rectangle([(0, 0), (img.width, bar_h)], fill=(0, 0, 0))
    draw.text((8, 4), text, fill=(255, 255, 255))
    return img


def record_policy_gif(env, policy, action_names: List[str], seed: int, episodes: int, max_steps: int) -> List[Image.Image]:
    """Run episodes following a (probabilistic) policy and collect frames."""
    frames: List[Image.Image] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        state = obs

        # initial frame
        frame = env.render()
        img = overlay_label(frame, f"state={state}, action=-")
        if img:
            frames.append(img)

        ep_reward = 0.0
        for t in range(max_steps):
            # greedy action from policy distribution
            action_probs = policy.get_action_probabilities(state)
            action = max(action_probs.items(), key=lambda x: x[1])[0] if action_probs else env.action_space.sample()

            obs_next, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            frame = env.render()
            img = overlay_label(frame, f"state={state}, action={action_names[action]}")
            if img:
                frames.append(img)

            state = obs_next
            if terminated or truncated:
                # pad a few frames
                for _ in range(5):
                    if frames:
                        frames.append(frames[-1])
                break
    return frames


def summarize_stochasticity(mdp: MDPNetwork, sample_states: int = 400):
    """
    Quick sanity-check: report how many (s,a) have >1 successors (stochastic).
    """
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
    print(f"[Stochasticity] (s,a) pairs with >1 successors: {multi_succ}/{total_pairs} ({ratio:.1%})")
    for s, a, trans in examples:
        print(f"  e.g. s={s}, a={a} -> {trans}")


# -----------------------------
# Main test
# -----------------------------
if __name__ == "__main__":
    output_dir = "./outputs"
    ensure_dir(output_dir)

    print("=== Taxi x NetworkX Unit Test (R(s,a,s')) ===")
    action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
    gamma, theta, max_iter = 0.99, 1e-6, 1000

    # ===== Group A: Deterministic (original Taxi dynamics => MDPNetwork) =====
    print("\n=== Group A: Deterministic (dry) ===")
    taxi_env = CustomisedTaxiEnv(render_mode=None, is_rainy=False)  # dry
    taxi_env.reset(seed=42)
    print(f"Taxi (dry): {taxi_env.observation_space.n} states, {taxi_env.action_space.n} actions")

    # Build MDP from Taxi (no sampling)
    det_mdp: MDPNetwork = taxi_env.get_mdp_network()
    print(f"MDP (dry): |S|={len(det_mdp.states)}, |T|={len(det_mdp.terminal_states)}")
    print(f"Start states: {len(det_mdp.start_states)}")

    # Export JSON
    det_mdp_path = os.path.join(output_dir, "det_taxi_mdp.json")
    det_mdp.export_to_json(det_mdp_path)
    print(f"Exported deterministic MDP: {det_mdp_path}")

    # DP on deterministic MDP
    rand_pol_A = create_random_policy(det_mdp)
    print("\n--- Policy Evaluation (A) ---")
    pe_A = policy_evaluation(det_mdp, rand_pol_A, gamma, theta, max_iter)
    print("--- Optimal Value Iteration (A) ---")
    vA, qA = optimal_value_iteration(det_mdp, gamma, theta, max_iter)

    # Export CSVs
    rand_pol_A.export_to_csv(os.path.join(output_dir, "det_taxi_random_policy.csv"))
    pe_A.export_to_csv(os.path.join(output_dir, "det_taxi_pe_values.csv"))
    vA.export_to_csv(os.path.join(output_dir, "det_taxi_optimal_values.csv"))
    qA.export_to_csv(os.path.join(output_dir, "det_taxi_optimal_q_values.csv"))
    print("Saved deterministic CSVs (policy/value/Q).")

    # Greedy policy from Q*
    opt_pol_A = q_table_to_policy(qA, det_mdp.states, det_mdp.num_actions, temperature=0.0)

    # Record GIF on original Taxi env (no NetworkX)
    print("\nCreating deterministic GIF...")
    taxi_rgb = CustomisedTaxiEnv(render_mode="rgb_array", is_rainy=False)
    frames_A = record_policy_gif(taxi_rgb, opt_pol_A, action_names, seed=100, episodes=3, max_steps=100)
    if frames_A:
        gif_A = os.path.join(output_dir, "det_taxi_optimal_policy_demo.gif")
        frames_A[0].save(gif_A, save_all=True, append_images=frames_A[1:], duration=500, loop=0)
        print(f"Saved GIF: {gif_A}")
    else:
        print("Warning: no frames for deterministic GIF")

    # ===== Group B: Stochastic (rainy Taxi => MDPNetwork => NetworkX) =====
    print("\n=== Group B: Stochastic (rainy) via NetworkX ===")
    rainy_env = CustomisedTaxiEnv(render_mode=None, is_rainy=True)  # rainy => stochastic transitions
    rainy_env.reset(seed=123)
    print(f"Taxi (rainy): {rainy_env.observation_space.n} states, {rainy_env.action_space.n} actions")

    # Build MDP from rainy Taxi
    stoch_mdp: MDPNetwork = rainy_env.get_mdp_network()
    print(f"MDP (rainy): |S|={len(stoch_mdp.states)}, |T|={len(stoch_mdp.terminal_states)}")
    summarize_stochasticity(stoch_mdp)  # prove stochastic transitions exist

    # Export JSON
    stoch_mdp_path = os.path.join(output_dir, "stoch_taxi_mdp.json")
    stoch_mdp.export_to_json(stoch_mdp_path)
    print(f"Exported stochastic MDP: {stoch_mdp_path}")

    # NetworkX-backed env using the stochastic MDP
    nx_env = NetworkXMDPEnvironment(mdp_network=stoch_mdp, render_mode=None)
    print(f"NetworkX env (rainy MDP): {nx_env.observation_space.n} states")

    # Short random walk sanity-check on NetworkX env
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

    # DP on stochastic MDP
    rand_pol_B = create_random_policy(stoch_mdp)
    print("\n--- Policy Evaluation (B) ---")
    pe_B = policy_evaluation(stoch_mdp, rand_pol_B, gamma, theta, max_iter)
    print("--- Optimal Value Iteration (B) ---")
    vB, qB = optimal_value_iteration(stoch_mdp, gamma, theta, max_iter)

    # Export CSVs
    rand_pol_B.export_to_csv(os.path.join(output_dir, "stoch_taxi_random_policy.csv"))
    pe_B.export_to_csv(os.path.join(output_dir, "stoch_taxi_pe_values.csv"))
    vB.export_to_csv(os.path.join(output_dir, "stoch_taxi_optimal_values.csv"))
    qB.export_to_csv(os.path.join(output_dir, "stoch_taxi_optimal_q_values.csv"))
    print("Saved stochastic CSVs (policy/value/Q).")

    # Greedy policy from Q*
    opt_pol_B = q_table_to_policy(qB, stoch_mdp.states, stoch_mdp.num_actions, temperature=0.0)

    # Record GIF on NetworkX-backed env (stochastic dynamics come from stoch_mdp)
    print("\nCreating stochastic GIF (NetworkX-backed with rainy MDP)...")
    taxi_nx_rgb = CustomisedTaxiEnv(render_mode="rgb_array", networkx_env=nx_env)  # backend: NetworkXMDPEnvironment
    frames_B = record_policy_gif(taxi_nx_rgb, opt_pol_B, action_names, seed=200, episodes=3, max_steps=100)
    if frames_B:
        gif_B = os.path.join(output_dir, "stoch_taxi_optimal_policy_demo.gif")
        frames_B[0].save(gif_B, save_all=True, append_images=frames_B[1:], duration=500, loop=0)
        print(f"Saved GIF: {gif_B}")
    else:
        print("Warning: no frames for stochastic GIF")

    # Summary
    print("\n=== Summary ===")
    print(f"Deterministic MDP JSON: {det_mdp_path}")
    print(f"Stochastic   MDP JSON: {stoch_mdp_path}")
    print("Deterministic CSVs: det_taxi_random_policy.csv, det_taxi_pe_values.csv, det_taxi_optimal_values.csv, det_taxi_optimal_q_values.csv")
    print("Stochastic   CSVs: stoch_taxi_random_policy.csv, stoch_taxi_pe_values.csv, stoch_taxi_optimal_values.csv, stoch_taxi_optimal_q_values.csv")
    print("GIFs: det_taxi_optimal_policy_demo.gif, stoch_taxi_optimal_policy_demo.gif")
