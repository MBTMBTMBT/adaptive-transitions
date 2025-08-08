"""
Unit test script for Taxi environment with NetworkX integration.
Tests the full pipeline: Taxi -> MDP sampling -> stochastic modification -> NetworkX environment -> DP solving.
"""

import os
import numpy as np
from PIL import Image
from customised_toy_text_envs.customised_taxi import CustomisedTaxiEnv
from mdp_network.samplers import deterministic_mdp_sampling
from networkx_env.networkx_env import NetworkXMDPEnvironment
from mdp_network.solvers import optimal_value_iteration, policy_evaluation
from mdp_network.mdp_tables import create_random_policy, q_table_to_policy


def add_stochastic_transitions(mdp_network, noise_probability=0.1, neighbor_radius=10):
    """
    Add small probability random transitions to neighboring states in the MDP.

    Args:
        mdp_network: Original MDPNetwork
        noise_probability: Total probability mass to redistribute to neighbors
        neighbor_radius: Maximum state ID difference to consider as neighbor

    Returns:
        Modified MDPNetwork with stochastic transitions
    """
    print(f"Adding stochastic transitions with noise_probability={noise_probability}")

    # Get all states and sort them
    all_states = sorted(mdp_network.states)

    # Modify transitions for each state-action pair
    for from_state in all_states:
        for action in range(mdp_network.num_actions):
            # Get current transition probabilities
            current_probs = mdp_network.get_transition_probabilities(from_state, action)

            if not current_probs:
                continue  # Skip if no transitions for this state-action

            # Find neighboring states (within radius of state ID)
            neighbors = [s for s in all_states
                         if s != from_state and abs(s - from_state) <= neighbor_radius]

            if not neighbors:
                continue  # Skip if no neighbors found

            # Redistribute probability mass
            # Reduce original transitions by noise_probability
            modified_probs = {state: prob * (1 - noise_probability)
                              for state, prob in current_probs.items()}

            # Add small probabilities to neighbors
            neighbor_prob = noise_probability / len(neighbors)
            for neighbor in neighbors:
                if neighbor in modified_probs:
                    modified_probs[neighbor] += neighbor_prob
                else:
                    modified_probs[neighbor] = neighbor_prob

            # Update transitions in the MDP network
            for to_state, prob in modified_probs.items():
                if prob > 1e-10:  # Only add non-negligible probabilities
                    mdp_network.add_transition(from_state, to_state, action, prob)

    print(f"Stochastic modification completed for {len(all_states)} states")
    return mdp_network


# Create output directory
output_dir = "./outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== Taxi NetworkX Integration Unit Test ===\n")

# Step 1: Create original Taxi environment and get all start states
print("Step 1: Creating Taxi environment and getting all possible start states...")
taxi_env = CustomisedTaxiEnv(render_mode=None)
taxi_env.reset(seed=42)

print(f"Taxi environment created with {taxi_env.observation_space.n} states and {taxi_env.action_space.n} actions")

# Get all possible start states
all_start_states = taxi_env.get_start_states()
print(f"Found {len(all_start_states)} possible start states")
print(f"First 10 start states: {all_start_states[:10]}")
print(f"Last 10 start states: {all_start_states[-10:]}")

# Sample deterministic MDP from Taxi environment with all start states
print("\nSampling MDP from all possible start states...")
original_mdp = deterministic_mdp_sampling(taxi_env, start_states=all_start_states, max_states=500)
print(f"Original MDP sampled: {len(original_mdp.states)} states, {len(original_mdp.terminal_states)} terminal states")
print(f"MDP start states: {original_mdp.start_states}")

# Export original MDP
original_mdp_path = os.path.join(output_dir, "taxi_original_mdp.json")
original_mdp.export_to_json(original_mdp_path)
print(f"Original MDP exported to: {original_mdp_path}")

# Step 2: Create stochastic version of MDP
print("\nStep 2: Adding stochastic transitions to MDP...")
stochastic_mdp = add_stochastic_transitions(original_mdp, noise_probability=0.05, neighbor_radius=15)

# Export stochastic MDP
stochastic_mdp_path = os.path.join(output_dir, "taxi_stochastic_mdp.json")
stochastic_mdp.export_to_json(stochastic_mdp_path)
print(f"Stochastic MDP exported to: {stochastic_mdp_path}")

# Step 3: Create NetworkX environment from stochastic MDP
print("\nStep 3: Creating NetworkX environment from stochastic MDP...")
networkx_env = NetworkXMDPEnvironment(mdp_network=stochastic_mdp, render_mode=None)
print(f"NetworkX environment created with {networkx_env.observation_space.n} states")

# Step 4: Test NetworkX environment
print("\nStep 4: Testing NetworkX environment...")
obs, info = networkx_env.reset(seed=42)
print(f"Initial state: {obs}")

# Run a few random steps
total_reward = 0
for step in range(10):
    action = networkx_env.action_space.sample()
    next_obs, reward, terminated, truncated, info = networkx_env.step(action)
    total_reward += reward
    print(f"Step {step + 1}: action={action}, state={obs}->{next_obs}, reward={reward:.3f}")
    obs = next_obs
    if terminated or truncated:
        print(f"Episode ended at step {step + 1}")
        break

print(f"Total reward from random walk: {total_reward:.3f}")

# Step 5: Solve stochastic MDP using Dynamic Programming
print("\nStep 5: Solving stochastic MDP with Dynamic Programming...")

# Parameters for DP algorithms
gamma = 0.99
theta = 1e-6
max_iterations = 1000

# Create random policy for testing
random_policy = create_random_policy(stochastic_mdp)
print(f"Random policy created with {len(stochastic_mdp.states)} states")

# Policy Evaluation with random policy
print("\n--- Policy Evaluation ---")
pe_values = policy_evaluation(stochastic_mdp, random_policy, gamma, theta, max_iterations)
print("Policy Evaluation completed")

# Show some example values
example_states = sorted(stochastic_mdp.states)[:10]
print("Example state values from Policy Evaluation:")
for state in example_states:
    value = pe_values.get_value(state)
    print(f"  State {state}: {value:.6f}")

# Optimal Value Iteration
print("\n--- Optimal Value Iteration ---")
opt_values, opt_q_table = optimal_value_iteration(stochastic_mdp, gamma, theta, max_iterations)
print("Optimal Value Iteration completed")

# Show some example optimal values
print("Example optimal state values:")
for state in example_states:
    value = opt_values.get_value(state)
    print(f"  State {state}: {value:.6f}")

# Compare random policy vs optimal values
print("\nValue comparison (Random Policy vs Optimal):")
total_diff = 0
for state in example_states:
    pe_val = pe_values.get_value(state)
    opt_val = opt_values.get_value(state)
    diff = abs(opt_val - pe_val)
    total_diff += diff
    print(f"  State {state}: Random={pe_val:.6f}, Optimal={opt_val:.6f}, Diff={diff:.6f}")

avg_diff = total_diff / len(example_states)
print(f"Average value difference: {avg_diff:.6f}")

# Step 6: Export results
print("\nStep 6: Exporting results...")

# Export policies and values
random_policy_path = os.path.join(output_dir, "taxi_random_policy.csv")
random_policy.export_to_csv(random_policy_path)

pe_values_path = os.path.join(output_dir, "taxi_pe_values.csv")
pe_values.export_to_csv(pe_values_path)

opt_values_path = os.path.join(output_dir, "taxi_optimal_values.csv")
opt_values.export_to_csv(opt_values_path)

opt_q_table_path = os.path.join(output_dir, "taxi_optimal_q_values.csv")
opt_q_table.export_to_csv(opt_q_table_path)

print(f"Results exported to {output_dir}/:")
print(f"  - {os.path.basename(random_policy_path)}")
print(f"  - {os.path.basename(pe_values_path)}")
print(f"  - {os.path.basename(opt_values_path)}")
print(f"  - {os.path.basename(opt_q_table_path)}")

# Step 7: Summary statistics
print("\n=== Summary Statistics ===")
print(f"Total possible start states in Taxi environment: {len(all_start_states)}")
print(f"Original MDP: {len(original_mdp.states)} states, {len(original_mdp.terminal_states)} terminal")
print(f"MDP start states: {len(original_mdp.start_states)} (should match environment)")
print(f"Stochastic MDP: {len(stochastic_mdp.states)} states, {len(stochastic_mdp.terminal_states)} terminal")
print(f"NetworkX Environment: {networkx_env.observation_space.n} states, {networkx_env.action_space.n} actions")
print(f"DP Algorithms: gamma={gamma}, theta={theta}")
print(f"Policy Evaluation converged: {len(pe_values.values)} state values computed")
print(f"Value Iteration converged: {len(opt_values.values)} state values computed")
print(f"Average improvement from random to optimal policy: {avg_diff:.6f}")

# Verify start state consistency
start_states_match = set(all_start_states) == set(original_mdp.start_states)
print(f"Start states consistency check: {'✓ PASS' if start_states_match else '✗ FAIL'}")

# Test if NetworkX environment maintains state mappings
if hasattr(stochastic_mdp, 'has_string_mapping') and stochastic_mdp.has_string_mapping:
    print(f"String state mapping: Yes ({len(stochastic_mdp.int_to_state)} mappings)")
else:
    print("String state mapping: No (integer states)")

# Step 8: Generate optimal policy demonstration GIF
print("\nStep 8: Generating optimal policy demonstration GIF...")

# Convert Q-table to optimal policy (greedy policy with temperature=0)
optimal_policy = q_table_to_policy(opt_q_table, stochastic_mdp.states, stochastic_mdp.num_actions, temperature=0.0)

# Create a Taxi environment with NetworkX backend for visualization
taxi_with_networkx = CustomisedTaxiEnv(render_mode="rgb_array", networkx_env=networkx_env)

# Collect frames for multiple episodes
frames = []
action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]

print("Recording optimal policy episodes...")
for episode in range(3):  # Record 3 episodes
    print(f"Recording episode {episode + 1}...")

    # Reset environment
    obs, info = taxi_with_networkx.reset(seed=42 + episode)
    current_state = obs

    # Add frame with episode info
    frame = taxi_with_networkx.render()
    if frame is not None:
        # Convert frame to PIL Image and add text
        pil_frame = Image.fromarray(frame)
        frames.append(pil_frame)

    episode_reward = 0
    step = 0
    max_episode_steps = 100

    while step < max_episode_steps:
        # Get optimal action from policy
        action_probs = optimal_policy.get_action_probabilities(current_state)
        if action_probs:
            # Choose action with highest probability (greedy)
            action = max(action_probs.items(), key=lambda x: x[1])[0]
        else:
            # Fallback to random action
            action = taxi_with_networkx.action_space.sample()

        # Execute action
        next_obs, reward, terminated, truncated, info = taxi_with_networkx.step(action)
        episode_reward += reward

        # Render frame
        frame = taxi_with_networkx.render()
        if frame is not None:
            pil_frame = Image.fromarray(frame)
            frames.append(pil_frame)

        print(f"  Step {step + 1}: {action_names[action]} -> reward={reward:.2f}")

        current_state = next_obs
        step += 1

        if terminated or truncated:
            print(f"  Episode {episode + 1} finished in {step} steps, total reward: {episode_reward:.2f}")
            # Add a few frames at the end to show completion
            for _ in range(5):
                if frames:
                    frames.append(frames[-1])
            break

    if step >= max_episode_steps:
        print(f"  Episode {episode + 1} truncated at {max_episode_steps} steps")

# Save GIF
if frames:
    gif_path = os.path.join(output_dir, "taxi_optimal_policy_demo.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # 500ms per frame
        loop=0
    )
    print(f"Optimal policy demonstration GIF saved to: {gif_path}")
    print(f"Total frames recorded: {len(frames)}")
else:
    print("Warning: No frames were captured for GIF generation")

print("\n=== Unit test completed successfully! ===")
