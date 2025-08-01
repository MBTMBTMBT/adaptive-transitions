import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from plots import plot_values, plot_policy, plot_q_values
from mdp_tables import q_table_to_policy
from solvers import *
from customisable_minigrid.minigrid.customisable_minigrid_env import CustomMiniGridEnv
from samplers import sample_mdp_network_deterministic


def ensure_output_dir(output_dir: str = "output_plots"):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def create_random_policy(mdp_network: MDPNetwork) -> PolicyTable:
    """Create a uniform probabilistic policy for the MDP."""
    policy = PolicyTable()
    for state in mdp_network.states:
        if mdp_network.is_terminal_state(state):
            # Terminal states have deterministic action 0
            policy.set_action(state, 0)
        else:
            # Create uniform probability distribution over actions
            uniform_prob = 1.0 / mdp_network.num_actions

            action_probs = {}
            for action in range(mdp_network.num_actions):
                action_probs[action] = uniform_prob

            policy.set_action_probabilities(state, action_probs)

    return policy


# Unit tests and visualization script
if __name__ == "__main__":
    print("=== MDP Solver Unit Tests with Network Visualization ===\n")

    # Create output directory
    output_dir = ensure_output_dir("output_plots")
    print(f"Output directory: {output_dir}\n")

    # Test configurations
    mdps = []
    prefixes = []

    # Load the chain MDP
    mdps.append(MDPNetwork(config_path="./mdps/chain.json"))
    prefixes.append("chain")

    # Sample MDP from environment
    env = CustomMiniGridEnv(
        json_file_path='customisable_minigrid/maps/door-key-no-random.json',
        config=None,
        display_size=None,
        display_mode="middle",
        random_rotate=False,
        random_flip=False,
        render_carried_objs=True,
        render_mode="rgb_array",
    )
    sampled_mdp, int_to_string, string_to_int = sample_mdp_network_deterministic(env, max_states=500)
    mdps.append(sampled_mdp)
    prefixes.append("sampled")

    # Common parameters
    gamma = 0.9
    theta = 1e-6
    max_iterations = 1000

    # Run tests for each MDP
    for i, (mdp, prefix) in enumerate(zip(mdps, prefixes)):
        print(f"\n{'=' * 50}")
        print(f"Testing MDP {i + 1}: {prefix}")
        print(f"{'=' * 50}")
        print(f"Loaded MDP with {len(mdp.states)} states and {mdp.num_actions} actions")
        print(f"Start states: {mdp.start_states}")
        print(f"Terminal states: {list(mdp.terminal_states)}")

        # Create a random policy
        random_policy = create_random_policy(mdp)

        # Test 1: Policy Evaluation with Random Policy
        print("=== Test 1: Policy Evaluation ===")
        pe_values = policy_evaluation(mdp, random_policy, gamma, theta, max_iterations)
        print("Policy Evaluation Results:")
        for state in sorted(mdp.states)[:5]:
            value = pe_values.get_value(state)
            print(f"  State {state}: {value:.6f}")

        # Plot results
        plot_policy(mdp, random_policy, "Input: Random Probabilistic Policy",
                    os.path.join(output_dir, f"{prefix}_random_policy.png"))
        plot_values(mdp, pe_values, "Output: State Values from Policy Evaluation",
                    os.path.join(output_dir, f"{prefix}_policy_evaluation_values.png"))

        # Test 2: Optimal Value Iteration
        print("\n=== Test 2: Optimal Value Iteration ===")
        opt_values, opt_q_table = optimal_value_iteration(mdp, gamma, theta, max_iterations)

        print("Optimal Value Iteration Results:")
        for state in sorted(mdp.states)[:5]:
            value = opt_values.get_value(state)
            print(f"  State {state}: Value={value:.6f}")

        # Convert Q-table to greedy policy
        greedy_policy = q_table_to_policy(opt_q_table, mdp.states, mdp.num_actions, temperature=0.0)

        # Plot results
        plot_values(mdp, opt_values, "Output: Optimal State Values from Value Iteration",
                    os.path.join(output_dir, f"{prefix}_optimal_values.png"))
        plot_q_values(mdp, opt_q_table, "Output: Optimal Q-Values from Value Iteration",
                      os.path.join(output_dir, f"{prefix}_optimal_q_values.png"))
        plot_policy(mdp, greedy_policy, "Output: Optimal Greedy Policy from Q-Values",
                    os.path.join(output_dir, f"{prefix}_optimal_policy.png"))

        # Test 3: Q-Learning
        print("\n=== Test 3: Q-Learning ===")
        ql_q_table, ql_values = q_learning(
            mdp, alpha=0.1, gamma=gamma, epsilon=0.1,
            num_episodes=5000, max_steps_per_episode=100, seed=42)

        print("Q-Learning Results:")
        for state in sorted(mdp.states)[:5]:
            value = ql_values.get_value(state)
            print(f"  State {state}: Value={value:.6f}")

        # Create policy from Q-learning results
        ql_policy = q_table_to_policy(ql_q_table, mdp.states, mdp.num_actions, temperature=0.0)

        # Plot results
        plot_values(mdp, ql_values, "Output: State Values from Q-Learning",
                    os.path.join(output_dir, f"{prefix}_qlearning_values.png"))
        plot_q_values(mdp, ql_q_table, "Output: Learned Q-Values from Q-Learning",
                      os.path.join(output_dir, f"{prefix}_qlearning_q_values.png"))
        plot_policy(mdp, ql_policy, "Output: Learned Greedy Policy from Q-Learning",
                    os.path.join(output_dir, f"{prefix}_qlearning_policy.png"))

        # Test 4: Occupancy Measure
        print("\n=== Test 4: Occupancy Measure ===")
        occupancy = compute_occupancy_measure(mdp, random_policy, gamma, theta, max_iterations)

        print("Occupancy Measure Results:")
        for state in sorted(mdp.states)[:5]:
            occ_val = occupancy.get_value(state)
            print(f"  State {state}: {occ_val:.6f}")

        # Plot occupancy measure
        plot_values(mdp, occupancy, "Output: State Occupancy Frequencies",
                    os.path.join(output_dir, f"{prefix}_occupancy_measure.png"))

        # Print summary statistics
        print(f"\n=== Summary Statistics for {prefix} ===")
        print(f"Total states: {len(mdp.states)}")
        print(f"Terminal states: {len(mdp.terminal_states)}")
        print(f"Actions available: {mdp.num_actions}")

        print("Value differences (vs Optimal):")
        for state in sorted(mdp.states)[:5]:
            pe_val = pe_values.get_value(state)
            opt_val = opt_values.get_value(state)
            ql_val = ql_values.get_value(state)
            pe_diff = abs(pe_val - opt_val)
            ql_diff = abs(ql_val - opt_val)
            print(f"  State {state}: Random Policy diff={pe_diff:.6f}, Q-Learning diff={ql_diff:.6f}")

    print(f"\n=== All tests completed! ===")
    print(f"Generated plots in '{output_dir}' with prefixes: {prefixes}")
