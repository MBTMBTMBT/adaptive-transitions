import os
from plots import plot_values, plot_policy, plot_q_values
from mdp_tables import q_table_to_policy
from solvers import *
from customisable_minigrid.customisable_minigrid_env import CustomMiniGridEnv
from samplers import deterministic_mdp_sampling


def ensure_output_dir(output_dir: str = "./outputs"):
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
    output_dir = ensure_output_dir("./outputs")
    print(f"Output directory: {output_dir}\n")

    # Test configurations
    mdps = []
    prefixes = []

    # Load the chain MDP
    mdps.append(MDPNetwork(config_path="./mdps/chain.json"))
    prefixes.append("chain")

    # Sample MDP from environment
    env = CustomMiniGridEnv(
        json_file_path='../customisable_minigrid/maps/door-key-no-random.json',
        config=None,
        display_size=None,
        display_mode="middle",
        random_rotate=False,
        random_flip=False,
        render_carried_objs=True,
        render_mode="rgb_array",
    )
    sampled_mdp, int_to_string, string_to_int = deterministic_mdp_sampling(env, )
    mdps.append(sampled_mdp)
    prefixes.append("sampled")

    # Common parameters
    gamma = 0.99
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

        # Export MDP network to JSON
        mdp_json_path = os.path.join(output_dir, f"{prefix}_{i}_mdp_network.json")
        mdp.export_to_json(mdp_json_path)
        print(f"Exported MDP network to: {mdp_json_path}")

        # Check if MDP is large (>100 states) to decide whether to plot or print
        use_plots = len(mdp.states) <= 100

        # Create a random policy
        random_policy = create_random_policy(mdp)

        # Test 1: Policy Evaluation with Random Policy
        print("=== Test 1: Policy Evaluation ===")
        pe_values = policy_evaluation(mdp, random_policy, gamma, theta, max_iterations)
        print("Policy Evaluation Results:")
        for state in sorted(mdp.states)[:5]:
            value = pe_values.get_value(state)
            print(f"  State {state}: {value:.6f}")

        # Save CSV files for Policy Evaluation
        random_policy.export_to_csv(os.path.join(output_dir, f"{prefix}_{i}_random_policy.csv"))
        pe_values.export_to_csv(os.path.join(output_dir, f"{prefix}_{i}_policy_evaluation_values.csv"))

        # Plot or print results
        if use_plots:
            plot_policy(mdp, random_policy, "Input: Random Probabilistic Policy",
                        os.path.join(output_dir, f"{prefix}_{i}_random_policy.png"))
            plot_values(mdp, pe_values, "Output: State Values from Policy Evaluation",
                        os.path.join(output_dir, f"{prefix}_{i}_policy_evaluation_values.png"))
        else:
            print("Random Policy:")
            print(random_policy)
            print("Policy Evaluation Values:")
            print(pe_values)

        # Test 2: Optimal Value Iteration
        print("\n=== Test 2: Optimal Value Iteration ===")
        opt_values, opt_q_table = optimal_value_iteration(mdp, gamma, theta, max_iterations)

        print("Optimal Value Iteration Results:")
        for state in sorted(mdp.states)[:5]:
            value = opt_values.get_value(state)
            print(f"  State {state}: Value={value:.6f}")

        # Convert Q-table to greedy policy
        greedy_policy = q_table_to_policy(opt_q_table, mdp.states, mdp.num_actions, temperature=0.0)

        # Save CSV files for Value Iteration
        opt_values.export_to_csv(os.path.join(output_dir, f"{prefix}_{i}_optimal_values.csv"))
        opt_q_table.export_to_csv(os.path.join(output_dir, f"{prefix}_{i}_optimal_q_values.csv"))
        greedy_policy.export_to_csv(os.path.join(output_dir, f"{prefix}_{i}_optimal_policy.csv"))

        # Plot or print results
        if use_plots:
            plot_values(mdp, opt_values, "Output: Optimal State Values from Value Iteration",
                        os.path.join(output_dir, f"{prefix}_{i}_optimal_values.png"))
            plot_q_values(mdp, opt_q_table, "Output: Optimal Q-Values from Value Iteration",
                          os.path.join(output_dir, f"{prefix}_{i}_optimal_q_values.png"))
            plot_policy(mdp, greedy_policy, "Output: Optimal Greedy Policy from Q-Values",
                        os.path.join(output_dir, f"{prefix}_{i}_optimal_policy.png"))
        else:
            print("Optimal Values:")
            print(opt_values)
            print("Optimal Q-Values:")
            print(opt_q_table)
            print("Optimal Greedy Policy:")
            print(greedy_policy)

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

        # Save CSV files for Q-Learning
        ql_values.export_to_csv(os.path.join(output_dir, f"{prefix}_{i}_qlearning_values.csv"))
        ql_q_table.export_to_csv(os.path.join(output_dir, f"{prefix}_{i}_qlearning_q_values.csv"))
        ql_policy.export_to_csv(os.path.join(output_dir, f"{prefix}_{i}_qlearning_policy.csv"))

        # Plot or print results
        if use_plots:
            plot_values(mdp, ql_values, "Output: State Values from Q-Learning",
                        os.path.join(output_dir, f"{prefix}_{i}_qlearning_values.png"))
            plot_q_values(mdp, ql_q_table, "Output: Learned Q-Values from Q-Learning",
                          os.path.join(output_dir, f"{prefix}_{i}_qlearning_q_values.png"))
            plot_policy(mdp, ql_policy, "Output: Learned Greedy Policy from Q-Learning",
                        os.path.join(output_dir, f"{prefix}_{i}_qlearning_policy.png"))
        else:
            print("Q-Learning Values:")
            print(ql_values)
            print("Q-Learning Q-Values:")
            print(ql_q_table)
            print("Q-Learning Policy:")
            print(ql_policy)

        # Test 4: Occupancy Measure
        print("\n=== Test 4: Occupancy Measure ===")
        occupancy = compute_occupancy_measure(mdp, random_policy, gamma, theta, max_iterations)

        print("Occupancy Measure Results:")
        for state in sorted(mdp.states)[:5]:
            occ_val = occupancy.get_value(state)
            print(f"  State {state}: {occ_val:.6f}")

        # Save CSV files for Occupancy Measure
        occupancy.export_to_csv(os.path.join(output_dir, f"{prefix}_{i}_occupancy_measure.csv"))

        # Plot or print occupancy measure
        if use_plots:
            plot_values(mdp, occupancy, "Output: State Occupancy Frequencies",
                        os.path.join(output_dir, f"{prefix}_{i}_occupancy_measure.png"))
        else:
            print("Occupancy Measure:")
            print(occupancy)

        # Test 5: Reward Distribution
        print("\n=== Test 5: Reward Distribution ===")
        reward_count_dist, reward_prob_dist = compute_reward_distribution(mdp, occupancy, delta=1e-6)

        print("Reward Distribution Results:")
        print("Reward Count Distribution:")
        print(reward_count_dist)
        print("Reward Probability Distribution:")
        print(reward_prob_dist)

        # Save CSV files for Reward Distribution
        reward_count_dist.export_to_csv(os.path.join(output_dir, f"{prefix}_{i}_reward_count_distribution.csv"))
        reward_prob_dist.export_to_csv(os.path.join(output_dir, f"{prefix}_{i}_reward_prob_distribution.csv"))

        # Print reward distribution tables if not too large
        if not use_plots or len(reward_count_dist.get_all_rewards()) <= 20:
            print("Reward Count Distribution:")
            print(reward_count_dist)
            print("Reward Probability Distribution:")
            print(reward_prob_dist)

        # Test 6: Gaussian Fitting for Reward Distribution
        print("\n=== Test 6: Gaussian Fitting for Reward Distribution ===")

        # Fit Gaussian to count distribution (more accurate)
        mu_count, sigma_count, fit_stats_count = reward_count_dist.fit_gaussian(force_count=True)

        print("Gaussian Fitting Results (Count Distribution):")
        print(f"  Fitted parameters: μ={mu_count:.6f}, σ={sigma_count:.6f}")
        print(f"  Method: {fit_stats_count.get('method', 'N/A')}")
        print(f"  Total count: {fit_stats_count.get('total_count', 'N/A')}")
        print(f"  Unique rewards: {fit_stats_count.get('unique_rewards', 'N/A')}")
        print(f"  Reward range: {fit_stats_count.get('reward_range', 'N/A')}")

        if 'ks_p_value' in fit_stats_count:
            ks_result = "GOOD" if fit_stats_count['ks_p_value'] > 0.05 else "POOR"
            print(f"  KS test p-value: {fit_stats_count['ks_p_value']:.6f} ({ks_result} fit)")

        if 'sample_mean' in fit_stats_count and 'sample_std' in fit_stats_count:
            print(
                f"  Sample statistics: mean={fit_stats_count['sample_mean']:.6f}, std={fit_stats_count['sample_std']:.6f}")

        # Save Gaussian fitting results to text file
        gaussian_results_path = os.path.join(output_dir, f"{prefix}_{i}_gaussian_fit_results.txt")
        with open(gaussian_results_path, 'w') as f:
            f.write(f"Gaussian Fitting Results for {prefix}_{i}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Count Distribution Gaussian Fit:\n")
            f.write(f"  Fitted μ (mean): {mu_count:.6f}\n")
            f.write(f"  Fitted σ (std):  {sigma_count:.6f}\n")
            f.write(f"  Method: {fit_stats_count.get('method', 'N/A')}\n")
            f.write(f"  Total count: {fit_stats_count.get('total_count', 'N/A')}\n")
            f.write(f"  Unique rewards: {fit_stats_count.get('unique_rewards', 'N/A')}\n")
            f.write(f"  Reward range: {fit_stats_count.get('reward_range', 'N/A')}\n")

            if 'ks_p_value' in fit_stats_count:
                f.write(f"  KS test statistic: {fit_stats_count.get('ks_statistic', 'N/A'):.6f}\n")
                f.write(f"  KS test p-value: {fit_stats_count['ks_p_value']:.6f}\n")
                f.write(f"  KS test significant: {fit_stats_count.get('ks_significant', 'N/A')}\n")

            if 'sample_mean' in fit_stats_count:
                f.write(f"  Sample mean: {fit_stats_count['sample_mean']:.6f}\n")
                f.write(f"  Sample std: {fit_stats_count['sample_std']:.6f}\n")
                f.write(f"  Weighted mean: {fit_stats_count.get('weighted_mean', 'N/A'):.6f}\n")
                f.write(f"  Weighted variance: {fit_stats_count.get('weighted_variance', 'N/A'):.6f}\n")

            f.write("\nAll Fit Statistics:\n")
            for key, value in fit_stats_count.items():
                f.write(f"  {key}: {value}\n")

        print(f"Gaussian fitting results saved to: {gaussian_results_path}")

        # Print summary statistics
        print(f"\n=== Summary Statistics for {prefix} ===")
        print(f"Total states: {len(mdp.states)}")
        print(f"Terminal states: {len(mdp.terminal_states)}")
        print(f"Actions available: {mdp.num_actions}")
        print(f"Gaussian fit: μ={mu_count:.4f}, σ={sigma_count:.4f}")

        print("Value differences (vs Optimal):")
        for state in sorted(mdp.states)[:5]:
            pe_val = pe_values.get_value(state)
            opt_val = opt_values.get_value(state)
            ql_val = ql_values.get_value(state)
            pe_diff = abs(pe_val - opt_val)
            ql_diff = abs(ql_val - opt_val)
            print(f"  State {state}: Random Policy diff={pe_diff:.6f}, Q-Learning diff={ql_diff:.6f}")

    print(f"\n=== All tests completed! ===")
    print(
        f"Generated plots, CSV files, JSON networks, and Gaussian fit results in '{output_dir}' with prefixes: {prefixes}")

    print("\nGenerated JSON files (MDP Networks):")
    json_files = []
    for i, prefix in enumerate(prefixes):
        json_files.append(f"{prefix}_{i}_mdp_network.json")

    for json_file in json_files:
        print(f"  - {json_file}")

    print("\nGenerated CSV files:")
    csv_files = []
    for i, prefix in enumerate(prefixes):
        csv_files.extend([
            f"{prefix}_{i}_random_policy.csv",
            f"{prefix}_{i}_policy_evaluation_values.csv",
            f"{prefix}_{i}_optimal_values.csv",
            f"{prefix}_{i}_optimal_q_values.csv",
            f"{prefix}_{i}_optimal_policy.csv",
            f"{prefix}_{i}_qlearning_values.csv",
            f"{prefix}_{i}_qlearning_q_values.csv",
            f"{prefix}_{i}_qlearning_policy.csv",
            f"{prefix}_{i}_occupancy_measure.csv",
            f"{prefix}_{i}_reward_count_distribution.csv",
            f"{prefix}_{i}_reward_prob_distribution.csv"
        ])

    for csv_file in csv_files:
        print(f"  - {csv_file}")

    print("\nGenerated Gaussian Fit Result files:")
    gaussian_files = []
    for i, prefix in enumerate(prefixes):
        gaussian_files.append(f"{prefix}_{i}_gaussian_fit_results.txt")

    for gaussian_file in gaussian_files:
        print(f"  - {gaussian_file}")
