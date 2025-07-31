import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import os
from mdp_tables import QTable, ValueTable, PolicyTable
from mdp_network import MDPNetwork
from solvers import *


def ensure_output_dir(output_dir: str = "output_plots"):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def get_node_layout(states, layout_type="linear"):
    """Generate node positions for different layout types."""
    if layout_type == "linear":
        return {state: (state * 2, 0) for state in states}
    elif layout_type == "circular":
        n = len(states)
        return {state: (3 * np.cos(2 * np.pi * i / n), 3 * np.sin(2 * np.pi * i / n))
                for i, state in enumerate(states)}
    else:
        return {state: (state * 2, 0) for state in states}


def plot_mdp_values(mdp_network: MDPNetwork, value_table: ValueTable, title: str, save_path: Optional[str] = None):
    """Plot state values on the MDP network graph with color and numerical labels."""
    G = mdp_network.get_graph_copy()

    # Create figure with proper aspect ratio
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get layout
    states = sorted(mdp_network.states)
    pos = get_node_layout(states, "linear")

    # Get values and normalize for colors
    values = [value_table.get_value(state) for state in states]
    min_val, max_val = min(values), max(values)

    if max_val > min_val:
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
    else:
        norm = plt.Normalize(vmin=0, vmax=1)

    # Create colormap
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap

    # Draw edges first (so they appear behind nodes)
    nx.draw_networkx_edges(G, pos, edge_color='#666666', alpha=0.5, arrows=True,
                           arrowsize=15, arrowstyle='->', width=1.5, ax=ax)

    # Draw nodes with colors based on values
    node_colors = [cmap(norm(value)) for value in values]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800,
                           alpha=0.9, edgecolors='black', linewidths=2, ax=ax)

    # Add labels
    for i, state in enumerate(states):
        x, y = pos[state]
        value = values[i]
        is_terminal = mdp_network.is_terminal_state(state)
        is_start = state in mdp_network.start_states

        # State identifier in center of node
        state_label = f"S{state}"
        ax.text(x, y + 0.1, state_label, ha='center', va='center',
                fontweight='bold', fontsize=11, color='black')

        # Value below the node with background for readability
        value_label = f"{value:.3f}"
        ax.text(x, y - 0.4, value_label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='black',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor='black', alpha=0.9))

        # Add state type indicators
        if is_terminal:
            ax.text(x, y - 0.2, "(Term)", ha='center', va='center',
                    fontsize=8, style='italic', color='red')
        if is_start:
            ax.text(x, y + 0.3, "(Start)", ha='center', va='center',
                    fontsize=8, style='italic', color='green')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.1)
    cbar.set_label('State Value', rotation=270, labelpad=20, fontsize=12)

    # Set title and clean up axes
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('equal')
    ax.axis('off')

    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_mdp_policy(mdp_network: MDPNetwork, policy_table: PolicyTable, title: str, save_path: Optional[str] = None):
    """Plot policy on the MDP network graph with action arrows and labels."""
    G = mdp_network.get_graph_copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get layout
    states = sorted(mdp_network.states)
    pos = get_node_layout(states, "linear")

    # Draw edges (transition possibilities)
    nx.draw_networkx_edges(G, pos, edge_color='#CCCCCC', alpha=0.4, arrows=True,
                           arrowsize=12, arrowstyle='->', width=1, ax=ax)

    # Color nodes by state type
    node_colors = []
    for state in states:
        if mdp_network.is_terminal_state(state):
            node_colors.append('#FF9999')  # Light red for terminal
        elif state in mdp_network.start_states:
            node_colors.append('#99FF99')  # Light green for start
        else:
            node_colors.append('#9999FF')  # Light blue for regular

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800,
                           alpha=0.8, edgecolors='black', linewidths=2, ax=ax)

    # Action arrows and labels
    action_vectors = {
        0: (0, 0),  # Stay
        1: (0.6, 0),  # Right
        2: (-0.6, 0),  # Left
        3: (0, 0.6),  # Up
        4: (0, -0.6)  # Down
    }

    action_names = {
        0: "Stay", 1: "Right", 2: "Left", 3: "Up", 4: "Down"
    }

    for state in states:
        x, y = pos[state]
        action = policy_table.get_action(state)
        is_terminal = mdp_network.is_terminal_state(state)
        is_start = state in mdp_network.start_states

        # State label
        state_label = f"S{state}"
        ax.text(x, y + 0.1, state_label, ha='center', va='center',
                fontweight='bold', fontsize=11, color='black')

        # Action label and arrow
        if not is_terminal and action in action_vectors:
            action_name = action_names.get(action, f"A{action}")

            # Action label below node
            ax.text(x, y - 0.4, action_name, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.25", facecolor='yellow',
                              edgecolor='black', alpha=0.8))

            # Draw action arrow if not "Stay"
            if action != 0:
                dx, dy = action_vectors[action]
                ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.15,
                         fc='red', ec='red', linewidth=3, alpha=0.8)

        # State type indicators
        if is_terminal:
            ax.text(x, y - 0.2, "(Term)", ha='center', va='center',
                    fontsize=8, style='italic', color='red')
        if is_start:
            ax.text(x, y + 0.3, "(Start)", ha='center', va='center',
                    fontsize=8, style='italic', color='green')

    # Add legend
    legend_elements = [
        mpatches.Patch(color='#99FF99', alpha=0.8, label='Start State'),
        mpatches.Patch(color='#9999FF', alpha=0.8, label='Regular State'),
        mpatches.Patch(color='#FF9999', alpha=0.8, label='Terminal State'),
        mpatches.FancyArrowPatch((0, 0), (0.3, 0), color='red',
                                 arrowstyle='->', linewidth=3, label='Policy Action')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('equal')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_mdp_q_values(mdp_network: MDPNetwork, q_table: QTable, title: str, save_path: Optional[str] = None):
    """Plot Q-values on the MDP network graph with colors and numerical labels."""
    G = mdp_network.get_graph_copy()

    # Create larger figure for Q-values
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get layout with more spacing for Q-value labels
    states = sorted(mdp_network.states)
    pos = {state: (state * 3, 0) for state in states}  # More spacing

    # Get all Q-values for normalization
    all_q_values = []
    for state in states:
        for action in range(mdp_network.num_actions):
            all_q_values.append(q_table.get_q_value(state, action))

    min_q, max_q = min(all_q_values), max(all_q_values)
    norm = plt.Normalize(vmin=min_q, vmax=max_q)
    cmap = plt.cm.RdYlGn

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='#666666', alpha=0.3, arrows=True,
                           arrowsize=12, arrowstyle='->', width=1, ax=ax)

    # Draw nodes (neutral color since we'll show Q-values separately)
    nx.draw_networkx_nodes(G, pos, node_color='#DDDDDD', node_size=600,
                           alpha=0.7, edgecolors='black', linewidths=2, ax=ax)

    # Q-value positions around each node
    q_positions = {
        0: (0, 0.8),  # Stay (top)
        1: (0.8, 0),  # Right
        2: (-0.8, 0),  # Left
        3: (0, 0.8),  # Up (same as stay if both exist)
        4: (0, -0.8)  # Down
    }

    for state in states:
        x, y = pos[state]
        is_terminal = mdp_network.is_terminal_state(state)
        is_start = state in mdp_network.start_states

        # State label in center
        state_label = f"S{state}"
        ax.text(x, y, state_label, ha='center', va='center',
                fontweight='bold', fontsize=10, color='black')

        # State type indicators
        if is_terminal:
            ax.text(x, y + 1.2, "(Term)", ha='center', va='center',
                    fontsize=8, style='italic', color='red')
        if is_start:
            ax.text(x, y + 1.2, "(Start)", ha='center', va='center',
                    fontsize=8, style='italic', color='green')

        # Q-value labels around the node
        if not is_terminal:
            for action in range(mdp_network.num_actions):
                if action in q_positions:
                    q_value = q_table.get_q_value(state, action)
                    dx, dy = q_positions[action]

                    # Adjust position if action 0 and 3 both exist (avoid overlap)
                    if action == 3 and mdp_network.num_actions > 3:
                        dy = 1.0

                    q_x, q_y = x + dx, y + dy

                    # Color based on Q-value
                    color = cmap(norm(q_value))

                    # Q-value box with action label and value
                    ax.text(q_x, q_y, f"A{action}\n{q_value:.3f}",
                            ha='center', va='center', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                                      edgecolor='black', alpha=0.9))

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.05)
    cbar.set_label('Q-Value', rotation=270, labelpad=20, fontsize=12)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('equal')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_mdp_occupancy(mdp_network: MDPNetwork, occupancy_table: QTable, title: str, save_path: Optional[str] = None):
    """Plot occupancy measures with colors and numerical labels."""
    G = mdp_network.get_graph_copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get layout with spacing
    states = sorted(mdp_network.states)
    pos = {state: (state * 3, 0) for state in states}

    # Get all non-zero occupancy values for normalization
    all_occ_values = []
    for state in states:
        for action in range(mdp_network.num_actions):
            occ_val = occupancy_table.get_q_value(state, action)
            if occ_val > 1e-6:
                all_occ_values.append(occ_val)

    if all_occ_values:
        min_occ, max_occ = min(all_occ_values), max(all_occ_values)
        norm = plt.Normalize(vmin=min_occ, vmax=max_occ)
    else:
        min_occ, max_occ = 0, 1
        norm = plt.Normalize(vmin=0, vmax=1)

    cmap = plt.cm.plasma

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='#666666', alpha=0.3, arrows=True,
                           arrowsize=12, arrowstyle='->', width=1, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='#DDDDDD', node_size=600,
                           alpha=0.7, edgecolors='black', linewidths=2, ax=ax)

    # Occupancy positions around nodes
    occ_positions = {
        0: (0, 0.8),  # Stay (top)
        1: (0.8, 0),  # Right
        2: (-0.8, 0),  # Left
        3: (0, 1.0),  # Up
        4: (0, -0.8)  # Down
    }

    for state in states:
        x, y = pos[state]
        is_terminal = mdp_network.is_terminal_state(state)
        is_start = state in mdp_network.start_states

        # State label
        state_label = f"S{state}"
        ax.text(x, y, state_label, ha='center', va='center',
                fontweight='bold', fontsize=10, color='black')

        # State type indicators
        if is_terminal:
            ax.text(x, y + 1.4, "(Term)", ha='center', va='center',
                    fontsize=8, style='italic', color='red')
        if is_start:
            ax.text(x, y + 1.4, "(Start)", ha='center', va='center',
                    fontsize=8, style='italic', color='green')

        # Occupancy measure labels
        for action in range(mdp_network.num_actions):
            occ_value = occupancy_table.get_q_value(state, action)

            # Only show non-zero occupancy measures
            if occ_value > 1e-6 and action in occ_positions:
                dx, dy = occ_positions[action]
                occ_x, occ_y = x + dx, y + dy

                # Color based on occupancy value
                color = cmap(norm(occ_value))

                # Occupancy box
                ax.text(occ_x, occ_y, f"A{action}\n{occ_value:.3f}",
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                                  edgecolor='black', alpha=0.9))

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.05)
    cbar.set_label('Occupancy Measure', rotation=270, labelpad=20, fontsize=12)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('equal')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


# Unit tests and visualization script
if __name__ == "__main__":
    print("=== MDP Solver Unit Tests with Network Visualization ===\n")

    # Create output directory
    output_dir = ensure_output_dir("output_plots")
    print(f"Output directory: {output_dir}\n")

    # Load the chain MDP using the correct initialization method
    print("Loading MDP...")
    mdp = MDPNetwork(config_path="./mdps/chain.json")
    print(f"Loaded MDP with {len(mdp.states)} states and {mdp.num_actions} actions")
    print(f"States: {mdp.states}")
    print(f"Start states: {mdp.start_states}")
    print(f"Terminal states: {list(mdp.terminal_states)}")
    print(f"State rewards: {mdp.state_rewards}\n")

    # Common parameters
    gamma = 0.9
    theta = 1e-6
    max_iterations = 1000

    # Test 1: Policy Evaluation
    print("=== Test 1: Policy Evaluation ===")

    # Create a simple policy: always choose action 1 (move right)
    test_policy = PolicyTable()
    for state in mdp.states:
        if mdp.is_terminal_state(state):
            test_policy.set_action(state, 0)  # Terminal states can have any action
        else:
            test_policy.set_action(state, 1)  # Always move right for non-terminal states

    print("Testing policy evaluation with right-moving policy...")
    print("Policy actions:", {state: test_policy.get_action(state) for state in mdp.states})

    pe_values = policy_evaluation(mdp, test_policy, gamma, theta, max_iterations)

    print("Policy Evaluation Results:")
    for state in sorted(mdp.states):
        value = pe_values.get_value(state)
        print(f"  State {state}: {value:.6f}")

    # Plot policy evaluation results on network
    plot_mdp_policy(mdp, test_policy, "Policy Evaluation: Right-Moving Policy",
                    os.path.join(output_dir, "policy_evaluation_network.png"))
    plot_mdp_values(mdp, pe_values, "Policy Evaluation: State Values",
                    os.path.join(output_dir, "policy_evaluation_values_network.png"))

    # Test 2: Optimal Value Iteration
    print("\n=== Test 2: Optimal Value Iteration ===")

    print("Computing optimal value iteration...")
    opt_values, opt_q_table, opt_policy = optimal_value_iteration(
        mdp, gamma, theta, max_iterations)

    print("Optimal Value Iteration Results:")
    print("State Values:")
    for state in sorted(mdp.states):
        value = opt_values.get_value(state)
        action = opt_policy.get_action(state)
        print(f"  State {state}: Value={value:.6f}, Optimal Action={action}")

    print("Q-Values:")
    for state in sorted(mdp.states):
        q_vals = []
        for action in range(mdp.num_actions):
            q_val = opt_q_table.get_q_value(state, action)
            q_vals.append(f"A{action}={q_val:.6f}")
        print(f"  State {state}: {', '.join(q_vals)}")

    # Plot optimal value iteration results on network
    plot_mdp_values(mdp, opt_values, "Optimal Value Iteration: State Values",
                    os.path.join(output_dir, "optimal_values_network.png"))
    plot_mdp_policy(mdp, opt_policy, "Optimal Value Iteration: Policy",
                    os.path.join(output_dir, "optimal_policy_network.png"))
    plot_mdp_q_values(mdp, opt_q_table, "Optimal Value Iteration: Q-Values",
                      os.path.join(output_dir, "optimal_q_values_network.png"))

    # Test 3: Q-Learning
    print("\n=== Test 3: Q-Learning ===")

    print("Running Q-Learning...")
    ql_q_table, ql_policy, ql_values = q_learning(
        mdp, alpha=0.1, gamma=gamma, epsilon=0.1,
        num_episodes=5000, max_steps_per_episode=100, seed=42)

    print("Q-Learning Results:")
    print("State Values:")
    for state in sorted(mdp.states):
        value = ql_values.get_value(state)
        action = ql_policy.get_action(state)
        print(f"  State {state}: Value={value:.6f}, Learned Action={action}")

    print("Q-Values:")
    for state in sorted(mdp.states):
        q_vals = []
        for action in range(mdp.num_actions):
            q_val = ql_q_table.get_q_value(state, action)
            q_vals.append(f"A{action}={q_val:.6f}")
        print(f"  State {state}: {', '.join(q_vals)}")

    # Plot Q-Learning results on network
    plot_mdp_values(mdp, ql_values, "Q-Learning: State Values",
                    os.path.join(output_dir, "qlearning_values_network.png"))
    plot_mdp_policy(mdp, ql_policy, "Q-Learning: Learned Policy",
                    os.path.join(output_dir, "qlearning_policy_network.png"))
    plot_mdp_q_values(mdp, ql_q_table, "Q-Learning: Q-Values",
                      os.path.join(output_dir, "qlearning_q_values_network.png"))

    # Test 4: Occupancy Measure
    print("\n=== Test 4: Occupancy Measure ===")

    print("Computing occupancy measure for optimal policy...")
    occupancy = compute_occupancy_measure(mdp, opt_policy, gamma, theta, max_iterations)

    print("Occupancy Measure Results:")
    for state in sorted(mdp.states):
        occ_vals = []
        for action in range(mdp.num_actions):
            occ_val = occupancy.get_q_value(state, action)
            if occ_val > 1e-6:  # Only show non-zero occupancy measures
                occ_vals.append(f"A{action}={occ_val:.6f}")
        if occ_vals:
            print(f"  State {state}: {', '.join(occ_vals)}")
        else:
            print(f"  State {state}: All zero")

    # Plot occupancy measure on network
    plot_mdp_occupancy(mdp, occupancy, "Occupancy Measure: Optimal Policy",
                       os.path.join(output_dir, "occupancy_measure_network.png"))

    # Test 5: Simplified Algorithm Comparison
    print("\n=== Test 5: Algorithm Comparison ===")

    # Create a single comparison plot
    fig, ax = plt.subplots(figsize=(16, 6))

    states = sorted(mdp.states)
    pos = {state: (state * 2, 0) for state in states}
    G = mdp.get_graph_copy()

    # Draw network structure
    nx.draw_networkx_edges(G, pos, edge_color='#666666', alpha=0.3, arrows=True,
                           arrowsize=12, arrowstyle='->', width=1, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='#DDDDDD', node_size=600,
                           alpha=0.7, edgecolors='black', linewidths=2, ax=ax)

    # Add comparison values for each state
    for state in states:
        x, y = pos[state]

        # State label
        ax.text(x, y, f"S{state}", ha='center', va='center',
                fontweight='bold', fontsize=10, color='black')

        # Values from different algorithms
        pe_val = pe_values.get_value(state)
        opt_val = opt_values.get_value(state)
        ql_val = ql_values.get_value(state)

        # Display values below node
        ax.text(x, y - 0.8, f"PE: {pe_val:.3f}\nOpt: {opt_val:.3f}\nQL: {ql_val:.3f}",
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor='black', alpha=0.9))

    ax.set_title('Algorithm Comparison: State Values', fontsize=14, fontweight='bold', pad=20)
    ax.axis('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "algorithm_comparison_network.png"),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total states: {len(mdp.states)}")
    print(f"Terminal states: {len(mdp.terminal_states)}")
    print(f"Actions available: {mdp.num_actions}")

    print("\nValue differences (vs Optimal):")
    for state in sorted(mdp.states):
        pe_val = pe_values.get_value(state)
        opt_val = opt_values.get_value(state)
        ql_val = ql_values.get_value(state)
        pe_diff = abs(pe_val - opt_val)
        ql_diff = abs(ql_val - opt_val)
        print(f"  State {state}: PE diff={pe_diff:.6f}, QL diff={ql_diff:.6f}")

    print(f"\n=== All tests completed! ===")
    print(f"Generated network-based plots in '{output_dir}':")
    print("- policy_evaluation_network.png")
    print("- policy_evaluation_values_network.png")
    print("- optimal_values_network.png")
    print("- optimal_policy_network.png")
    print("- optimal_q_values_network.png")
    print("- qlearning_values_network.png")
    print("- qlearning_policy_network.png")
    print("- qlearning_q_values_network.png")
    print("- occupancy_measure_network.png")
    print("- algorithm_comparison_network.png")
