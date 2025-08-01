from typing import Optional

import networkx as nx
from matplotlib import pyplot as plt, patches as mpatches

from mdp_network import MDPNetwork
from mdp_tables import ValueTable, PolicyTable, QTable


def plot_values(mdp_network: MDPNetwork, value_table: ValueTable, title: str, save_path: Optional[str] = None):
    """Plot state values on the MDP network graph with color and numerical labels."""
    G = mdp_network.get_graph_copy()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get layout
    states = sorted(mdp_network.states)
    pos = {state: (state * 2, 0) for state in states}

    # Get values and normalize for colors
    values = [value_table.get_value(state) for state in states]
    min_val, max_val = min(values), max(values)

    if max_val > min_val:
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
    else:
        norm = plt.Normalize(vmin=0, vmax=1)

    cmap = plt.cm.RdYlGn

    # Draw edges first
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
        ax.text(x, y + 0.1, f"S{state}", ha='center', va='center',
                fontweight='bold', fontsize=11, color='black')

        # Value below the node
        ax.text(x, y - 0.4, f"{value:.3f}", ha='center', va='center',
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
    cbar.set_label('Value', rotation=270, labelpad=20, fontsize=12)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_policy(mdp_network: MDPNetwork, policy_table: PolicyTable, title: str, save_path: Optional[str] = None):
    """Plot policy on the MDP network graph, showing action probabilities."""
    G = mdp_network.get_graph_copy()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get layout
    states = sorted(mdp_network.states)
    pos = {state: (state * 2, 0) for state in states}

    # Draw edges
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

    for state in states:
        x, y = pos[state]
        is_terminal = mdp_network.is_terminal_state(state)
        is_start = state in mdp_network.start_states

        # State label
        ax.text(x, y + 0.1, f"S{state}", ha='center', va='center',
                fontweight='bold', fontsize=11, color='black')

        # Policy distribution below node
        if not is_terminal:
            action_probs = policy_table.get_action_probabilities(state)

            # Format probability distribution
            prob_strs = []
            for action in sorted(action_probs.keys()):
                prob = action_probs[action]
                if prob > 0.001:  # Only show non-negligible probabilities
                    prob_strs.append(f"A{action}:{prob:.2f}")

            policy_text = "\n".join(prob_strs) if prob_strs else "A0:1.00"

            ax.text(x, y - 0.5, policy_text, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.25", facecolor='yellow',
                              edgecolor='black', alpha=0.8))

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
        mpatches.Patch(color='#FF9999', alpha=0.8, label='Terminal State')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_q_values(mdp_network: MDPNetwork, q_table: QTable, title: str, save_path: Optional[str] = None):
    """Plot Q-values on the MDP network graph, grouping all actions for each state together."""
    G = mdp_network.get_graph_copy()
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get layout with more spacing
    states = sorted(mdp_network.states)
    pos = {state: (state * 2.5, 0) for state in states}

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

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='#DDDDDD', node_size=600,
                           alpha=0.7, edgecolors='black', linewidths=2, ax=ax)

    for state in states:
        x, y = pos[state]
        is_terminal = mdp_network.is_terminal_state(state)
        is_start = state in mdp_network.start_states

        # State label in center
        ax.text(x, y, f"S{state}", ha='center', va='center',
                fontweight='bold', fontsize=10, color='black')

        # State type indicators
        if is_terminal:
            ax.text(x, y + 0.8, "(Term)", ha='center', va='center',
                    fontsize=8, style='italic', color='red')
        if is_start:
            ax.text(x, y + 0.8, "(Start)", ha='center', va='center',
                    fontsize=8, style='italic', color='green')

        # Q-values grouped together below the node
        if not is_terminal:
            q_value_strs = []

            for action in range(mdp_network.num_actions):
                q_value = q_table.get_q_value(state, action)
                q_value_strs.append(f"A{action}: {q_value:.3f}")

            # Create a single text box with all Q-values
            q_text = "\n".join(q_value_strs)

            # Use average color for the background
            avg_q = np.mean([q_table.get_q_value(state, action) for action in range(mdp_network.num_actions)])
            bg_color = cmap(norm(avg_q))

            ax.text(x, y - 0.8, q_text, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=bg_color,
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
