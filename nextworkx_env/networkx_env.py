import json
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pygame
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from mdp_network import MDPNetwork


class NetworkXMDPEnvironment(gym.Env):
    """
    A custom Gymnasium environment that uses MDPNetwork to represent MDP.
    The environment is configured through JSON files or MDPNetwork objects.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, mdp_network: Optional[MDPNetwork] = None, config_path: Optional[str] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize the NetworkX MDP environment.

        Args:
            mdp_network: MDPNetwork object
            config_path: Path to the JSON configuration file (if mdp_network is None)
            render_mode: Rendering mode ("human", "rgb_array", or None)
        """
        super().__init__()

        # Initialize MDP network
        if mdp_network is not None:
            self.mdp = mdp_network
        elif config_path is not None:
            self.mdp = MDPNetwork(config_path=config_path)
        else:
            raise ValueError("Either mdp_network or config_path must be provided")

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.mdp.num_actions)
        self.observation_space = spaces.Discrete(len(self.mdp.states))

        # Initialize rendering
        self.render_mode = render_mode
        self.window_size = 800
        self.window = None
        self.clock = None

        # Current state
        self.current_state = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[int, Dict]:
        """Reset the environment to a random start state."""
        super().reset(seed=seed)

        # Choose random start state
        self.current_state = self.mdp.sample_start_state(self.np_random)

        if self.render_mode == "human":
            self.render()

        return self.current_state, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.current_state is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        # Sample next state
        next_state = self.mdp.sample_next_state(self.current_state, action, self.np_random)

        # Update current state
        self.current_state = next_state

        # Get reward for entering the new state
        reward = self.mdp.get_state_reward(next_state)

        # Check if terminal state
        terminated = self.mdp.is_terminal_state(next_state)
        truncated = False  # No time limit in this environment

        if self.render_mode == "human":
            self.render()

        return next_state, reward, terminated, truncated, {}

    def render(self):
        """Render the environment using Pygame."""
        if self.render_mode is None:
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background

        # Get graph for rendering
        graph = self.mdp.get_graph_copy()

        # Calculate layout using spring layout (leave space for legend)
        graph_area_size = self.window_size - 120  # Reserve space for legend
        pos = nx.spring_layout(graph, k=3, iterations=50, seed=42)

        # Scale positions to graph area
        margin = 50
        for node in pos:
            x, y = pos[node]
            pos[node] = (
                margin + (x + 1) * (graph_area_size - 2 * margin) / 2,
                margin + (y + 1) * (graph_area_size - 2 * margin) / 2
            )

        # Draw edges (transitions)
        for edge in graph.edges():
            start_pos = pos[edge[0]]
            end_pos = pos[edge[1]]
            pygame.draw.line(canvas, (128, 128, 128), start_pos, end_pos, 2)

            # Draw arrow head
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = np.sqrt(dx ** 2 + dy ** 2)
            if length > 0:
                dx_norm = dx / length * 15
                dy_norm = dy / length * 15
                arrow_end = (end_pos[0] - dx_norm, end_pos[1] - dy_norm)
                arrow_left = (arrow_end[0] + dy_norm * 0.3, arrow_end[1] - dx_norm * 0.3)
                arrow_right = (arrow_end[0] - dy_norm * 0.3, arrow_end[1] + dx_norm * 0.3)
                pygame.draw.polygon(canvas, (128, 128, 128), [end_pos, arrow_left, arrow_right])

        # Draw nodes (states) with different patterns for B&W compatibility
        for node in graph.nodes():
            x, y = pos[node]

            # Choose color and pattern based on state type
            if node == self.current_state:
                color = (255, 0, 0)  # Red
                # Draw filled circle with cross pattern
                pygame.draw.circle(canvas, color, (int(x), int(y)), 25)
                pygame.draw.line(canvas, (255, 255, 255), (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), 3)
                pygame.draw.line(canvas, (255, 255, 255), (int(x - 15), int(y + 15)), (int(x + 15), int(y - 15)), 3)
            elif node in self.mdp.terminal_states:
                color = (0, 255, 0)  # Green
                # Draw filled circle with checkmark pattern
                pygame.draw.circle(canvas, color, (int(x), int(y)), 25)
                pygame.draw.circle(canvas, (255, 255, 255), (int(x), int(y)), 15, 3)
            elif node in self.mdp.start_states:
                color = (0, 0, 255)  # Blue
                # Draw filled circle with dot pattern
                pygame.draw.circle(canvas, color, (int(x), int(y)), 25)
                pygame.draw.circle(canvas, (255, 255, 255), (int(x), int(y)), 8)
            else:
                color = (200, 200, 200)  # Gray
                # Draw simple filled circle
                pygame.draw.circle(canvas, color, (int(x), int(y)), 25)

            # Draw black border for all nodes
            pygame.draw.circle(canvas, (0, 0, 0), (int(x), int(y)), 25, 2)

            # Draw state number
            font = pygame.font.Font(None, 24)
            text = font.render(str(node), True, (0, 0, 0))
            text_rect = text.get_rect(center=(int(x), int(y)))
            canvas.blit(text, text_rect)

        # Draw legend
        legend_x = graph_area_size + 10
        legend_y = 50
        legend_font = pygame.font.Font(None, 20)
        title_font = pygame.font.Font(None, 24)

        # Legend title
        title_text = title_font.render("Legend", True, (0, 0, 0))
        canvas.blit(title_text, (legend_x, legend_y))
        legend_y += 35

        # Legend items: (color, pattern_func, label)
        legend_items = [
            ((255, 0, 0), lambda x, y: [
                pygame.draw.line(canvas, (255, 255, 255), (x - 8, y - 8), (x + 8, y + 8), 2),
                pygame.draw.line(canvas, (255, 255, 255), (x - 8, y + 8), (x + 8, y - 8), 2)
            ], "Current"),
            ((0, 255, 0), lambda x, y: pygame.draw.circle(canvas, (255, 255, 255), (x, y), 8, 2), "Terminal"),
            ((0, 0, 255), lambda x, y: pygame.draw.circle(canvas, (255, 255, 255), (x, y), 4), "Start"),
            ((200, 200, 200), lambda x, y: None, "Regular")
        ]

        for color, pattern_func, label in legend_items:
            # Draw colored circle
            pygame.draw.circle(canvas, color, (legend_x + 15, legend_y + 10), 12)
            pygame.draw.circle(canvas, (0, 0, 0), (legend_x + 15, legend_y + 10), 12, 1)

            # Draw pattern
            if pattern_func:
                pattern_func(legend_x + 15, legend_y + 10)

            # Draw label
            label_text = legend_font.render(label, True, (0, 0, 0))
            canvas.blit(label_text, (legend_x + 35, legend_y + 5))
            legend_y += 30

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        """Close the rendering window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def test_environment(config_path: str, num_episodes: int = 3):
    """Test the environment with random actions."""
    print(f"\nTesting environment with config: {config_path}")

    # Method 1: Create environment directly from config file
    env = NetworkXMDPEnvironment(config_path=config_path, render_mode="human")

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        state, info = env.reset()
        print(f"Initial state: {state}")

        total_reward = 0
        step_count = 0

        while True:
            # Choose random action
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1

            print(f"Step {step_count}: Action={action}, State={state}->{next_state}, Reward={reward:.3f}")

            state = next_state

            if terminated or truncated:
                print(f"Episode finished. Total reward: {total_reward:.3f}, Steps: {step_count}")
                break

            if step_count > 50:  # Prevent infinite loops
                print("Episode truncated due to step limit")
                break

    env.close()


if __name__ == "__main__":
    config_files = ["../mdp_network/mdps/chain-3.json", "../mdp_network/mdps/grid.json", "../mdp_network/mdps/branching.json",]
    for config_file in config_files:
        test_environment(config_file, num_episodes=1)
    print("\nAll tests completed!")
