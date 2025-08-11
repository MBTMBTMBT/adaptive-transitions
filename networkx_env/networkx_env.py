from typing import Dict, Tuple, Optional, Union, Any
import numpy as np
import pygame
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from mdp_network import MDPNetwork


class NetworkXMDPEnvironment(gym.Env):
    def __init__(self,
                 mdp_network: Optional[MDPNetwork] = None,
                 config_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 output_string_states: bool = False,
                 seed: Optional[int] = None):
        super().__init__()

        # Init MDP
        if mdp_network is not None:
            self.mdp = mdp_network
        elif config_path is not None:
            self.mdp = MDPNetwork(config_path=config_path)
        else:
            raise ValueError("Either mdp_network or config_path must be provided")

        self.output_string_states = output_string_states and self.mdp.has_string_mapping

        self.action_space = spaces.Discrete(self.mdp.num_actions)
        self.observation_space = spaces.Discrete(len(self.mdp.states))

        # Rendering
        self.render_mode = render_mode
        self.window_size = 800
        self.window = None
        self.clock = None

        # Random generator (numpy RNG)
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # State
        self.current_state: Optional[Union[int, str]] = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """
        Reset to a random start state.
        Uses numpy Generator self.rng for reproducible randomness.
        """
        super().reset(seed=seed)
        if seed is not None:
            # Re-seed numpy generator for reproducibility
            self.rng = np.random.default_rng(seed)

        self.current_state = self.mdp.sample_start_state(self.rng, as_string=self.output_string_states)

        if self.render_mode == "human":
            self.render()

        return self.current_state, {}

    def step(self, action: int):
        """One environment step using R(s, a, s')."""
        if self.current_state is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        next_state, reward = self.mdp.sample_step(
            self.current_state,
            int(action),
            self.rng,  # use numpy RNG
            as_string=self.output_string_states
        )

        self.current_state = next_state
        terminated = self.mdp.is_terminal_state(next_state)
        truncated = False

        if self.render_mode == "human":
            self.render()

        return next_state, float(reward), bool(terminated), bool(truncated), {}


    def render(self):
        """Pygame render of the underlying graph."""
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

        graph = self.mdp.get_graph_copy()

        # Convert current state to internal id for highlight
        if isinstance(self.current_state, str) and self.mdp.has_string_mapping:
            render_current_state = self.mdp.state_to_int[self.current_state]
        else:
            render_current_state = self.current_state

        # Layout
        graph_area_size = self.window_size - 120
        pos = nx.spring_layout(graph, k=3, iterations=50, seed=42)

        # Scale positions into canvas
        margin = 50
        for node in pos:
            x, y = pos[node]
            pos[node] = (
                margin + (x + 1) * (graph_area_size - 2 * margin) / 2,
                margin + (y + 1) * (graph_area_size - 2 * margin) / 2
            )

        # Draw edges
        for u, v in graph.edges():
            start_pos = pos[u]
            end_pos = pos[v]
            pygame.draw.line(canvas, (128, 128, 128), start_pos, end_pos, 2)

            # Arrow head
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = np.hypot(dx, dy)
            if length > 0:
                dxn, dyn = dx / length * 15, dy / length * 15
                arrow_end = (end_pos[0] - dxn, end_pos[1] - dyn)
                left = (arrow_end[0] + dyn * 0.3, arrow_end[1] - dxn * 0.3)
                right = (arrow_end[0] - dyn * 0.3, arrow_end[1] + dxn * 0.3)
                pygame.draw.polygon(canvas, (128, 128, 128), [end_pos, left, right])

        # Draw nodes
        for node in graph.nodes():
            x, y = pos[node]

            if node == render_current_state:
                color = (255, 0, 0)  # Current
                pygame.draw.circle(canvas, color, (int(x), int(y)), 25)
                pygame.draw.line(canvas, (255, 255, 255), (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), 3)
                pygame.draw.line(canvas, (255, 255, 255), (int(x - 15), int(y + 15)), (int(x + 15), int(y - 15)), 3)
            elif node in self.mdp.terminal_states:
                color = (0, 255, 0)  # Terminal
                pygame.draw.circle(canvas, color, (int(x), int(y)), 25)
                pygame.draw.circle(canvas, (255, 255, 255), (int(x), int(y)), 15, 3)
            elif node in self.mdp.start_states:
                color = (0, 0, 255)  # Start
                pygame.draw.circle(canvas, color, (int(x), int(y)), 25)
                pygame.draw.circle(canvas, (255, 255, 255), (int(x), int(y)), 8)
            else:
                color = (200, 200, 200)  # Regular
                pygame.draw.circle(canvas, color, (int(x), int(y)), 25)

            # Border
            pygame.draw.circle(canvas, (0, 0, 0), (int(x), int(y)), 25, 2)

            # Label (respect mapping)
            font = pygame.font.Font(None, 20)
            if self.mdp.has_string_mapping and node in self.mdp.int_to_state:
                label = str(self.mdp.int_to_state[node])
            else:
                label = str(node)
            if len(label) > 8:
                label = label[:6] + ".."
            text = font.render(label, True, (0, 0, 0))
            canvas.blit(text, text.get_rect(center=(int(x), int(y))))

        # Legend
        legend_x = graph_area_size + 10
        legend_y = 50
        legend_font = pygame.font.Font(None, 20)
        title_font = pygame.font.Font(None, 24)

        title_text = title_font.render("Legend", True, (0, 0, 0))
        canvas.blit(title_text, (legend_x, legend_y))
        legend_y += 35

        legend_items = [
            ((255, 0, 0), lambda x, y: [
                pygame.draw.line(canvas, (255, 255, 255), (x - 8, y - 8), (x + 8, y + 8), 2),
                pygame.draw.line(canvas, (255, 255, 255), (x - 8, y + 8), (x + 8, y - 8), 2)
            ], "Current"),
            ((0, 255, 0), lambda x, y: pygame.draw.circle(canvas, (255, 255, 255), (x, y), 8, 2), "Terminal"),
            ((0, 0, 255), lambda x, y: pygame.draw.circle(canvas, (255, 255, 255), (x, y), 4), "Start"),
            ((200, 200, 200), lambda x, y: None, "Regular"),
        ]

        for color, pattern_func, label in legend_items:
            pygame.draw.circle(canvas, color, (legend_x + 15, legend_y + 10), 12)
            pygame.draw.circle(canvas, (0, 0, 0), (legend_x + 15, legend_y + 10), 12, 1)
            if pattern_func:
                pattern_func(legend_x + 15, legend_y + 10)
            label_text = legend_font.render(label, True, (0, 0, 0))
            canvas.blit(label_text, (legend_x + 35, legend_y + 5))
            legend_y += 30

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # "rgb_array"
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        """Close rendering."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def test_environment(config_path: str, num_episodes: int = 3):
    """Quick manual test with random actions."""
    print(f"\nTesting environment with config: {config_path}")
    env = NetworkXMDPEnvironment(config_path=config_path, render_mode="human")

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        state, info = env.reset()
        print(f"Initial state: {state}")

        total_reward = 0.0
        step_count = 0

        while True:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            step_count += 1
            print(f"Step {step_count}: Action={action}, State={state}->{next_state}, Reward={reward:.3f}")

            state = next_state
            if terminated or truncated:
                print(f"Episode finished. Total reward: {total_reward:.3f}, Steps: {step_count}")
                break
            if step_count > 50:
                print("Episode truncated due to step limit")
                break

    env.close()


if __name__ == "__main__":
    config_files = [
        "../mdp_network/mdps/chain-3.json",
        "../mdp_network/mdps/grid.json",
        "../mdp_network/mdps/branching.json",
    ]
    for config_file in config_files:
        test_environment(config_file, num_episodes=1)
    print("\nAll tests completed!")
