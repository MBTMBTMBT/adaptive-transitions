from typing import Union, List, Optional
from customisable_env_abs import CustomisableEnvAbs
from mdp_network import MDPNetwork


def deterministic_mdp_sampling(
        env: CustomisableEnvAbs,
        start_states: Optional[Union[List[str], List[int], str, int]] = None,
        max_states: int = float('inf')
) -> 'MDPNetwork':
    """
    Unified deterministic MDP transition sampling function for both string and int state environments.

    Args:
        env: Environment instance that supports encode_state(), decode_state(), step() and reset() methods
        start_states: Optional starting points. Can be:
                     - None: use current environment state after reset
                     - Single state (str or int): single starting point
                     - List of states: multiple starting points
        max_states: Maximum number of states to explore. Default is infinite.

    Returns:
        MDPNetwork: The sampled MDP as a network structure with optional string mapping
    """
    # Verify environment has required methods
    required_methods = ['encode_state', 'decode_state', 'step', 'reset']
    for method in required_methods:
        if not hasattr(env, method):
            raise AttributeError(f"Environment must have {method} method")

    # Reset environment and setup
    env.reset()
    current_state = env.encode_state()
    uses_string_states = isinstance(current_state, str)

    # Store and disable sparse rewards if exists
    original_sparse = None
    if hasattr(env, 'reward_config') and 'sparse' in env.reward_config:
        original_sparse = env.reward_config["sparse"]
        env.reward_config["sparse"] = False

    try:
        # Normalize start_states input
        if start_states is None:
            start_points = [current_state]
        elif isinstance(start_states, (str, int)):
            start_points = [start_states]
        elif isinstance(start_states, list):
            start_points = start_states
        else:
            raise ValueError("start_states must be None, a single state, or a list of states")

        print(f"Starting deterministic MDP sampling from {len(start_points)} start state(s)...")
        print(f"Environment uses {'string' if uses_string_states else 'integer'} states")

        # Initialize data structures
        state_to_int = {}  # Original state -> internal int ID
        int_to_state = {}  # Internal int ID -> original state
        next_state_id = 0
        transitions = {}  # {state_id: {action: {next_state_id: probability}}}
        state_rewards = {}  # {state_id: reward}
        terminal_states = set()
        start_state_ids = []
        visited_states = set()
        exploration_queue = []

        # Process starting points
        for start_state in start_points:
            env.decode_state(start_state)
            actual_start_state = env.encode_state()

            if actual_start_state not in state_to_int:
                state_to_int[actual_start_state] = next_state_id
                int_to_state[next_state_id] = actual_start_state
                next_state_id += 1

            start_state_id = state_to_int[actual_start_state]
            if start_state_id not in start_state_ids:
                start_state_ids.append(start_state_id)

            if actual_start_state not in visited_states:
                exploration_queue.append(actual_start_state)
                visited_states.add(actual_start_state)

        # Main BFS exploration loop
        states_explored = 0
        while exploration_queue and states_explored < max_states:
            current_state = exploration_queue.pop(0)
            states_explored += 1

            if states_explored % 1000 == 0:
                print(f"Explored {states_explored} states, queue size: {len(exploration_queue)}")

            current_state_id = state_to_int[current_state]
            env.decode_state(current_state)

            if current_state_id not in transitions:
                transitions[current_state_id] = {}

            # Try each action
            for action in range(env.action_space.n):
                # If we have already stored a successor for (s,a) we skip:
                if action in transitions[current_state_id]:
                    continue

                # Save state, execute action, restore state
                state_before_action = env.encode_state()
                obs, reward, terminated, truncated, info = env.step(action)
                next_state = env.encode_state()
                env.decode_state(state_before_action)

                # Assign ID to next state if new
                if next_state not in state_to_int:
                    state_to_int[next_state] = next_state_id
                    int_to_state[next_state_id] = next_state
                    next_state_id += 1

                next_state_id_value = state_to_int[next_state]

                # Record transition
                if action not in transitions[current_state_id]:
                    transitions[current_state_id][action] = {}
                transitions[current_state_id][action][next_state_id_value] = 1.0

                # Record reward
                if next_state_id_value not in state_rewards:
                    state_rewards[next_state_id_value] = float(reward)

                # Handle terminal states and queue new states
                if terminated or truncated:
                    terminal_states.add(next_state_id_value)
                elif next_state not in visited_states and len(visited_states) < max_states:
                    exploration_queue.append(next_state)
                    visited_states.add(next_state)

        # Build MDP configuration
        all_state_ids = list(range(next_state_id))
        mdp_config = {
            "num_actions": env.action_space.n,
            "states": all_state_ids,
            "start_states": start_state_ids,
            "terminal_states": list(terminal_states),
            "default_reward": 0.0,
            "state_rewards": {str(sid): reward for sid, reward in state_rewards.items()},
            "transitions": {}
        }

        # Convert transitions to string format for MDPNetwork
        for state_id, state_transitions in transitions.items():
            mdp_config["transitions"][str(state_id)] = {}
            for action, action_transitions in state_transitions.items():
                mdp_config["transitions"][str(state_id)][str(action)] = {
                    str(target_state_id): prob for target_state_id, prob in action_transitions.items()
                }

        # Create MDPNetwork with optional string mapping
        if uses_string_states:
            mdp_network = MDPNetwork(config_data=mdp_config,
                                   int_to_state=int_to_state,
                                   state_to_int=state_to_int)
        else:
            mdp_network = MDPNetwork(config_data=mdp_config)

        # Print statistics
        total_transitions = sum(sum(len(action_dict) for action_dict in state_dict.values())
                                for state_dict in transitions.values())
        print(f"\nDeterministic MDP sampling completed!")
        print(f"Total states: {len(all_state_ids)}")
        print(f"Terminal states: {len(terminal_states)}")
        print(f"Start states: {len(start_state_ids)}")
        print(f"Total transitions: {total_transitions}")
        print(f"String mapping: {'Yes' if uses_string_states else 'No'}")

        return mdp_network

    finally:
        # Restore original sparse reward setting
        if original_sparse is not None and hasattr(env, 'reward_config'):
            env.reward_config["sparse"] = original_sparse
