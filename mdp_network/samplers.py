from typing import Dict, Any, Optional, List, Tuple, Union
from mdp_network import MDPNetwork
from customisable_minigrid.mdp import sample_mdp_transitions


def sample_mdp_network_deterministic(
        env: Any,
        start_states: Optional[Union[List[str], List[int]]] = None,
        max_states: int = float('inf'),
        use_encoding: bool = True
) -> Tuple[MDPNetwork, Optional[Dict[int, str]], Optional[Dict[str, int]]]:
    """
    Sample MDP from environment and return as MDPNetwork object with integer-encoded states.

    Args:
        env: Environment instance that supports encode_state(), decode_state(), and step() methods
               OR supports integer state access directly
        start_states: Optional list of state identifiers to use as starting points.
                     - If use_encoding=True: List of encoded state strings
                     - If use_encoding=False: List of integer state IDs
                     If None, uses current environment state as single starting point.
        max_states: Maximum number of states to explore
        use_encoding: If True, use string encoding/decoding methods. If False, use integer states directly.

    Returns:
        Tuple containing:
        - MDPNetwork: The constructed MDP network object
        - Dict[int, str] or None: Mapping from integer state IDs to string encodings (None if use_encoding=False)
        - Dict[str, int] or None: Mapping from string encodings to integer state IDs (None if use_encoding=False)
    """
    print(f"=== Starting MDP Network Sampling ({'Encoding' if use_encoding else 'Integer'} Mode) ===")

    if use_encoding:
        # Step 1: Sample raw transition table using encoding mode
        raw_transition_table = sample_mdp_transitions(env, start_states, max_states)

        if not raw_transition_table:
            raise ValueError("No transitions were sampled from the environment")

        print(f"\n=== Building State Mapping ===")

        # Step 2: Collect all unique states and create integer mapping
        all_states = set()

        # Collect states from transition table
        for state in raw_transition_table.keys():
            all_states.add(state)
            for action_dict in raw_transition_table[state].values():
                for next_state in action_dict.keys():
                    all_states.add(next_state)

        # Create sorted list for consistent ordering
        sorted_states = sorted(list(all_states))

        # Create bidirectional mappings
        int_to_string = {i: state_str for i, state_str in enumerate(sorted_states)}
        string_to_int = {state_str: i for i, state_str in enumerate(sorted_states)}

        print(f"Created state mapping for {len(sorted_states)} unique states")

        # Step 3: Determine start states and terminal states
        print(f"\n=== Analyzing State Properties ===")

        # Determine start states
        if start_states is None:
            # Single start state - use the environment's current initial state
            try:
                current_initial = env.encode_state()
                if current_initial in string_to_int:
                    mdp_start_states = [string_to_int[current_initial]]
                else:
                    # Fallback to first state in sorted order
                    mdp_start_states = [0]
                    print(f"Warning: Initial environment state not found in sampled states, using state 0")
            except Exception as e:
                print(f"Warning: Could not get current environment state: {e}")
                mdp_start_states = [0]
        else:
            # Multiple start states - map provided start states to integers
            mdp_start_states = []
            for start_state_str in start_states:
                if start_state_str in string_to_int:
                    mdp_start_states.append(string_to_int[start_state_str])
                else:
                    print(f"Warning: Start state {start_state_str[:30]}... not found in sampled states")

        # Identify terminal states (states with no outgoing transitions)
        terminal_states = []
        for state_str in sorted_states:
            state_int = string_to_int[state_str]
            if (state_str not in raw_transition_table or
                    not raw_transition_table[state_str] or
                    all(not action_dict for action_dict in raw_transition_table[state_str].values())):
                terminal_states.append(state_int)

        print(f"Identified {len(mdp_start_states)} start states: {mdp_start_states}")
        print(f"Identified {len(terminal_states)} terminal states")

        # Step 4: Determine number of actions
        all_actions = set()
        for state_dict in raw_transition_table.values():
            all_actions.update(state_dict.keys())

        num_actions = max(all_actions) + 1 if all_actions else 0
        print(f"Detected {num_actions} actions: {sorted(all_actions) if all_actions else 'None'}")

        # Step 5: Build MDPNetwork configuration
        print(f"\n=== Constructing MDP Network ===")

        # Create transitions dictionary with integer states
        transitions = {}
        state_rewards = {}

        for state_str, state_transitions in raw_transition_table.items():
            state_int = string_to_int[state_str]
            transitions[str(state_int)] = {}

            for action, next_state_rewards in state_transitions.items():
                transitions[str(state_int)][str(action)] = {}

                for next_state_str, reward in next_state_rewards.items():
                    next_state_int = string_to_int[next_state_str]

                    # For deterministic environments, probability is 1.0
                    transitions[str(state_int)][str(action)][str(next_state_int)] = 1.0

                    # Store reward as state reward (using reward for reaching next_state)
                    if str(next_state_int) not in state_rewards:
                        state_rewards[str(next_state_int)] = reward

        # Use the original mappings
        mapping_int_to_string = int_to_string
        mapping_string_to_int = string_to_int

    else:
        # Integer mode - sample directly using integer states
        print("Using integer state mode - sampling transitions directly")

        # Sample transitions using integer states
        int_transition_table = _sample_mdp_transitions_int_mode(env, start_states, max_states)

        if not int_transition_table:
            raise ValueError("No transitions were sampled from the environment")

        print(f"\n=== Analyzing Integer State Properties ===")

        # Collect all unique integer states
        all_int_states = set()
        for state in int_transition_table.keys():
            all_int_states.add(state)
            for action_dict in int_transition_table[state].values():
                for next_state in action_dict.keys():
                    all_int_states.add(next_state)

        sorted_int_states = sorted(list(all_int_states))
        print(
            f"Found {len(sorted_int_states)} unique integer states: {sorted_int_states[:10]}{'...' if len(sorted_int_states) > 10 else ''}")

        # Determine start states
        if start_states is None:
            # Single start state - use current environment state
            try:
                current_state = getattr(env, 'current_state', 0)  # Assuming environment has current_state attribute
                mdp_start_states = [current_state] if current_state in all_int_states else [min(all_int_states)]
            except:
                mdp_start_states = [min(all_int_states)]
        else:
            # Multiple start states - filter to ensure they exist in sampled states
            mdp_start_states = [s for s in start_states if s in all_int_states]
            if not mdp_start_states:
                print("Warning: None of the provided start states found in sampled states, using minimum state")
                mdp_start_states = [min(all_int_states)]

        # Identify terminal states
        terminal_states = []
        for state in sorted_int_states:
            if (state not in int_transition_table or
                    not int_transition_table[state] or
                    all(not action_dict for action_dict in int_transition_table[state].values())):
                terminal_states.append(state)

        print(f"Identified {len(mdp_start_states)} start states: {mdp_start_states}")
        print(f"Identified {len(terminal_states)} terminal states")

        # Determine number of actions
        all_actions = set()
        for state_dict in int_transition_table.values():
            all_actions.update(state_dict.keys())

        num_actions = max(all_actions) + 1 if all_actions else 0
        print(f"Detected {num_actions} actions: {sorted(all_actions) if all_actions else 'None'}")

        # Build transitions dictionary
        transitions = {}
        state_rewards = {}

        for state, state_transitions in int_transition_table.items():
            transitions[str(state)] = {}

            for action, next_state_rewards in state_transitions.items():
                transitions[str(state)][str(action)] = {}

                for next_state, reward in next_state_rewards.items():
                    # For deterministic environments, probability is 1.0
                    transitions[str(state)][str(action)][str(next_state)] = 1.0

                    # Store reward as state reward
                    if str(next_state) not in state_rewards:
                        state_rewards[str(next_state)] = reward

        # No string mappings in integer mode
        mapping_int_to_string = None
        mapping_string_to_int = None

    # Create MDP configuration
    mdp_config = {
        "num_actions": num_actions,
        "states": sorted_int_states if not use_encoding else list(range(len(sorted_states))),
        "start_states": mdp_start_states,
        "terminal_states": terminal_states,
        "default_reward": 0.0,
        "state_rewards": state_rewards,
        "transitions": transitions
    }

    # Create MDPNetwork object
    mdp_network = MDPNetwork(config_data=mdp_config)

    total_states = len(sorted_int_states) if not use_encoding else len(sorted_states)
    print(f"Successfully created MDPNetwork with {total_states} states")
    print(f"Network has {len(transitions)} states with transitions")

    return mdp_network, mapping_int_to_string, mapping_string_to_int


def _sample_mdp_transitions_int_mode(env: Any, start_states: Optional[List[int]] = None,
                                     max_states: int = float('inf')) -> Dict[int, Dict[int, Dict[int, float]]]:
    """
    Sample MDP transitions using integer states directly (no encoding/decoding).

    Args:
        env: Environment with integer state space
        start_states: Optional list of integer start states
        max_states: Maximum states to explore

    Returns:
        Dict[int, Dict[int, Dict[int, float]]]: Transition table with integer states
    """
    # Check if environment supports direct integer state access
    required_methods = ['set_state', 'get_state', 'step']  # Assume these methods for integer mode
    for method in required_methods:
        if not hasattr(env, method):
            raise AttributeError(f"Environment must have {method} method for integer mode")

    # Store original sparse reward setting if exists
    original_sparse = None
    if hasattr(env, 'reward_config') and 'sparse' in env.reward_config:
        original_sparse = env.reward_config["sparse"]
        env.reward_config["sparse"] = False

    # Initialize data structures
    combined_table = {}
    global_visited_states = set()
    total_states_explored = 0

    try:
        # Determine starting points
        if start_states is None:
            # Single start state mode
            initial_state = env.get_state()
            start_points = [initial_state]
            print(f"Starting single-point integer sampling from state: {initial_state}")
        else:
            # Multiple start states mode
            start_points = start_states
            print(f"Starting multi-point integer sampling from {len(start_points)} start states")

        # Process each starting point
        for start_idx, start_state in enumerate(start_points):
            if total_states_explored >= max_states:
                print(f"Reached maximum state limit ({max_states}), stopping exploration")
                break

            if start_states is not None:
                print(f"\n--- Sampling from start state {start_idx + 1}/{len(start_points)}: {start_state} ---")

                # Set environment to start state
                try:
                    env.set_state(start_state)
                except Exception as e:
                    print(f"Failed to set start state {start_state}: {e}")
                    continue

            # Initialize local exploration structures
            local_transition_table = {}
            local_visited_states = set()
            exploration_queue = []
            local_states_explored = 0

            # Add starting state to exploration
            current_start = env.get_state()
            exploration_queue.append(current_start)
            local_visited_states.add(current_start)

            # Calculate remaining budget
            remaining_budget = max_states - total_states_explored

            # Main BFS loop
            while exploration_queue and local_states_explored < remaining_budget:
                current_state = exploration_queue.pop(0)
                local_states_explored += 1

                if local_states_explored % 100 == 0:
                    print(f"Explored {local_states_explored} states locally, queue size: {len(exploration_queue)}")

                # Set environment to current state
                env.set_state(current_state)

                # Initialize transition entry
                if current_state not in local_transition_table:
                    local_transition_table[current_state] = {}

                # Try each possible action
                valid_actions = []
                for action in range(env.action_space.n):
                    # Save current state
                    state_before_action = env.get_state()

                    try:
                        # Execute action
                        obs, reward, terminated, truncated, info = env.step(action)
                        next_state = env.get_state()

                        # Record transition
                        if action not in local_transition_table[current_state]:
                            local_transition_table[current_state][action] = {}

                        local_transition_table[current_state][action][next_state] = float(reward)

                        # Add next state to exploration if not visited and not terminal
                        if (next_state not in local_visited_states and
                                not terminated and
                                not truncated and
                                len(local_visited_states) < remaining_budget):
                            exploration_queue.append(next_state)
                            local_visited_states.add(next_state)

                        valid_actions.append(action)

                    except Exception as e:
                        print(f"Action {action} failed from state {current_state}: {e}")
                        continue

                    finally:
                        # Restore state
                        try:
                            env.set_state(state_before_action)
                        except Exception as e:
                            print(f"Failed to restore state: {e}")
                            break

                if not valid_actions:
                    print(f"Warning: No valid actions found for state {current_state}")

            # Merge local results
            for state, transitions in local_transition_table.items():
                if state not in combined_table:
                    combined_table[state] = {}
                for action, next_states in transitions.items():
                    if action not in combined_table[state]:
                        combined_table[state][action] = {}
                    combined_table[state][action].update(next_states)

            # Update global tracking
            global_visited_states.update(local_visited_states)
            total_states_explored = len(combined_table)

            print(f"Completed start point {start_idx + 1}: {local_states_explored} local states explored")
            print(f"Total unique states discovered so far: {total_states_explored}")

        # Final statistics
        total_transitions = sum(
            sum(len(action_dict) for action_dict in state_dict.values())
            for state_dict in combined_table.values()
        )

        print(f"\nInteger mode MDP sampling completed!")
        print(f"Total start points processed: {len(start_points)}")
        print(f"Total unique states discovered: {len(combined_table)}")
        print(f"Total transitions recorded: {total_transitions}")

        return combined_table

    except KeyboardInterrupt:
        print(f"\nInteger sampling interrupted by user.")
        print(f"Partial results: {len(combined_table)} states sampled")
        return combined_table

    except Exception as e:
        print(f"Error during integer sampling: {e}")
        return combined_table

    finally:
        # Restore original sparse reward setting
        if original_sparse is not None:
            env.reward_config["sparse"] = original_sparse
            print(f"Restored original sparse reward setting: {original_sparse}")
