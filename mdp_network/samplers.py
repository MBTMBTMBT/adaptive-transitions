from typing import Dict, Any, Optional, List, Tuple
from mdp_network import MDPNetwork
from customisable_minigrid.mdp import sample_mdp_transitions


def sample_mdp_network_deterministic(env: Any, start_states: Optional[List[str]] = None, max_states: int = float('inf')) -> Tuple[
    MDPNetwork, Dict[int, str], Dict[str, int]]:
    """
    Sample MDP from environment and return as MDPNetwork object with integer-encoded states.

    Args:
        env: Environment instance that supports encode_state(), decode_state(), and step() methods
        start_states: Optional list of encoded state strings to use as starting points.
                     If None, uses current environment state as single starting point.
        max_states: Maximum number of states to explore

    Returns:
        Tuple containing:
        - MDPNetwork: The constructed MDP network object
        - Dict[int, str]: Mapping from integer state IDs to string encodings
        - Dict[str, int]: Mapping from string encodings to integer state IDs
    """
    print("=== Starting MDP Network Sampling ===")

    # Step 1: Sample raw transition table
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

    # Determine start states - improved logic
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
                # Note: This assumes rewards are for reaching states, not for transitions
                if str(next_state_int) not in state_rewards:
                    state_rewards[str(next_state_int)] = reward

    # Create MDP configuration
    mdp_config = {
        "num_actions": num_actions,
        "states": list(range(len(sorted_states))),
        "start_states": mdp_start_states,
        "terminal_states": terminal_states,
        "default_reward": 0.0,
        "state_rewards": state_rewards,
        "transitions": transitions
    }

    # Step 6: Create MDPNetwork object
    mdp_network = MDPNetwork(config_data=mdp_config)

    print(f"Successfully created MDPNetwork with {len(sorted_states)} states")
    print(f"Network has {len(transitions)} states with transitions")

    return mdp_network, int_to_string, string_to_int
