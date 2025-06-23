# --- Q-table Analysis Functions ---

def get_optimal_action_for_state(current_state: tuple, q_table: np.ndarray, state_to_index: dict, index_to_action: dict) -> tuple | None:
    """
    Determines the optimal action for a given state based on the learned Q-table.

    The optimal action is the one with the highest Q-value for the specified state.
    This represents the agent's greedy policy after learning.

    Args:
        current_state (tuple): The state tuple for which to find the optimal action.
        q_table (np.ndarray): The learned Q-table.
        state_to_index (dict): Mapping for state lookup (for validation and index).
        index_to_action (dict): Mapping for action lookup (to get action tuple from index).

    Returns:
        tuple: The optimal action tuple.
        None: If the current state is invalid (validated internally).
    """
    # Validate the input state
    if not is_valid_state(current_state, state_to_index):
        return None # Error message printed by is_valid_state

    # Get the index for the current state
    state_index = state_to_index[current_state]

    # Find the index of the action with the maximum Q-value for this state
    optimal_action_index = np.argmax(q_table[state_index, :])

    # Return the corresponding action tuple
    return index_to_action[optimal_action_index]

def get_top_n_actions_for_state(current_state: tuple, q_table: np.ndarray, state_to_index: dict, index_to_action: dict, n: int = 5) -> list | None:
    """
    Retrieves the top N recommended actions and their Q-values for a given state,
    sorted by Q-value in descending order.

    Useful for understanding the policy beyond just the single optimal action.

    Args:
        current_state (tuple): The state tuple for which to find top actions.
        q_table (np.ndarray): The learned Q-table.
        state_to_index (dict): Mapping for state lookup (for validation and index).
        index_to_action (dict): Mapping for action lookup (to get action tuple from index).
        n (int): The number of top actions to return.

    Returns:
        list: A list of tuples, where each tuple is (action_tuple, q_value),
              sorted by Q-value (highest first).
        None: If the current state is invalid (validated internally).
    """
    # Validate the input state
    if not is_valid_state(current_state, state_to_index):
        return None # Error message printed by is_valid_state

    # Get the index for the current state
    state_index = state_to_index[current_state]

    # Get all Q-values for this state
    q_values = q_table[state_index, :]

    # Get indices that sort Q-values in descending order
    sorted_action_indices = np.argsort(q_values)[::-1]

    # Get the top N action indices
    top_n_action_indices = sorted_action_indices[:n]

    # Create list of (action_tuple, q_value) for the top N actions
    top_n_actions_with_q_values = [(index_to_action[i], q_values[i]) for i in top_n_action_indices]

    return top_n_actions_with_q_values
