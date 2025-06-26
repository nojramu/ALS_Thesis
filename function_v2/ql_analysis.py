import numpy as np
from ql_setup import is_valid_state

def get_optimal_action_for_state(current_state: tuple, q_table: np.ndarray, state_to_index: dict, index_to_action: dict) -> tuple | None:
    """
    Determines the optimal action for a given state based on the learned Q-table.

    Args:
        current_state (tuple): The state tuple for which to find the optimal action.
        q_table (np.ndarray): The learned Q-table.
        state_to_index (dict): Mapping for state lookup (for validation and index).
        index_to_action (dict): Mapping for action lookup (to get action tuple from index).

    Returns:
        tuple: The optimal action tuple.
        None: If the current state is invalid.
    """
    if not is_valid_state(current_state, state_to_index):
        return None
    state_index = state_to_index[current_state]
    optimal_action_index = np.argmax(q_table[state_index, :])
    return index_to_action[optimal_action_index]

def get_top_n_actions_for_state(current_state: tuple, q_table: np.ndarray, state_to_index: dict, index_to_action: dict, n: int = 5) -> list | None:
    """
    Retrieves the top N recommended actions and their Q-values for a given state,
    sorted by Q-value in descending order.

    Args:
        current_state (tuple): The state tuple for which to find top actions.
        q_table (np.ndarray): The learned Q-table.
        state_to_index (dict): Mapping for state lookup (for validation and index).
        index_to_action (dict): Mapping for action lookup (to get action tuple from index).
        n (int): The number of top actions to return.

    Returns:
        list: A list of tuples, where each tuple is (action_tuple, q_value),
              sorted by Q-value (highest first).
        None: If the current state is invalid.
    """
    if not is_valid_state(current_state, state_to_index):
        return None
    state_index = state_to_index[current_state]
    q_values = q_table[state_index, :]
    sorted_action_indices = np.argsort(q_values)[::-1]
    top_n_action_indices = sorted_action_indices[:n]
    top_n_actions_with_q_values = [(index_to_action[i], q_values[i]) for i in top_n_action_indices]
    return top_n_actions_with_q_values

def print_policy_for_state(current_state, q_table, state_to_index, index_to_action, n=5):
    """
    Prints the top-N actions and their Q-values for a given state.
    """
    top_actions = get_top_n_actions_for_state(current_state, q_table, state_to_index, index_to_action, n)
    if top_actions is None:
        print("Invalid state.")
        return
    print(f"Top {n} actions for state {current_state}:")
    for action, q_value in top_actions:
        print(f"  Action: {action}, Q-value: {q_value:.3f}")

def plot_q_table_heatmap(q_table, filename="image/qtable_heatmap.png"):
    """
    Plots a heatmap of the Q-table and saves it to a file.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(12, 8))
    sns.heatmap(q_table, cmap="viridis")
    plt.title("Q-table Heatmap")
    plt.xlabel("Action Index")
    plt.ylabel("State Index")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()