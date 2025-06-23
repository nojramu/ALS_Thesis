# --- Task type Sequencing ---
import random
import numpy as np
from d0_ql_setup import action_to_index, index_to_action
from d1_ql_validator import is_valid_state, is_valid_action


def get_next_task_type_in_sequence(prev_task_type: str) -> str:
    """
    Determines the next task type in the sequence A -> B -> C -> D -> A.

    This function implements a strict rotation of task types.

    Args:
        prev_task_type (str): The type of the previous task ('A', 'B', 'C', or 'D').
                              It is assumed this input is always one of these four values
                              based on the state space definition.

    Returns:
        str: The next task type in the A->B->C->D->A sequence. Returns 'A' as a default
             for any unexpected input, though this case should ideally not occur
             with proper state handling.
    """
    if prev_task_type == 'A':
        return 'B'
    elif prev_task_type == 'B':
        return 'C'
    elif prev_task_type == 'C':
        return 'D'
    elif prev_task_type == 'D':
        return 'A'
    else:
        # This else block serves as a safeguard for unexpected input,
        # although the state space is defined to only include A, B, C, D.
        print(f"Warning: Invalid previous task type '{prev_task_type}' provided to get_next_task_type_in_sequence. Defaulting to 'A'.")
        return 'A'

# --- Core Q-Learning Functions ---

def epsilon_greedy_action_selection(current_state: tuple, q_table: np.ndarray, state_to_index: dict, index_to_action: dict, epsilon: float) -> tuple | None:
    """
    Selects an action using the epsilon-greedy policy, strictly enforcing the
    task type rotation (A->B->C->D->A).

    This policy balances exploration (trying random actions) and exploitation
    (choosing the action with the highest learned Q-value) only among actions
    that match the required next task type based on the current state's
    previous task type.

    Args:
        current_state (tuple): The agent's current state tuple.
        q_table (np.ndarray): The learned Q-table.
        state_to_index (dict): Mapping from state tuple to index (for state lookup).
        index_to_action (dict): Mapping from index to action tuple (for action lookup).
        epsilon (float): The probability of choosing a random action (exploration rate, 0 to 1).

    Returns:
        tuple: The selected action tuple (task_type, difficulty).
        None: If the current state is invalid (validated internally) or no valid
              actions are found for the required task type.
    """
    # Validate the current state
    if not is_valid_state(current_state, state_to_index):
        return None # Error message printed by is_valid_state

    # Get the index corresponding to the current state
    state_index = state_to_index[current_state]

    # Determine the required next task type based on the current state's previous task type
    # The previous task type is the 4th element (index 3) of the state tuple
    prev_task_type = current_state[3]
    required_task_type = get_next_task_type_in_sequence(prev_task_type)

    # Filter actions to include only those with the required task type
    # We need the original indices of these valid actions to access the Q-table
    valid_action_indices = [
        action_to_index[action] for action in action_to_index
        if action[0] == required_task_type # Check the task type component of the action tuple
    ]

    if not valid_action_indices:
        print(f"Error: No valid actions found for the required task type '{required_task_type}'.")
        return None # Should not happen with the defined action space, but good for safety

    # Get the Q-values for the current state, but only for the valid actions
    valid_q_values = q_table[state_index, valid_action_indices]

    # Decide between exploration and exploitation among the valid actions
    if random.random() < epsilon:
        # Exploration: Choose a random index from the list of valid action indices
        selected_valid_action_index_in_list = random.randrange(len(valid_action_indices))
        # Get the original index from the filtered list
        selected_original_action_index = valid_action_indices[selected_valid_action_index_in_list]
    else:
        # Exploitation: Choose the index within the valid_q_values array
        # that corresponds to the max Q-value among valid actions
        selected_valid_action_index_in_list = np.argmax(valid_q_values)
        # Get the original index from the filtered list
        selected_original_action_index = valid_action_indices[selected_valid_action_index_in_list]


    # Return the corresponding action tuple using the original index
    return index_to_action[selected_original_action_index]

def calculate_reward(current_state: tuple, next_state: tuple, state_to_index: dict) -> float | None:
    """
    Calculates the immediate reward for transitioning from current_state to next_state.

    The reward is based on the characteristics of the *next* state, encouraging:
    - Balanced cognitive load (mid-range simpsons_integral_level: 2, 3, 4).
    - Higher engagement_level (1 is low, 5 is high).
    - Completed tasks (task_completed = 1).

    Args:
        current_state (tuple): The state before the transition. Not used directly in this
                                reward calculation logic, but included for context.
        next_state (tuple): The state after the transition.
        state_to_index (dict): Mapping from state tuple to index (for next_state validation).

    Returns:
        float: The calculated reward.
        None: If the next state is invalid (validated internally).
    """
    # Validate the next state
    if not is_valid_state(next_state, state_to_index):
        return None # Error message printed by is_valid_state

    # Unpack next state components
    simpsons_integral_level, engagement_level, task_completed, prev_task_type = next_state

    # Reward component for cognitive load: higher for mid-range, penalty for extremes
    reward_cognitive_load: float
    if simpsons_integral_level in [2, 3, 4]:
        reward_cognitive_load = 10.0
    else:
        reward_cognitive_load = -5.0

    # Reward component for engagement level: increasing reward for higher levels
    engagement_reward_mapping = {1: -2.0, 2: 0.0, 3: 2.0, 4: 5.0, 5: 10.0}
    # Safe lookup because is_valid_state ensures engagement_level is 1-5
    reward_engagement: float = engagement_reward_mapping[engagement_level]

    # Reward component for task completion: positive for completed, penalty for not
    # Safe check because is_valid_state ensures task_completed is 0 or 1
    reward_task_completed: float = 15.0 if task_completed == 1 else -3.0

    # Combine rewards with weights (adjust weights to prioritize factors)
    weight_cognitive_load = 0.4
    weight_engagement = 0.4
    weight_task_completed = 0.2

    total_reward = (weight_cognitive_load * reward_cognitive_load +
                    weight_engagement * reward_engagement +
                    weight_task_completed * reward_task_completed)

    return total_reward


def update_q_table(q_table: np.ndarray,
                   current_state: tuple,
                   action: tuple,
                   reward: float,
                   next_state: tuple,
                   learning_rate: float,
                   discount_factor: float,
                   state_to_index: dict,
                   action_to_index: dict) -> np.ndarray:
    """
    Performs a single step of the Q-learning update rule.

    This function implements the core Q-learning equation to update the estimated
    value of taking the chosen action in the current state.

    Args:
        q_table (np.ndarray): The Q-table to update.
        current_state (tuple): The state before the action.
        action (tuple): The action taken.
        reward (float): The immediate reward received.
        next_state (tuple): The state after the action.
        learning_rate (float): The learning rate (alpha, 0 to 1).
        discount_factor (float): The discount factor (gamma, 0 to 1).
        state_to_index (dict): Mapping for state lookup (for validation and index).
        action_to_index (dict): Mapping for action lookup (for validation and index).

    Returns:
        np.ndarray: The updated Q-table. Returns the original table if current_state or action is invalid.
    """
    # Validate current state and action
    if not is_valid_state(current_state, state_to_index) or not is_valid_action(action, action_to_index):
        return q_table # Error messages printed by validation functions

    # Get indices for Q-table access
    current_state_index = state_to_index[current_state]
    action_index = action_to_index[action]

    # Determine the maximum Q-value for the next state
    # If next_state is invalid, assume no future reward
    if not is_valid_state(next_state, state_to_index):
        print(f"Warning: Invalid next_state {next_state} encountered during Q-table update. Assuming max_next_q = 0.0.")
        max_next_q = 0.0
    else:
        # Get index for next state and find max Q-value across all actions from there
        next_state_index = state_to_index[next_state]
        max_next_q = np.max(q_table[next_state_index, :])

    # Get the current Q-value for the state-action pair
    current_q_value = q_table[current_state_index, action_index]

    # Q-learning update formula: Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]
    td_target = reward + discount_factor * max_next_q
    td_error = td_target - current_q_value
    new_q_value = current_q_value + learning_rate * td_error

    # Update the Q-table
    q_table[current_state_index, action_index] = new_q_value

    return q_table