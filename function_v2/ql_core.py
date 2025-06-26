import numpy as np
import random
from ql_setup import is_valid_state, is_valid_action

def get_next_task_type_in_sequence(prev_task_type: str) -> str:
    """
    Determines the next task type in the sequence A -> B -> C -> D -> A.
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
        print(f"Warning: Invalid previous task type '{prev_task_type}' provided. Defaulting to 'A'.")
        return 'A'

def epsilon_greedy_action_selection(current_state, q_table, state_to_index, index_to_action, action_to_index, epsilon):
    """
    Epsilon-greedy action selection, enforcing task type rotation (A->B->C->D->A).
    """
    if not is_valid_state(current_state, state_to_index):
        return None

    state_index = state_to_index[current_state]
    prev_task_type = current_state[3]
    required_task_type = get_next_task_type_in_sequence(prev_task_type)

    valid_action_indices = [
        action_to_index[action] for action in action_to_index
        if action[0] == required_task_type
    ]

    if not valid_action_indices:
        print(f"Error: No valid actions found for required task type '{required_task_type}'.")
        return None

    valid_q_values = q_table[state_index, valid_action_indices]

    if random.random() < epsilon:
        selected_valid_action_index_in_list = random.randrange(len(valid_action_indices))
        selected_original_action_index = valid_action_indices[selected_valid_action_index_in_list]
    else:
        selected_valid_action_index_in_list = np.argmax(valid_q_values)
        selected_original_action_index = valid_action_indices[selected_valid_action_index_in_list]

    return index_to_action[selected_original_action_index]

def calculate_reward(current_state: tuple, next_state: tuple, state_to_index: dict) -> float | None:
    """
    Calculates the immediate reward for transitioning from current_state to next_state.
    """
    if not is_valid_state(next_state, state_to_index):
        return None

    simpsons_integral_level, engagement_level, task_completed, prev_task_type = next_state

    reward_cognitive_load = 10.0 if simpsons_integral_level in [2, 3, 4] else -5.0
    engagement_reward_mapping = {1: -2.0, 2: 0.0, 3: 2.0, 4: 5.0, 5: 10.0}
    reward_engagement = engagement_reward_mapping[engagement_level]
    reward_task_completed = 15.0 if task_completed == 1 else -3.0

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
    """
    if not is_valid_state(current_state, state_to_index) or not is_valid_action(action, action_to_index):
        return q_table

    current_state_index = state_to_index[current_state]
    action_index = action_to_index[action]

    if not is_valid_state(next_state, state_to_index):
        print(f"Warning: Invalid next_state {next_state} encountered during Q-table update. Assuming max_next_q = 0.0.")
        max_next_q = 0.0
    else:
        next_state_index = state_to_index[next_state]
        max_next_q = np.max(q_table[next_state_index, :])

    current_q_value = q_table[current_state_index, action_index]
    td_target = reward + discount_factor * max_next_q
    td_error = td_target - current_q_value
    new_q_value = current_q_value + learning_rate * td_error

    q_table[current_state_index, action_index] = new_q_value

    return q_table