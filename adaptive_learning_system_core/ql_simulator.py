import numpy as np
import random
from ql_setup import is_valid_state, is_valid_action
from ql_core import calculate_reward

def simulate_cognitive_load_change(current_simpsons_level: int, chosen_difficulty: int, current_completed: int) -> int:
    """
    Simulates the change in the simpsons_integral_level (cognitive load)
    based on the chosen task's difficulty and the outcome of the previous task.
    """
    simpsons_change = (chosen_difficulty - 5) * 0.2
    if current_completed == 1:
        simpsons_change -= 0.5
    next_simpsons_level_float = current_simpsons_level + simpsons_change
    return int(np.clip(round(next_simpsons_level_float), 1, 5))

def simulate_engagement_change(current_engagement: int, chosen_difficulty: int, current_completed: int) -> int:
    """
    Simulates the change in the engagement_level based on the chosen task's
    difficulty and the outcome of the previous task.
    """
    engagement_change = 0
    if chosen_difficulty > 7 and current_completed == 1:
        engagement_change += 1
    elif chosen_difficulty < 3 and current_completed == 0:
        engagement_change -= 1
    elif 3 <= chosen_difficulty <= 7:
        engagement_change += 0.5
    engagement_change += (current_engagement - 3) * 0.1
    next_engagement_level_float = current_engagement + engagement_change
    return int(np.clip(round(next_engagement_level_float), 1, 5))

def simulate_task_completion(next_engagement_level: int, chosen_difficulty: int) -> int:
    """
    Simulates the completion status (0 or 1) for the next task,
    based on the simulated engagement level for that task and its chosen difficulty.
    """
    completion_probability = 1.0 / (1 + np.exp(-(next_engagement_level * 0.5 - chosen_difficulty * 0.2)))
    return 1 if random.random() < completion_probability else 0

def simulate_next_state_and_reward(current_state: tuple, action: tuple, state_to_index: dict, action_to_index: dict, reward_mode="state") -> tuple:
    """
    Simulates the environment's response to an action taken in a given state.

    Returns:
        tuple: (next_state, reward)
    """
    if not is_valid_state(current_state, state_to_index) or not is_valid_action(action, action_to_index):
        return None, None

    current_simpsons_level, current_engagement, current_completed, current_prev_type = current_state
    chosen_task_type, chosen_difficulty = action

    next_simpsons_level = simulate_cognitive_load_change(current_simpsons_level, chosen_difficulty, current_completed)
    next_engagement_level = simulate_engagement_change(current_engagement, chosen_difficulty, current_completed)
    next_task_completed = simulate_task_completion(next_engagement_level, chosen_difficulty)
    next_prev_task_type = chosen_task_type

    next_state = (next_simpsons_level, next_engagement_level, next_task_completed, next_prev_task_type)

    if reward_mode == "state":
        reward = calculate_reward(current_state, next_state, state_to_index)
    else:
        reward = 1 if next_engagement_level >= current_engagement else -1

    return next_state, reward