import itertools
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# --- Environment Setup Functions ---

def define_state_space():
    """
    Defines the state space for the Q-learning agent in the adaptive learning environment.

    The state is a tuple comprising:
    - simpsons_integral_level (int: 1-5): Discretized cognitive load trend.
    - engagement_level (int: 1-5): Learner's engagement level.
    - task_completed (int: 0 or 1): Status of the previous task.
    - prev_task_type (str: 'A', 'B', 'C', 'D'): Type of the previously presented task.

    This function generates all possible combinations of these state variables
    and creates mappings between state tuples and unique integer indices.

    Returns:
        tuple: A tuple containing:
            - all_states_tuple (list): List of all unique possible state tuples.
            - state_to_index (dict): Dictionary mapping each state tuple to its unique integer index.
            - index_to_state (dict): Dictionary mapping each integer index back to its corresponding state tuple.
            - num_states (int): The total count of unique states in the state space.
    """
    # Define the possible values for each component of the state tuple
    simpsons_integral_levels = range(1, 6)
    engagement_levels = range(1, 6)
    task_completed_statuses = [0, 1]
    prev_task_types = ['A', 'B', 'C', 'D']

    # Generate all combinations using itertools.product
    all_states_tuple = list(itertools.product(
        simpsons_integral_levels,
        engagement_levels,
        task_completed_statuses,
        prev_task_types
    ))

    # Create mappings for efficient lookup
    state_to_index = {state: index for index, state in enumerate(all_states_tuple)}
    index_to_state = {index: state for state, index in state_to_index.items()}

    num_states = len(all_states_tuple)

    return all_states_tuple, state_to_index, index_to_state, num_states


def define_action_space():
    """
    Defines the action space for the Q-learning agent.

    An action is a tuple representing the next task to present:
    - task_type (str: 'A', 'B', 'C', 'D'): The category of the task.
    - difficulty (int: 0-10): The difficulty level of the task.

    This function generates all possible combinations of task type and difficulty
    and creates mappings between action tuples and unique integer indices.

    Returns:
        tuple: A tuple containing:
            - all_actions_tuple (list): List of all unique possible action tuples.
            - action_to_index (dict): Dictionary mapping each action tuple to its unique integer index.
            - index_to_action (dict): Dictionary mapping each integer index back to its corresponding action tuple.
            - num_actions (int): The total count of unique actions in the action space.
    """
    # Define the possible values for each component of the action tuple
    task_types = ['A', 'B', 'C', 'D']
    difficulties = range(0, 11) # 0 to 10 inclusive

    # Generate all combinations using itertools.product
    all_actions_tuple = list(itertools.product(task_types, difficulties))

    # Create mappings for efficient lookup
    action_to_index = {action: index for index, action in enumerate(all_actions_tuple)}
    index_to_action = {index: action for action, index in action_to_index.items()}

    num_actions = len(all_actions_tuple)

    return all_actions_tuple, action_to_index, index_to_action, num_actions


def initialize_q_table(num_states: int, num_actions: int) -> np.ndarray:
    """
    Initializes the Q-table with zeros.

    The Q-table is a fundamental component of Q-learning, storing the learned
    value for taking a specific action in a specific state.

    Args:
        num_states (int): The total number of states in the environment.
        num_actions (int): The total number of actions available to the agent.

    Returns:
        np.ndarray: The initialized Q-table as a NumPy array of shape (num_states, num_actions),
                    with all values set to 0.0 initially.
    """
    # Create a NumPy array filled with zeros with dimensions corresponding to the state and action space sizes
    q_table = np.zeros((num_states, num_actions))
    return q_table