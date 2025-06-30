import itertools
import numpy as np

def define_state_space(num_buckets=5):
    """
    Defines the state space for the Q-learning agent in the adaptive learning environment.

    The state is a tuple comprising:
    - simpsons_integral_level (int: 1-5): Discretized cognitive load trend.
    - engagement_level (int: 1-5): Learner's engagement level.
    - task_completed (int: 0 or 1): Status of the previous task.
    - prev_task_type (str: 'A', 'B', 'C', 'D'): Type of the previously presented task.

    Args:
        num_buckets (int): The number of buckets to discretize the cognitive load trend (simpsons_integral_level).

    Returns:
        tuple: (all_states_tuple, state_to_index, index_to_state, num_states)
    """
    simpsons_integral_levels = range(1, num_buckets + 1)
    engagement_levels = range(1, 6)
    task_completed_statuses = [0, 1]
    prev_task_types = ['A', 'B', 'C', 'D']

    all_states_tuple = list(itertools.product(
        simpsons_integral_levels,
        engagement_levels,
        task_completed_statuses,
        prev_task_types
    ))

    state_to_index = {state: index for index, state in enumerate(all_states_tuple)}
    index_to_state = {index: state for state, index in state_to_index.items()}
    num_states = len(all_states_tuple)

    return all_states_tuple, state_to_index, index_to_state, num_states

def define_action_space():
    """
    Defines the action space for the Q-learning agent.

    An action is a tuple representing the next task to present:
    - task_type (str: 'A', 'B', 'C', 'D')
    - difficulty (int: 0-10)

    Returns:
        tuple: (all_actions_tuple, action_to_index, index_to_action, num_actions)
    """
    task_types = ['A', 'B', 'C', 'D']
    difficulties = range(0, 11) # 0 to 10 inclusive

    all_actions_tuple = list(itertools.product(task_types, difficulties))

    action_to_index = {action: index for index, action in enumerate(all_actions_tuple)}
    index_to_action = {index: action for action, index in action_to_index.items()}
    num_actions = len(all_actions_tuple)

    return all_actions_tuple, action_to_index, index_to_action, num_actions

def initialize_q_table(num_states: int, num_actions: int) -> np.ndarray:
    """
    Initializes the Q-table with zeros.

    Args:
        num_states (int): The total number of states in the environment.
        num_actions (int): The total number of actions available to the agent.

    Returns:
        np.ndarray: The initialized Q-table as a NumPy array of shape (num_states, num_actions).
    """
    return np.zeros((num_states, num_actions))

def is_valid_state(state_tuple, state_to_index):
    """
    Checks if a state tuple is valid (exists in the state space).
    """
    return state_tuple in state_to_index

def is_valid_action(action_tuple, action_to_index):
    """
    Checks if an action tuple is valid (exists in the action space).
    """
    return action_tuple in action_to_index