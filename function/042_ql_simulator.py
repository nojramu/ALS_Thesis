# --- Simulation Environment Dynamics (Modularized) ---

def simulate_cognitive_load_change(current_simpsons_level: int, chosen_difficulty: int, current_completed: int) -> int:
    """
    Simulates the change in the simpsons_integral_level (cognitive load)
    based on the chosen task's difficulty and the outcome of the previous task.

    Args:
        current_simpsons_level (int): The learner's current discretized cognitive load level (1-5).
        chosen_difficulty (int): The difficulty level of the task chosen by the agent (0-10).
        current_completed (int): Whether the previous task was completed (0 or 1).

    Returns:
        int: The simulated next discretized cognitive load level (1-5), clamped to the valid range.
    """
    # Heuristic: Difficulty influences load change. Difficulty 5 is neutral.
    # Higher difficulty increases load, lower difficulty decreases load.
    simpsons_change = (chosen_difficulty - 5) * 0.2
    # Heuristic: Completing a task might slightly reduce load in the subsequent state.
    if current_completed == 1:
        simpsons_change -= 0.5
    # Calculate potential next level and clamp to the valid range [1, 5]
    next_simpsons_level_float = current_simpsons_level + simpsons_change
    return int(np.clip(round(next_simpsons_level_float), 1, 5))

def simulate_engagement_change(current_engagement: int, chosen_difficulty: int, current_completed: int) -> int:
    """
    Simulates the change in the engagement_level based on the chosen task's
    difficulty and the outcome of the previous task.

    Args:
        current_engagement (int): The learner's current engagement level (1-5).
        chosen_difficulty (int): The difficulty level of the task chosen by the agent (0-10).
        current_completed (int): Whether the previous task was completed (0 or 1).

    Returns:
        int: The simulated next engagement level (1-5), clamped to the valid range.
    """
    engagement_change = 0

    # Heuristics: Task difficulty and completion influence engagement.
    # Challenging completed tasks and mid-range difficulty are generally positive.
    # Very easy or very difficult tasks (especially if not completed) might be negative.
    if chosen_difficulty > 7 and current_completed == 1:
        engagement_change += 1 # Challenging and completed: boosts engagement
    elif chosen_difficulty < 3 and current_completed == 0:
        engagement_change -= 1 # Too easy and not completed (boredom?): harms engagement
    elif 3 <= chosen_difficulty <= 7:
         engagement_change += 0.5 # Mid-range difficulty is somewhat engaging

    # Heuristic: Tendency towards average engagement level (3)
    engagement_change += (current_engagement - 3) * 0.1

    # Calculate potential next level and clamp to the valid range [1, 5]
    next_engagement_level_float = current_engagement + engagement_change
    return int(np.clip(round(next_engagement_level_float), 1, 5))

def simulate_task_completion(next_engagement_level: int, chosen_difficulty: int) -> int:
    """
    Simulates the completion status (0 or 1) for the *next* task,
    based on the simulated engagement level for that task and its chosen difficulty.

    Args:
        next_engagement_level (int): The simulated engagement level for the next task (1-5).
        chosen_difficulty (int): The difficulty level of the task chosen by the agent (0-10).

    Returns:
        int: The simulated completion status (0: not completed, 1: completed) for the next task.
    """
    # Use a sigmoid-like probability based on engagement and difficulty.
    # Higher engagement and lower difficulty increase the probability of completion.
    completion_probability = 1.0 / (1 + np.exp(-(next_engagement_level * 0.5 - chosen_difficulty * 0.2)))
    # Randomly determine completion based on the calculated probability
    return 1 if random.random() < completion_probability else 0

def simulate_next_state_and_reward(current_state: tuple, action: tuple, state_to_index: dict, action_to_index: dict) -> tuple[tuple | None, float | None]:
    """
    Simulates the environment's response to an action taken in a given state.

    This function acts as the environment's dynamics model. It determines the
    subsequent state and the immediate reward received. It relies on modular
    simulation functions for state component changes and the calculate_reward
    function for reward determination. Includes input validation.

    Args:
        current_state (tuple): The state tuple before the action was taken.
        action (tuple): The action tuple chosen by the agent.
        state_to_index (dict): Mapping from state tuple to index (for validation).
        action_to_index (dict): Mapping from action tuple to index (for validation).

    Returns:
        tuple: A tuple containing (next_state, reward).
               - next_state (tuple): The simulated state after taking the action.
               - reward (float): The immediate reward obtained.
               Returns (None, None) if input validation fails or if the simulated
               next state is invalid for reward calculation.
    """
    # Validate input state and action
    if not is_valid_state(current_state, state_to_index) or not is_valid_action(action, action_to_index):
         return None, None # Error messages printed by validation functions

    # Unpack current state and action
    current_simpsons_level, current_engagement, current_completed, current_prev_type = current_state
    chosen_task_type, chosen_difficulty = action

    # Simulate components of the next state using modular functions
    next_simpsons_level = simulate_cognitive_load_change(current_simpsons_level, chosen_difficulty, current_completed)
    next_engagement_level = simulate_engagement_change(current_engagement, chosen_difficulty, current_completed)
    # Simulate completion for the *next* task based on its characteristics and predicted engagement
    next_task_completed = simulate_task_completion(next_engagement_level, chosen_difficulty)

    # The previous task type in the next state becomes the task type of the action just taken
    next_prev_task_type = chosen_task_type

    # Construct the simulated next state tuple
    next_state_simulated = (next_simpsons_level, next_engagement_level, next_task_completed, next_prev_task_type)

    # Calculate the immediate reward based on the simulated next state
    # calculate_reward internally validates the next_state
    reward_simulated = calculate_reward(current_state, next_state_simulated, state_to_index)

    return next_state_simulated, reward_simulated
