import numpy as np
from d1_ql_validator import is_valid_state
from d3_ql_core import epsilon_greedy_action_selection, update_q_table
from d2_ql_simulator import simulate_next_state_and_reward
# --- Q-Learning Training Function ---

def train_q_learning_agent(num_episodes: int, max_steps_per_episode: int, learning_rate: float, discount_factor: float, epsilon: float, epsilon_decay_rate: float, min_epsilon: float, state_to_index: dict, index_to_state: dict, action_to_index: dict, index_to_action: dict, q_table: np.ndarray):
    """
    Runs the Q-learning training loop for a specified number of episodes.

    Args:
        num_episodes (int): Total number of learning episodes.
        max_steps_per_episode (int): Maximum number of interactions within each episode.
        learning_rate (float): Alpha (α): How much the agent learns from each experience.
        discount_factor (float): Gamma (γ): Discount rate for future rewards.
        epsilon (float): Epsilon (ε): Initial probability of exploration.
        epsilon_decay_rate (float): Rate at which epsilon decreases after each episode.
        min_epsilon (float): Minimum value for epsilon.
        state_to_index (dict): Mapping from state tuple to index.
        index_to_state (dict): Mapping from index to state tuple.
        action_to_index (dict): Mapping from action tuple to index.
        index_to_action (dict): Mapping from index to action tuple.
        q_table (np.ndarray): The Q-table to be trained (modified in-place).

    Returns:
        list: A list containing the total reward accumulated in each episode.
    """
    total_rewards_per_episode = []
    current_epsilon = epsilon # Use a variable for the decaying epsilon

    print(f"\nStarting Q-Learning training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        # Initialize starting state for each episode (must be a valid state)
        starting_state = (3, 3, 0, 'A') # Example starting state

        # Validate starting state
        if not is_valid_state(starting_state, state_to_index):
            print(f"Episode {episode}: Invalid starting state {starting_state}. Skipping episode.")
            total_rewards_per_episode.append(0)
            continue

        current_state = starting_state
        total_episode_reward = 0

        # Episode loop
        for step in range(max_steps_per_episode):
            # Action selection
            action = epsilon_greedy_action_selection(current_state, q_table, state_to_index, index_to_action, action_to_index, current_epsilon)

            # Handle case where action selection failed (shouldn't happen with valid start state and valid_state check)
            if action is None:
               print(f"Episode {episode}, Step {step}: Action selection failed. Ending episode.")
               break

            # Simulate environment step
            next_state, reward = simulate_next_state_and_reward(current_state, action, state_to_index, action_to_index)

            # Handle simulation failure (e.g., invalid inputs passed internally which shouldn't occur now,
            # or calculate_reward returning None due to invalid simulated next_state)
            if next_state is None or reward is None:
                 print(f"Episode {episode}, Step {step}: Simulation or reward calculation failed. Ending episode.")
                 break

            # Q-table update
            update_q_table(q_table, current_state, action, reward, next_state, learning_rate, discount_factor, state_to_index, action_to_index)

            # Accumulate reward
            total_episode_reward += reward

            # Transition to next state
            current_state = next_state

            # Optional: End episode early condition (e.g., task completion)
            if current_state[2] == 1:
               # print(f"Episode {episode}, Step {step}: Task completed, ending episode.")
               pass # Keep episode running for fixed steps, or uncomment break
               # break

        # Epsilon decay after each episode
        current_epsilon = max(min_epsilon, current_epsilon - epsilon_decay_rate)

        # Store total reward for the episode
        total_rewards_per_episode.append(total_episode_reward)

        # Print periodic progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_episode_reward:.2f}, Epsilon: {current_epsilon:.4f}")


    print("\nQ-Learning training finished.")
    return total_rewards_per_episode