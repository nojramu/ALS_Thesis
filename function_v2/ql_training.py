import numpy as np
from ql_setup import is_valid_state
from ql_core import epsilon_greedy_action_selection, update_q_table
from ql_simulator import simulate_next_state_and_reward

def train_q_learning_agent(
    num_episodes=100,
    max_steps_per_episode=20,
    learning_rate=0.1,
    discount_factor=0.9,
    epsilon=0.2,
    epsilon_decay_rate=0.0,
    min_epsilon=0.01,
    reward_mode="state",
    progress_interval=10,
    state_for_policy_tracking=(3, 3, 0, 'A')
):
    """
    Trains a Q-learning agent with best practices and logging.
    Returns:
        q_table: Final Q-table.
        total_rewards_per_episode: List of total rewards per episode.
        max_q_values_over_time: List of max Q-values for plotting.
        policy_evolution: List of optimal actions for a tracked state.
    """
    from ql_setup import define_state_space, define_action_space, initialize_q_table

    states, state_to_index, index_to_state, num_states = define_state_space()
    actions, action_to_index, index_to_action, num_actions = define_action_space()
    q_table = initialize_q_table(num_states, num_actions)
    total_rewards_per_episode = []
    max_q_values_over_time = []
    policy_evolution = []
    current_epsilon = epsilon

    print(f"\nStarting Q-Learning training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        starting_state = (3, 3, 0, 'A')  # Example fixed valid starting state
        if not is_valid_state(starting_state, state_to_index):
            print(f"Episode {episode}: Invalid starting state {starting_state}. Skipping episode.")
            total_rewards_per_episode.append(0)
            continue

        current_state = starting_state
        total_episode_reward = 0

        for step in range(max_steps_per_episode):
            action = epsilon_greedy_action_selection(
                current_state, q_table, state_to_index, index_to_action, action_to_index, current_epsilon
            )
            if action is None:
                print(f"Episode {episode}, Step {step}: Action selection failed. Ending episode.")
                break

            next_state, reward = simulate_next_state_and_reward(
                current_state, action, state_to_index, action_to_index, reward_mode=reward_mode
            )
            if next_state is None or reward is None:
                print(f"Episode {episode}, Step {step}: Simulation or reward calculation failed. Ending episode.")
                break

            update_q_table(
                q_table, current_state, action, reward, next_state,
                learning_rate, discount_factor, state_to_index, action_to_index
            )

            total_episode_reward += reward
            current_state = next_state

        # Epsilon decay
        current_epsilon = max(min_epsilon, current_epsilon - epsilon_decay_rate)

        total_rewards_per_episode.append(total_episode_reward)

        # Track max Q-value and policy evolution
        if episode % 10 == 0:
            max_q_values_over_time.append(np.max(q_table))
            if state_for_policy_tracking in state_to_index:
                optimal_action = np.argmax(q_table[state_to_index[state_for_policy_tracking]])
                policy_evolution.append(optimal_action)

        # Print progress
        if (episode + 1) % progress_interval == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_episode_reward:.2f}, Epsilon: {current_epsilon:.4f}")

    print("\nQ-Learning training finished.")
    return q_table, total_rewards_per_episode, max_q_values_over_time, policy_evolution

def test_q_learning():
    """
    Example test function to run Q-learning training and print results.
    """
    q_table, rewards, max_q_values_over_time, policy_evolution = train_q_learning_agent(
        num_episodes=100,
        max_steps_per_episode=20,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.2,
        epsilon_decay_rate=0.001,
        min_epsilon=0.01,
        reward_mode="state",
        progress_interval=10
    )
    print("Training complete. Final Q-table:")
    print(q_table)
    print("Total rewards per episode:", rewards)
    return q_table, rewards, max_q_values_over_time, policy_evolution