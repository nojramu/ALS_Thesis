import numpy as np
import itertools

# --- State and Action Space Definitions ---

def define_state_space():
    cognitive_load_levels = range(1, 6)  # 1-5
    engagement_levels = range(1, 6)      # 1-5
    task_completed_statuses = [0, 1]     # 0: not completed, 1: completed
    prev_task_types = ['A', 'B', 'C', 'D']
    all_states_tuple = list(itertools.product(
        cognitive_load_levels,
        engagement_levels,
        task_completed_statuses,
        prev_task_types
    ))
    state_to_index = {state: idx for idx, state in enumerate(all_states_tuple)}
    index_to_state = {idx: state for state, idx in state_to_index.items()}
    num_states = len(all_states_tuple)
    return all_states_tuple, state_to_index, index_to_state, num_states

def define_action_space():
    task_types = ['A', 'B', 'C', 'D']
    difficulties = range(0, 11)  # 0-10
    all_actions_tuple = list(itertools.product(task_types, difficulties))
    action_to_index = {action: idx for idx, action in enumerate(all_actions_tuple)}
    index_to_action = {idx: action for action, idx in action_to_index.items()}
    num_actions = len(all_actions_tuple)
    return all_actions_tuple, action_to_index, index_to_action, num_actions

def is_valid_state(state_tuple, state_to_index):
    return state_tuple in state_to_index

def is_valid_action(action_tuple, action_to_index):
    return action_tuple in action_to_index

# --- Q-table and Policy Functions ---

def initialize_q_table(num_states, num_actions):
    return np.zeros((num_states, num_actions))

def epsilon_greedy_action_selection(current_state, q_table, state_to_index, index_to_action, action_to_index, epsilon):
    if current_state not in state_to_index:
        return None
    state_index = state_to_index[current_state]
    if np.random.rand() < epsilon:
        action_index = np.random.choice(list(index_to_action.keys()))  # Explore
    else:
        action_index = np.argmax(q_table[state_index])  # Exploit
    return index_to_action[action_index]

def update_q_table(q_table, current_state, action, reward, next_state, learning_rate, discount_factor, state_to_index, action_to_index):
    if not (current_state in state_to_index and action in action_to_index and next_state in state_to_index):
        return
    s = state_to_index[current_state]
    a = action_to_index[action]
    s_next = state_to_index[next_state]
    max_next_q = np.max(q_table[s_next])
    q_table[s, a] = q_table[s, a] + learning_rate * (reward + discount_factor * max_next_q - q_table[s, a])

# --- Simulation Functions ---

def simulate_cognitive_load_change(current_simpsons_level, chosen_difficulty, current_completed):
    simpsons_change = (chosen_difficulty - 5) * 0.2
    if current_completed == 1:
        simpsons_change -= 0.5
    next_simpsons_level_float = current_simpsons_level + simpsons_change
    return int(np.clip(round(next_simpsons_level_float), 1, 5))

def simulate_engagement_change(current_engagement, chosen_difficulty, current_completed):
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

def simulate_task_completion(next_engagement_level, chosen_difficulty):
    completion_probability = 1.0 / (1 + np.exp(-(next_engagement_level * 0.5 - chosen_difficulty * 0.2)))
    return 1 if np.random.rand() < completion_probability else 0

def simulate_next_state_and_reward(current_state, action, state_to_index, action_to_index, reward_mode="state"):
    if not (current_state in state_to_index and action in action_to_index):
        return None, None
    current_simpsons_level, current_engagement, current_completed, current_prev_type = current_state
    chosen_task_type, chosen_difficulty = action

    next_simpsons_level = simulate_cognitive_load_change(current_simpsons_level, chosen_difficulty, current_completed)
    next_engagement_level = simulate_engagement_change(current_engagement, chosen_difficulty, current_completed)
    next_task_completed = simulate_task_completion(next_engagement_level, chosen_difficulty)
    next_prev_task_type = chosen_task_type

    next_state = (next_simpsons_level, next_engagement_level, next_task_completed, next_prev_task_type)

    if reward_mode == "state":
        reward = calculate_state_based_reward(current_state, next_state)
    else:
        reward = 1 if next_engagement_level >= current_engagement else -1

    return next_state, reward

# --- Reward Calculation ---

def calculate_state_based_reward(current_state, next_state):
    simpsons_integral_level, engagement_level, task_completed, prev_task_type = next_state
    reward_cognitive_load = 10.0 if simpsons_integral_level in [2, 3, 4] else -5.0
    engagement_reward_mapping = {1: -2.0, 2: 0.0, 3: 2.0, 4: 5.0, 5: 10.0}
    reward_engagement = engagement_reward_mapping.get(engagement_level, 0.0)
    reward_task_completed = 15.0 if task_completed == 1 else -3.0
    weight_cognitive_load = 0.4
    weight_engagement = 0.4
    weight_task_completed = 0.2
    total_reward = (weight_cognitive_load * reward_cognitive_load +
                    weight_engagement * reward_engagement +
                    weight_task_completed * reward_task_completed)
    return total_reward

# --- Control Chart for Anomaly Detection ---

def initialize_control_chart(window_size=10):
    return {'data': [], 'window_size': window_size}

def add_engagement_data(chart_state, engagement_rate):
    chart_state['data'].append(engagement_rate)
    if len(chart_state['data']) > chart_state['window_size']:
        chart_state['data'].pop(0)

def check_for_engagement_anomaly(chart_state):
    data = chart_state['data']
    if len(data) < chart_state['window_size']:
        return False
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    ucl = mean + 3 * std
    lcl = mean - 3 * std
    latest = data[-1]
    return latest > ucl or latest < lcl

# --- Q-learning Training ---

def train_q_learning_agent(
    num_episodes=100,
    max_steps_per_episode=20,
    learning_rate=0.1,
    discount_factor=0.9,
    epsilon=0.2,
    epsilon_decay_rate=0.0,
    min_epsilon=0.01,
    reward_mode="state",
    progress_interval=10
):
    """
    Trains a Q-learning agent with best practices and logging.
    """
    states, state_to_index, index_to_state, num_states = define_state_space()
    actions, action_to_index, index_to_action, num_actions = define_action_space()
    q_table = initialize_q_table(num_states, num_actions)
    control_chart = initialize_control_chart(window_size=10)
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

            add_engagement_data(control_chart, next_state[1])  # engagement_level
            is_anomaly = check_for_engagement_anomaly(control_chart)
            if reward_mode == "anomaly":
                reward = -1 if is_anomaly else 1

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
            state_for_policy_tracking = (3, 3, 0, 'A')
            if state_for_policy_tracking in state_to_index:
                optimal_action = np.argmax(q_table[state_to_index[state_for_policy_tracking]])
                policy_evolution.append(optimal_action)

        # Print progress and save Q-table snapshot
        if (episode + 1) % progress_interval == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_episode_reward:.2f}, Epsilon: {current_epsilon:.4f}")
        if episode % 100 == 0:
            np.save(f"qtable_snapshot_ep{episode}.npy", q_table)
            print(f"Q-table snapshot at episode {episode} saved.")

    print("\nQ-Learning training finished.")
    return q_table, total_rewards_per_episode, max_q_values_over_time, policy_evolution

# --- Example usage and Plotting ---

# if __name__ == "__main__":
#     from plot_responsive import plot_line_chart, save_figure_to_image_folder

#     # Run Q-learning with state-based reward
#     q_table, rewards, max_q_values_over_time, policy_evolution = train_q_learning_agent(
#         num_episodes=20,
#         max_steps_per_episode=20,
#         learning_rate=0.1,
#         discount_factor=0.9,
#         epsilon=0.2,
#         epsilon_decay_rate=0.0,
#         min_epsilon=0.01,
#         reward_mode="state",
#         progress_interval=10
#     )
#     print("Training complete. Final Q-table:")
#     print(q_table)
#     print("Total rewards per episode:", rewards)

#     # Plot total rewards per episode
#     fig = plot_line_chart(
#         x=list(range(1, len(rewards) + 1)),
#         y=[rewards],
#         xlabel='Episode',
#         ylabel='Total Reward',
#         title='Total Reward per Episode',
#         legend_labels=['Total Reward'],
#         show=False
#     )
#     save_figure_to_image_folder(fig, prefix='q_learning_rewards', image_dir='image')

#     # Plot max Q-value evolution
#     fig2 = plot_line_chart(
#         x=np.arange(0, len(max_q_values_over_time)),
#         y=[max_q_values_over_time],
#         xlabel="Episode",
#         ylabel="Max Q-value",
#         title="Max Q-value Evolution During Training",
#         legend_labels=["Max Q-value"],
#         show=False
#     )
#     save_figure_to_image_folder(fig2, prefix="q_value_evolution", image_dir="image")

#     # Plot policy evolution for a tracked state
#     fig3 = plot_line_chart(
#         x=np.arange(0, len(policy_evolution)),
#         y=[policy_evolution],
#         xlabel="Episode",
#         ylabel="Optimal Action Index",
#         title="Policy Evolution for State X",
#         legend_labels=["Optimal Action"],
#         show=False
#     )
#     save_figure_to_image_folder(fig3, prefix="policy_evolution_stateX", image_dir="image")

#     # Or run with anomaly-based reward
#     # q_table, rewards, max_q_values_over_time, policy_evolution = train_q_learning_agent(num_episodes=20, reward_mode="anomaly")
