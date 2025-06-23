# Execute the training cell
# --- Main Execution Block ---

# Set random seed for reproducibility (optional, but good for debugging/comparison)
# random.seed(42)
# np.random.seed(42)


# --- Environment Setup ---
print("Setting up state space, action space, and Q-table...")
all_states_tuple, state_to_index, index_to_state, num_states = define_state_space()
all_actions_tuple, action_to_index, index_to_action, num_actions = define_action_space()
q_table = initialize_q_table(num_states, num_actions)
print(f"Setup complete. Total states: {num_states}, Total actions: {num_actions}, Q-table shape: {q_table.shape}")


# --- Training ---
# Define training parameters
num_episodes = 1000
max_steps_per_episode = 100
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0             # Initial epsilon
epsilon_decay_rate = 0.001
min_epsilon = 0.01

# Run the training loop using the function
total_rewards_per_episode = train_q_learning_agent(
    num_episodes, max_steps_per_episode, learning_rate, discount_factor,
    epsilon, epsilon_decay_rate, min_epsilon,
    state_to_index, index_to_state, action_to_index, index_to_action, q_table
)

# Execute the analysis cell
# --- Analysis and Visualization ---

print("\n--- Learned Policy Examples ---")

# Select representative states to analyze the learned policy
example_states = [
    (3, 5, 1, 'D'),  # Mid-cognitive load, High engagement, Task Completed, Previous task D -> Next should be A
    (1, 1, 0, 'A'),  # Low cognitive load, Low engagement, Task Not Completed, Previous task A -> Next should be B
    (5, 3, 0, 'B'),  # High cognitive load, Mid engagement, Task Not Completed, Previous task B -> Next should be C
    (2, 4, 1, 'C'),  # Low-ish cognitive load, High engagement, Task Completed, Previous task C -> Next should be D
    (3, 3, 0, 'A')   # The starting state used in training, Previous task A -> Next should be B
]

# Analyze and print policy for valid example states
valid_example_states = [state for state in example_states if is_valid_state(state, state_to_index)]

if not valid_example_states:
    print("No valid example states to display policy for.")
else:
    for state in valid_example_states:
        print(f"\n--- Policy for State: {state} ---")

        # Get and print optimal action
        optimal_action = get_optimal_action_for_state(state, q_table, state_to_index, index_to_action)
        if optimal_action:
            print(f"Optimal action: {optimal_action}")
        else:
             print("Could not determine optimal action.")

        # Get and print top N actions
        num_top_actions_to_show = 5
        top_actions = get_top_n_actions_for_state(state, q_table, state_to_index, index_to_action, n=num_top_actions_to_show)
        if top_actions:
            print(f"Top {num_top_actions_to_show} recommended actions and their Q-values:")
            for action, q_value in top_actions:
                print(f"  {action}: {q_value:.4f}")
        else:
             print(f"Could not determine top {num_top_actions_to_show} actions.")

# Plot the total rewards per episode learning curve
# Create a Matplotlib figure and axes for the reward plot
reward_fig, reward_ax = plt.subplots()

# Add a scatter trace for the total rewards per episode
reward_ax.plot(list(range(len(total_rewards_per_episode))), total_rewards_per_episode, label='Total Reward')

# Update the layout with title and axis labels
reward_ax.set_title('Total Reward per Episode during Training')
reward_ax.set_xlabel('Episode')
reward_ax.set_ylabel('Total Reward')

# Add a legend
reward_ax.legend()

# Display the Matplotlib figure
plt.show()