if __name__ == "__main__":
    from ql_setup import define_state_space, define_action_space
    from ql_training import train_q_learning_agent
    from ql_analysis import get_optimal_action_for_state, get_top_n_actions_for_state, plot_q_table_heatmap
    from plot_utils import plot_line_chart

    # --- Training ---
    q_table, rewards, max_q_values, policy_evolution = train_q_learning_agent(
        num_episodes=200,
        max_steps_per_episode=30,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay_rate=0.005,
        min_epsilon=0.05,
        reward_mode="state",
        progress_interval=20
    )

    # --- Analysis and Visualization ---
    print("\n--- Learned Policy Examples ---")
    states, state_to_index, index_to_state, num_states = define_state_space()
    actions, action_to_index, index_to_action, num_actions = define_action_space()

    example_states = [
        (3, 5, 1, 'D'),  # Mid-cognitive load, High engagement, Task Completed, Previous task D -> Next should be A
        (1, 1, 0, 'A'),  # Low cognitive load, Low engagement, Task Not Completed, Previous task A -> Next should be B
        (5, 3, 0, 'B'),  # High cognitive load, Mid engagement, Task Not Completed, Previous task B -> Next should be C
        (2, 4, 1, 'C'),  # Low-ish cognitive load, High engagement, Task Completed, Previous task C -> Next should be D
        (3, 3, 0, 'A')   # The starting state used in training, Previous task A -> Next should be B
    ]

    for state in example_states:
        print(f"\n--- Policy for State: {state} ---")
        optimal_action = get_optimal_action_for_state(state, q_table, state_to_index, index_to_action)
        print(f"Optimal action: {optimal_action}")
        top_actions = get_top_n_actions_for_state(state, q_table, state_to_index, index_to_action, n=5)
        print("Top actions:")
        if top_actions is not None:
            for action, q_value in top_actions:
                print(f"  {action}: {q_value:.4f}")
        else:
            print("  No actions available for this state.")

    # Plot learning curve
    plot_line_chart(
        x=list(range(1, len(rewards) + 1)),
        y=[rewards],
        xlabel='Episode',
        ylabel='Total Reward',
        title='Total Reward per Episode',
        legend_labels=['Total Reward'],
        show=True
    )

    # Plot Q-table heatmap
    plot_q_table_heatmap(q_table)