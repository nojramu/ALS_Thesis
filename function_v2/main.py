if __name__ == "__main__":
    from ql_setup import define_state_space, define_action_space, initialize_q_table
    from ql_core import epsilon_greedy_action_selection, update_q_table
    from shewhart_control import initialize_control_chart, add_engagement_data, check_for_engagement_anomaly, get_adjusted_decision

    num_steps = 15
    epsilon = 0.1
    states, state_to_index, index_to_state, num_states = define_state_space()
    actions, action_to_index, index_to_action, num_actions = define_action_space()
    q_table = initialize_q_table(num_states, num_actions)
    chart_state = initialize_control_chart(window_size=10)
    current_state = states[0]
    for step in range(num_steps):
        action = epsilon_greedy_action_selection(current_state, q_table, state_to_index, index_to_action, action_to_index, epsilon=epsilon)
        # Ensure action is not None
        if action is None:
            action = actions[0]  #hoose a sensible default action
        # Simulate environment (replace with your own logic)
        next_state = states[(state_to_index[current_state] + 1) % num_states]
        engagement_rate = next_state[1] / 5.0  # Example: normalize engagement_level
        add_engagement_data(chart_state, engagement_rate)
        is_anomaly = check_for_engagement_anomaly(chart_state)
        reward = -1 if is_anomaly else 1
        update_q_table(q_table, current_state, action, reward, next_state, 0.1, 0.9, state_to_index, action_to_index)
        current_state = next_state
        # Feedback interface (optional, e.g., every 5 steps)
        if step % 5 == 0:
            adjusted_action, chart_fig = get_adjusted_decision(chart_state, current_state, action)
    print("Final Q-table:", q_table)
    print("Final Chart State:", chart_state)