import numpy as np
from e0_sc_monitor import initialize_control_chart, add_engagement_data, check_for_engagement_anomaly
from d0_ql_setup import define_state_space, define_action_space, initialize_q_table
from d3_ql_core import epsilon_greedy_action_selection, update_q_table
from e1_sc_feedback import get_adjusted_decision
# 1. Define state and action spaces
all_states_tuple, state_to_index, index_to_state, num_states = define_state_space()
all_actions_tuple, action_to_index, index_to_action, num_actions = define_action_space()
q_table = initialize_q_table(num_states, num_actions)

# 2. Initialize Shewhart control chart
control_chart_state = initialize_control_chart(window_size=10)

# 3. Learning loop parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 100

for episode in range(num_episodes):
    # Example: randomly pick a starting state (or use a fixed one)
    current_state = all_states_tuple[np.random.randint(len(all_states_tuple))]
    for step in range(20):  # steps per episode
        # Select action using epsilon-greedy
        action = epsilon_greedy_action_selection(current_state, q_table, state_to_index, index_to_action, action_to_index, epsilon)
        if action is None:
            # Skip update if no valid action is returned
            continue
        # Simulate environment: get new engagement rate (here, random for demo)
        new_engagement_rate = np.random.uniform(0.6, 0.9)
        add_engagement_data(control_chart_state, new_engagement_rate)
        # Shewhart anomaly detection for reward
        is_anomaly = check_for_engagement_anomaly(control_chart_state)
        reward = -1 if is_anomaly else 1
        # Simulate next state (for demo, random valid state)
        next_state = all_states_tuple[np.random.randint(len(all_states_tuple))]
        # Q-learning update
        update_q_table(q_table, current_state, action, reward, next_state, alpha, gamma, state_to_index, action_to_index)
        # Move to next state
        current_state = next_state

print("Simulation complete. Q-table updated based on Shewhart signals.")

# Instantiate a ShewhartControlChart object with a window size of 10
# control_chart = ShewhartControlChart(window_size=10) # Replace class instantiation
control_chart_state = initialize_control_chart(window_size=10) # Initialize state dictionary

# Instantiate a FeedbackInterface object, passing the created control chart instance
# feedback_interface = FeedbackInterf7ace(control_chart=control_chart) # No longer needed

# 1. Populate the control chart with sample data (at least 10 for window_size=10)
sample_engagement_history = [0.75, 0.78, 0.80, 0.76, 0.82, 0.79, 0.81, 0.77, 0.83, 0.79]

print(f"Populating control chart with {len(sample_engagement_history)} sample data points...")
for rate in sample_engagement_history:
    # control_chart.add_data(rate) # Replace method call
    add_engagement_data(control_chart_state, rate) # Call the function

print("Control chart populated. Initial limits calculated.")
# Access limits directly from the state dictionary
print(f"CL: {control_chart_state['cl']:.4f}, UCL: {control_chart_state['ucl']:.4f}, LCL: {control_chart_state['lcl']:.4f}")


# 2. Add a new engagement rate data point that is expected to be an anomaly
# Based on the sample data (mean around 0.79, std dev around 0.025), LCL is ~0.7165
# A value like 0.6 should be an anomaly.
new_engagement_rate = 0.6

# Add this new engagement rate to the ShewhartControlChart instance
# control_chart.add_data(new_engagement_rate) # Replace method call
add_engagement_data(control_chart_state, new_engagement_rate) # Call the function
print(f"\nAdded new engagement rate: {new_engagement_rate}")

# 3. Check if the latest data point added is an anomaly and print an alert
# is_anomaly = control_chart.check_for_anomaly() # Replace method call
is_anomaly = check_for_engagement_anomaly(control_chart_state)

if is_anomaly:
    print(f"ALERT: Anomaly detected! Latest engagement rate ({new_engagement_rate:.2f}) is outside control limits.")
else:
    print(f"Latest engagement rate ({new_engagement_rate:.2f}) is within control limits.")

# 4. Simulate getting a recommended action from the Q-Learning engine.
simulated_current_state = (3, 4, 1, 'B') # Example state
simulated_recommended_action = ('C', 6) # Example recommended action (Task C, Difficulty 6)
print(f"\nSimulated current state: {simulated_current_state}")
print(f"Simulated recommended action from Q-Learning: {simulated_recommended_action}")

# 5. Call the get_adjusted_decision() method of the FeedbackInterface instance
# This will now display the populated chart and prompt for user input.
print("\n--- Engagement Control Chart (via FeedbackInterface) ---")
# Capture both the adjusted decision and the chart figure returned by the method
# adjusted_decision, chart_fig = feedback_interface.get_adjusted_decision(simulated_current_state, simulated_recommended_action) # Replace method call
adjusted_decision, chart_fig = get_adjusted_decision(control_chart_state, simulated_current_state, simulated_recommended_action) # Call the function

# Display the chart figure if it was generated
if chart_fig:
    chart_fig.show()


# 6. Print the adjusted decision obtained from the user interface.
print(f"\nFinal Adjusted Decision after Feedback: {adjusted_decision}")

def handle_anomaly_and_update_q(
    current_state, action, next_state, q_table,
    state_to_index, action_to_index,
    learning_rate, discount_factor,
    anomaly_detected, teacher_override_enabled=True
):
    """
    Handles reward shaping and Q-table update when an anomaly is detected.
    Optionally allows teacher override for reward.
    """
    # Default: negative reward for anomaly, neutral for normal
    reward = -20.0 if anomaly_detected else 0.0

    # Optionally allow teacher override only if anomaly detected
    if anomaly_detected and teacher_override_enabled:
        try:
            user_input = input(f"Anomaly detected! Enter custom reward (or press Enter to use {reward}): ")
            if user_input.strip() != "":
                reward = float(user_input)
        except Exception:
            print("Invalid input. Using default anomaly reward.")

    # Update Q-table
    update_q_table(
        q_table, current_state, action, reward, next_state,
        learning_rate, discount_factor, state_to_index, action_to_index
    )
    print(f"Q-table updated with reward: {reward}")

# ... after add_engagement_data and is_anomaly check ...
if is_anomaly:
    print("Anomaly detected! Updating Q-table with penalty and (optional) teacher override.")
    # Define dummy/example variables for demonstration; replace with actual logic as needed
    current_state = simulated_current_state
    action = simulated_recommended_action
    next_state = simulated_current_state  # Replace with actual next state logic
    q_table = {}  # Replace with actual Q-table object or data structure
    state_to_index = {}  # Replace with actual mapping
    action_to_index = {}  # Replace with actual mapping
    learning_rate = 0.1
    discount_factor = 0.9

    handle_anomaly_and_update_q(
        current_state, action, next_state, q_table,
        state_to_index, action_to_index,
        learning_rate, discount_factor,
        anomaly_detected=True,
        teacher_override_enabled=True  # Set to False to disable manual override
    )
else:
    # Optionally update Q-table with normal reward or skip
    pass
