# Instantiate a ShewhartControlChart object with a window size of 10
# control_chart = ShewhartControlChart(window_size=10) # Replace class instantiation
control_chart_state = initialize_control_chart(window_size=10) # Initialize state dictionary

# Instantiate a FeedbackInterface object, passing the created control chart instance
# feedback_interface = FeedbackInterface(control_chart=control_chart) # No longer needed

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
is_anomaly = check_for_engagement_anomaly(control_chart_state) # Call the function

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