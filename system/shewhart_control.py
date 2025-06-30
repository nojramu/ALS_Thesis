from plot_utils import plot_line_chart, save_figure_to_image_folder
from ql_core import update_q_table  # Import at the top for clarity
import numpy as np

def initialize_control_chart(window_size=10, anomaly_buffer_size=3):
    """
    Initialize the control chart state with a specified window size.
    """
    return {
        'window_size': window_size,
        'engagement_data': [],
        'cl': None,
        'ucl': None,
        'lcl': None,
        'anomalies': [],
        'anomaly_buffer': [False] * anomaly_buffer_size  # Track recent anomalies
    }

def calculate_control_limits(chart_state, num_stddev=3):
    """
    Calculate the control limits (CL, UCL, LCL) for the current engagement data using median and IQR.
    Marks anomalies if data points are outside control limits.
    """
    data = chart_state['engagement_data']
    if len(data) < 2:
        chart_state['cl'] = chart_state['ucl'] = chart_state['lcl'] = None
        chart_state['anomalies'] = []
        return
    arr = np.array(data)
    median = np.median(arr)
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = q75 - q25
    # For normal distribution, std ≈ IQR/1.349
    robust_std = iqr / 1.349 if iqr > 0 else 0.0
    chart_state['cl'] = median
    chart_state['ucl'] = min(1.0, float(median + num_stddev * robust_std))
    chart_state['lcl'] = max(0.0, float(median - num_stddev * robust_std))
    chart_state['anomalies'] = [i for i, v in enumerate(data) if v > chart_state['ucl'] or v < chart_state['lcl']]

def add_engagement_data(chart_state, engagement_rate, num_stddev=3):
    """
    Add a new engagement rate to the control chart, maintaining the window size.
    Recalculates control limits after adding.
    (No need to call calculate_control_limits separately.)
    """
    chart_state['engagement_data'].append(engagement_rate)
    if len(chart_state['engagement_data']) > chart_state['window_size']:
        chart_state['engagement_data'].pop(0)
    calculate_control_limits(chart_state, num_stddev=num_stddev)
    # Update anomaly buffer
    is_anomaly = check_for_engagement_anomaly(chart_state)
    chart_state['anomaly_buffer'].append(is_anomaly)
    if len(chart_state['anomaly_buffer']) > 3:  # buffer size
        chart_state['anomaly_buffer'].pop(0)

def check_for_engagement_anomaly(chart_state):
    """
    Check if the latest engagement rate is an anomaly (outside control limits).
    """
    if chart_state['cl'] is None or not chart_state['engagement_data']:
        return False
    latest = chart_state['engagement_data'][-1]
    return latest > chart_state['ucl'] or latest < chart_state['lcl']

def get_control_chart_data(chart_state):
    """
    Always return a dict with all keys, even if values are None or empty.
    """
    return {
        'engagement_rates': chart_state.get('engagement_data', []),
        'cl': chart_state.get('cl'),
        'ucl': chart_state.get('ucl'),
        'lcl': chart_state.get('lcl'),
        'anomalies': chart_state.get('anomalies', [])
    }

def plot_shewhart_chart(chart_state, filename_prefix='shewhart_chart', image_dir='image'):
    """
    Plots the Shewhart control chart with engagement rates, CL, UCL, LCL, and anomalies.
    Returns the matplotlib figure.
    """
    chart_data = get_control_chart_data(chart_state)
    if not chart_data:
        print("Not enough data to plot the control chart.")
        return None

    x = list(range(len(chart_data['engagement_rates'])))
    y = [chart_data['engagement_rates']]
    legend = ['Engagement Rate']
    cl = [chart_data['cl']] * len(x)
    ucl = [chart_data['ucl']] * len(x)
    lcl = [chart_data['lcl']] * len(x)
    y.extend([cl, ucl, lcl])
    legend.extend(['CL', 'UCL', 'LCL'])

    fig = plot_line_chart(
        x=x,
        y=y,
        xlabel='Task Index (within window)',
        ylabel='Engagement Rate',
        title='Shewhart Control Chart',
        legend_labels=legend,
        show=False
    )

    # Highlight anomalies
    if chart_data['anomalies']:
        ax = fig.axes[0]
        anomaly_x = [i for i in chart_data['anomalies']]
        anomaly_y = [chart_data['engagement_rates'][i] for i in anomaly_x]
        ax.plot(anomaly_x, anomaly_y, 'rx', markersize=10, label='Anomaly')
        ax.legend()

    save_figure_to_image_folder(fig, prefix=filename_prefix, image_dir=image_dir)
    print("Shewhart control chart plotted and saved.")
    return fig

def get_adjusted_decision(chart_state, current_state, recommended_action):
    """
    Modular feedback interface: displays the control chart, system recommendation,
    and allows user to adjust difficulty. Returns (adjusted_action, chart_fig).
    """
    print("\n--- Engagement Control Chart ---")
    chart_fig = plot_shewhart_chart(chart_state)
    recommended_task_type, recommended_difficulty = recommended_action
    print(f"\nSystem Recommendation: Task Type: {recommended_task_type}, Difficulty: {recommended_difficulty}")
    print(f"Current State: {current_state}")

    while True:
        try:
            user_input = input(f"Enter desired difficulty (0-10) or press Enter to accept [{recommended_difficulty}]: ")
            if user_input == "":
                adjusted_difficulty = recommended_difficulty
                print("Accepted recommended difficulty.")
                break
            adjusted_difficulty = int(user_input)
            if 0 <= adjusted_difficulty <= 10:
                print(f"User adjusted difficulty to {adjusted_difficulty}.")
                break
            else:
                print("Invalid input. Difficulty must be 0-10.")
        except Exception:
            print("Invalid input. Using recommended difficulty.")
            adjusted_difficulty = recommended_difficulty
            break

    adjusted_action = (recommended_task_type, adjusted_difficulty)
    return adjusted_action, chart_fig

def handle_anomaly_and_update_q(
    current_state, action, next_state, q_table,
    state_to_index, action_to_index,
    learning_rate, discount_factor,
    anomaly_detected, teacher_override_enabled=True, default_anomaly_reward=-20.0
):
    """
    Handles reward shaping and Q-table update when an anomaly is detected.
    Optionally allows teacher (user) override for the anomaly reward.

    Args:
        current_state: Current state tuple.
        action: Action tuple.
        next_state: Next state tuple.
        q_table: Q-table (np.ndarray).
        state_to_index: State-to-index mapping.
        action_to_index: Action-to-index mapping.
        learning_rate: Q-learning alpha.
        discount_factor: Q-learning gamma.
        anomaly_detected (bool): Whether an anomaly was detected.
        teacher_override_enabled (bool): If True, prompt user for custom reward.
        default_anomaly_reward (float): Default penalty for anomaly.

    Returns:
        float: The reward used for the Q-table update.
    """
    reward = default_anomaly_reward if anomaly_detected else 0.0

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
    return reward

def enough_anomalies(chart_state, threshold=2):
    """Return True if enough anomalies in buffer to trigger adaptation."""
    return sum(chart_state.get('anomaly_buffer', [])) >= threshold

if __name__ == "__main__":
    # Example usage of Shewhart control chart utilities

    # 1. Initialize chart state
    chart_state = initialize_control_chart(window_size=10)

    # 2. Simulate adding engagement data
    example_engagement_rates = [0.7, 0.8, 0.75, 0.9, 0.85, 0.95, 0.6, 0.65, 0.8, 0.92, 0.5]
    for rate in example_engagement_rates:
        add_engagement_data(chart_state, rate)
        print(f"Added engagement rate: {rate:.2f}")

    # 3. Check for anomaly
    anomaly = check_for_engagement_anomaly(chart_state)
    print(f"Anomaly detected: {'Yes' if anomaly else 'No'}")

    # 4. Plot and save the control chart
    plot_shewhart_chart(chart_state, filename_prefix='example_shewhart_chart')

    # 5. Example of getting adjusted decision (interactive, for CLI use)
    # current_state = (3, 4, 1, 'B')
    # recommended_action = ('C', 7)
    # adjusted_action, fig = get_adjusted_decision(chart_state, current_state, recommended_action)
    # print(f"Adjusted action: {adjusted_action}")
