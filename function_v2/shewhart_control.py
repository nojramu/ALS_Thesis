from function_v2.plot_utils import plot_line_chart, save_figure_to_image_folder
import numpy as np

def initialize_control_chart(window_size=10):
    """
    Initialize the control chart state with a specified window size.
    """
    return {
        'window_size': window_size,
        'engagement_data': [],
        'cl': None,
        'ucl': None,
        'lcl': None,
        'anomalies': []
    }

def calculate_control_limits(chart_state):
    """
    Calculate the control limits (CL, UCL, LCL) for the current engagement data.
    Marks anomalies if data points are outside control limits.
    """
    data = chart_state['engagement_data']
    if len(data) < 2:
        chart_state['cl'] = chart_state['ucl'] = chart_state['lcl'] = None
        chart_state['anomalies'] = []
        return
    arr = np.array(data)
    chart_state['cl'] = np.mean(arr)
    std = np.std(arr, ddof=1)
    chart_state['ucl'] = min(1.0, chart_state['cl'] + 3 * std)
    chart_state['lcl'] = max(0.0, chart_state['cl'] - 3 * std)
    chart_state['anomalies'] = [i for i, v in enumerate(data) if v > chart_state['ucl'] or v < chart_state['lcl']]

def add_engagement_data(chart_state, engagement_rate):
    """
    Add a new engagement rate to the control chart, maintaining the window size.
    Recalculates control limits after adding.
    """
    chart_state['engagement_data'].append(engagement_rate)
    if len(chart_state['engagement_data']) > chart_state['window_size']:
        chart_state['engagement_data'].pop(0)
    calculate_control_limits(chart_state)

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
    Return the current control chart data for plotting or analysis.
    """
    if chart_state['cl'] is None or chart_state['ucl'] is None or chart_state['lcl'] is None:
        return None
    return {
        'engagement_rates': chart_state['engagement_data'],
        'cl': chart_state['cl'],
        'ucl': chart_state['ucl'],
        'lcl': chart_state['lcl'],
        'anomalies': chart_state['anomalies']
    }

def get_adjusted_decision(chart_state, current_state, recommended_action):
    """
    Displays the control chart, system recommendation, and allows user to adjust difficulty.
    Returns (adjusted_action, chart_fig)
    """
    print("\n--- Engagement Control Chart ---")
    chart_data = get_control_chart_data(chart_state)
    chart_fig = None
    if chart_data:
        x = list(range(len(chart_data['engagement_rates'])))
        y = [chart_data['engagement_rates']]
        legend = ['Engagement Rate']
        cl = [chart_data['cl']] * len(x)
        ucl = [chart_data['ucl']] * len(x)
        lcl = [chart_data['lcl']] * len(x)
        y.extend([cl, ucl, lcl])
        legend.extend(['CL', 'UCL', 'LCL'])
        chart_fig = plot_line_chart(
            x=x,
            y=y,
            xlabel='Task Index (within window)',
            ylabel='Engagement Rate',
            title='Shewhart Control Chart',
            legend_labels=legend,
            show=False
        )
        save_figure_to_image_folder(chart_fig, prefix='shewhart_chart', image_dir='image')
        print("Chart plotted and saved.")
    else:
        print("Not enough data to display the control chart.")

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
    

# Example usage:
# if __name__ == "__main__":
#     from q_learning import (
#     define_state_space, define_action_space, initialize_q_table,
#     epsilon_greedy_action_selection, update_q_table)
#     """
#     Example function to run a Q-learning loop with Shewhart feedback.
#     """
#     num_steps=15
#     epsilon=0.1
#     states, state_to_index, index_to_state, num_states = define_state_space()
#     actions, action_to_index, index_to_action, num_actions = define_action_space()
#     q_table = initialize_q_table(num_states, num_actions)
#     chart_state = initialize_control_chart(window_size=10)
#     current_state = states[0]
#     for step in range(num_steps):
#         action = epsilon_greedy_action_selection(current_state, q_table, state_to_index, index_to_action, action_to_index, epsilon=epsilon)
#         # Simulate environment (replace with your own logic)
#         next_state = states[(state_to_index[current_state] + 1) % num_states]
#         engagement_rate = next_state[1] / 5.0  # Example: normalize engagement_level
#         add_engagement_data(chart_state, engagement_rate)
#         is_anomaly = check_for_engagement_anomaly(chart_state)
#         reward = -1 if is_anomaly else 1
#         update_q_table(q_table, current_state, action, reward, next_state, 0.1, 0.9, state_to_index, action_to_index)
#         current_state = next_state
#         # Feedback interface (optional, e.g., every 5 steps)
#         if step % 5 == 0:
#             adjusted_action, chart_fig = get_adjusted_decision(chart_state, current_state, action)
#     print("Final Q-table:", q_table)
#     print("Final Chart State:", chart_state)