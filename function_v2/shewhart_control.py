from plot_utils import plot_line_chart, save_figure_to_image_folder
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
