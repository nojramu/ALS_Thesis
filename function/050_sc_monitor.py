import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Used for visualization
import random # Assuming random might be used for simulating new data in tests, though not strictly in the FeedbackInterface logic itself.
import sys # To handle input from user


def initialize_control_chart(window_size=10):
    """
    Initializes the Shewhart Control Chart data structure.

    Args:
        window_size (int): The number of recent data points to use for calculating
                           CL, UCL, and LCL. Defaults to 10.

    Returns:
        dict: A dictionary representing the control chart state, including:
              - 'window_size' (int): The size of the rolling window.
              - '_engagement_data' (list): Stores data within the window.
              - 'cl' (float or None): Central Line.
              - 'ucl' (float or None): Upper Control Limit.
              - 'lcl' (float or None): Lower Control Limit.
              - '_anomalies' (list): Stores indices relative to the current window data.
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    return {
        'window_size': window_size,
        '_engagement_data': [],
        'cl': None,
        'ucl': None,
        'lcl': None,
        '_anomalies': []
    }

def calculate_control_limits(chart_state: dict):
    """
    Calculates and updates the Central Line (CL), Upper Control Limit (UCL),
    and Lower Control Limit (LCL) based on the data in the rolling window
    within the provided chart state. Also updates the list of anomalies.
    Requires at least 2 data points in the window to calculate standard deviation.

    Args:
        chart_state (dict): The dictionary representing the control chart state
                            to be updated in-place.
    """
    data_window = chart_state['_engagement_data']
    if len(data_window) < 2:
        chart_state['cl'] = None
        chart_state['ucl'] = None
        chart_state['lcl'] = None
        chart_state['_anomalies'] = [] # Clear anomalies if not enough data
        return

    # Calculate based on the data currently in the window
    data_array = np.array(data_window)

    chart_state['cl'] = np.mean(data_array)
    std_dev = np.std(data_array) # Using population std dev (default for np.std)

    chart_state['ucl'] = chart_state['cl'] + 3 * std_dev
    chart_state['lcl'] = chart_state['cl'] - 3 * std_dev

    # Ensure LCL is not below 0 and UCL is not above 1.0 for engagement rate (0-1)
    chart_state['lcl'] = max(0.0, chart_state['lcl'])
    chart_state['ucl'] = min(1.0, chart_state['ucl'])

    # Update anomalies based on the *current* window data and the *newly calculated* limits
    chart_state['_anomalies'] = [i for i, rate in enumerate(data_window)
                           if rate > chart_state['ucl'] or rate < chart_state['lcl']]


def add_engagement_data(chart_state: dict, engagement_rate: float):
    """
    Adds a new engagement rate data point to the chart state, maintains the
    rolling window, and recalculates limits and anomalies for the current window.

    Args:
        chart_state (dict): The dictionary representing the control chart state
                            to be updated in-place.
        engagement_rate (float): The new engagement rate value (expected to be between 0 and 1).
    """
    if not isinstance(engagement_rate, (int, float)):
         print(f"Warning: Received non-numeric engagement rate: {engagement_rate}. Skipping.")
         return

    chart_state['_engagement_data'].append(engagement_rate)
    # Maintain rolling window size
    if len(chart_state['_engagement_data']) > chart_state['window_size']:
        chart_state['_engagement_data'].pop(0) # Remove the oldest data point

    # Always recalculate limits and anomalies after adding data
    calculate_control_limits(chart_state)


def check_for_engagement_anomaly(chart_state: dict) -> bool:
    """
    Checks if the latest data point added to the chart state is an anomaly
    based on the current control limits.

    Args:
        chart_state (dict): The dictionary representing the control chart state.

    Returns:
        bool: True if the latest rate is an anomaly (outside UCL/LCL),
              False otherwise. Returns False if not enough data for limits.
    """
    if chart_state['cl'] is None or len(chart_state['_engagement_data']) == 0:
        return False

    latest_engagement_rate = chart_state['_engagement_data'][-1]

    return latest_engagement_rate > chart_state['ucl'] or latest_engagement_rate < chart_state['lcl']


def get_control_chart_data(chart_state: dict) -> dict:
    """
    Returns the current chart data including engagement rates and control limits
    from the provided chart state.

    Args:
        chart_state (dict): The dictionary representing the control chart state.

    Returns:
        dict: A dictionary containing:
              - 'engagement_rates': List of engagement rates in the current window.
              - 'cl': Central Line value.
              - 'ucl': Upper Control Limit value.
              - 'lcl': Lower Control Limit value.
              - 'anomalies': List of indices (relative to the current window data) of anomalous points.
              Returns None if there is not enough data to calculate limits.
    """
    if chart_state['cl'] is None or chart_state['ucl'] is None or chart_state['lcl'] is None:
        print("Not enough data to get chart data.")
        return None

    return {
        'engagement_rates': chart_state['_engagement_data'],
        'cl': chart_state['cl'],
        'ucl': chart_state['ucl'],
        'lcl': chart_state['lcl'],
        'anomalies': chart_state['_anomalies']
    }


def visualize_control_chart(chart_state: dict):
    """
    Creates a Matplotlib line chart of the engagement rate over time (within the window)
    from the provided chart state, including lines for CL, UCL, and LCL, and highlights anomalies.

    Args:
        chart_state (dict): The dictionary representing the control chart state.

    Returns:
        matplotlib.figure.Figure or None: The Matplotlib figure object, or None if
                                        there is not enough data to visualize.
    """
    chart_data = get_control_chart_data(chart_state)
    if chart_data is None:
        print("Not enough data to visualize the chart.")
        return None

    engagement_rates = chart_data['engagement_rates']
    cl = chart_data['cl']
    ucl = chart_data['ucl']
    lcl = chart_data['lcl']
    anomalies = chart_data['anomalies']
    window_size = chart_state['window_size'] # Get window size from state

    fig, ax = plt.subplots()

    # Add engagement rate data
    ax.plot(list(range(len(engagement_rates))), engagement_rates, marker='o', linestyle='-', label='Engagement Rate')

    # Add CL, UCL, LCL lines
    x_range = list(range(len(engagement_rates)))
    ax.axhline(cl, color='green', linestyle='--', label='Central Line (CL)')
    ax.axhline(ucl, color='red', linestyle='--', label='Upper Control Limit (UCL)')
    ax.axhline(lcl, color='red', linestyle='--', label='Lower Control Limit (LCL)')

    # Highlight anomalies
    # Use the indices stored in _anomalies, which are relative to the current window
    anomaly_x = [i for i in anomalies]
    anomaly_y = [engagement_rates[i] for i in anomalies] # Get the actual values at those indices
    if anomaly_x: # Only add if there are anomalies
         ax.plot(anomaly_x, anomaly_y, 'rx', markersize=10, label='Anomaly') # 'rx' means red 'x' markers


    # Update layout
    ax.set_title(f'Shewhart Control Chart (Rolling Window {window_size})')
    ax.set_xlabel('Task Index (within window)')
    ax.set_ylabel('Engagement Rate')
    ax.legend(loc='upper right') # Adjust legend location

    return fig # Return the figure object for display