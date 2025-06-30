import numpy as np
import pandas as pd

def simpsons_rule(y, h):
    """
    Applies Simpson's Rule for numerical integration.

    Args:
        y (array-like): 1D array or list of function values (the data points).
        h (float): The step size (distance between consecutive data points).

    Returns:
        float: The approximate value of the integral.
        None: If an error occurs (e.g., less than 3 points after dropping).
    """
    y = np.asarray(y)  # Convert input to numpy array for easier manipulation
    n = len(y)         # Number of data points

    # Check if the number of data points is even and drop the first point if necessary
    if n % 2 == 0:
        print("Number of data points is even. Dropping the first data point to apply Simpson's Rule.")
        y = y[1:]      # Drop the first data point to make n odd
        n = len(y)

    if n < 3:
        print("Error: Simpson's Rule requires at least 3 points (after potential dropping).")
        return None

    # Apply Simpson's Rule formula
    integral = y[0] + y[-1]  # Add first and last terms
    integral += 4 * np.sum(y[1:n-1:2])  # Add 4 times odd-indexed terms
    integral += 2 * np.sum(y[2:n-2:2])  # Add 2 times even-indexed terms (excluding first and last)
    integral = integral * h / 3          # Multiply by h/3
    return integral

def discretize_simpsons_result(simpsons_integral_value, num_buckets=5, historical_integral_values=None):
    """
    Discretizes the result of Simpson's Rule into a specified number of buckets
    and returns the bucket number as an integer.

    Args:
        simpsons_integral_value (float): The result from the simpsons_rule function.
        num_buckets (int, optional): The number of buckets to discretize into.
                                     Defaults to 5. Must be at least 2.
        historical_integral_values (list or np.array, optional): A collection of
                                   historically observed integral values. If provided,
                                   the discretization range will be based on the min/max
                                   of these values. If None, a dynamic range based on
                                   the input value and estimated max is used.

    Returns:
        int: The bucket number (1-based index).
             Returns None if simpsons_integral_value is None or num_buckets < 2.
    """
    if simpsons_integral_value is None or num_buckets < 2:
        print("Error: Invalid input for discretization.")
        return None

    # --- Robust Range Logic ---
    # 1. Use historical data if available and valid
    if historical_integral_values is not None and len(historical_integral_values) > 1:
        min_range = np.min(historical_integral_values)
        max_range = np.max(historical_integral_values)
        print(f"Using historical data range for discretization: [{min_range:.2f}, {max_range:.2f}]")
    else:
        # 2. Try to use global cognitive_load_values and h if available
        cognitive_load_values = globals().get('cognitive_load_values', None)
        h = globals().get('h', None)
        if cognitive_load_values is not None and len(cognitive_load_values) > 0 and h is not None:
            max_range_estimate = np.max(cognitive_load_values) * len(cognitive_load_values) * h
            max_range = max(max_range_estimate * 1.1, simpsons_integral_value * 1.1)
            min_range = 0
            print(f"Using global cognitive_load_values and h for discretization: [{min_range:.2f}, {max_range:.2f}]")
        else:
            # 3. Fallback: Use a dynamic range based on the input value
            min_range = 0
            max_range = simpsons_integral_value * 2 if simpsons_integral_value > 0 else 100
            print(f"Using dynamic range based on input value for discretization: [{min_range:.2f}, {max_range:.2f}]")

    # Ensure min is less than max for bucket creation
    if min_range >= max_range:
        print(f"Warning: Calculated range [{min_range:.2f}, {max_range:.2f}] is invalid. Using a default range based on input value.")
        min_range = 0
        max_range = simpsons_integral_value + 1
        if max_range <= min_range:
            max_range = min_range + 1

    # Define the bucket edges
    bins = np.linspace(min_range, max_range, num_buckets + 1).tolist()

    # Assign the value to a bin using pd.cut
    integral_series = pd.Series([simpsons_integral_value])
    bucket_index_category = pd.cut(integral_series, bins=bins, include_lowest=True, labels=False)

    # Handle case where value is outside the bins
    if bucket_index_category.isnull().any():
        print(f"Warning: Simpson's integral value {simpsons_integral_value} is outside the calculated range [{min_range:.2f}, {max_range:.2f}]. Assigning to the closest bucket.")
        if simpsons_integral_value < min_range:
            bucket_index = 0
        else:
            bucket_index = num_buckets - 1
    else:
        bucket_index = int(np.clip(bucket_index_category[0], 0, num_buckets - 1))

    return bucket_index + 1  # Return 1-based bucket index

def redefine_state_space(num_buckets):
    simpsons_integral_levels = range(1, num_buckets + 1)
    # ... rest as in define_state_space ...

if __name__ == "__main__":
    from data_handling import load_csv

    # Load the CSV with smoothed cognitive load using data_handling utility
    df = load_csv("data/sample_predictions_with_smoothed.csv")

    if df is None:
        print("Failed to load data. Exiting.")
    else:
        # Use the smoothed cognitive load column
        smoothed_cognitive_load = df['smoothed_cognitive_load'].values
        h = 3  # Step size (adjust as appropriate for your data)

        # Apply Simpson's Rule
        integral = simpsons_rule(smoothed_cognitive_load, h)
        print(f"Simpson's Rule Integral (smoothed): {integral}")

        # Example: Discretize the integral into 5 buckets
        historical_integrals = None  # Or load/compute as needed
        bucket_5 = discretize_simpsons_result(integral, num_buckets=5, historical_integral_values=historical_integrals)
        print(f"Discretized Bucket (5 buckets): {bucket_5}")

        # Example: Discretize the integral into 7 buckets
        bucket_7 = discretize_simpsons_result(integral, num_buckets=7, historical_integral_values=historical_integrals)
        print(f"Discretized Bucket (7 buckets): {bucket_7}")