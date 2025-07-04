import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

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

    # Check if the number of data points is less than 3
    if n < 3:
        print("Error: Simpson's Rule requires at least 3 points.")
        return None
    
    # Check if the number of data points is even and drop the first point if necessary
    if n % 2 == 0:
        print("Number of data points is even. Averaging the first two data points to apply Simpson's Rule.")
        y = np.concatenate([(y[0] + y[1:2]).mean().reshape(1), y[2:]])  # Replace first two with their average
        n = len(y)

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
    # Use historical data if available and valid
    if historical_integral_values is not None and len(historical_integral_values) > 1:
        min_range = np.min(historical_integral_values)
        max_range = np.max(historical_integral_values)
        print(f"Using historical data range for discretization: [{min_range:.2f}, {max_range:.2f}]")
    else:
        # Fallback: Use a dynamic range based on the input value
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

def quantitative_analysis_simpsons_rule(y, h):
    """
    Perform quantitative analysis on Simpson's Rule impact by comparing it with
    the trapezoidal rule on the same data.

    Args:
        y (array-like): 1D array or list of function values (the data points).
        h (float): The step size (distance between consecutive data points).

    Returns:
        dict: Dictionary containing Simpson's integral, trapezoidal integral,
              absolute difference, and relative difference.
    """
    y = np.asarray(y)
    simpson_integral = simpsons_rule(y, h)
    trapezoidal_integral = trapezoid(y, dx=h)

    if simpson_integral is None:
        print("Simpson's Rule integration failed.")
        return None

    abs_diff = abs(simpson_integral - trapezoidal_integral)
    rel_diff = abs_diff / abs(trapezoidal_integral) if trapezoidal_integral != 0 else None

    result = {
        'simpson_integral': simpson_integral,
        'trapezoidal_integral': trapezoidal_integral,
        'absolute_difference': abs_diff,
        'relative_difference': rel_diff
    }
    return result

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

        # Perform quantitative analysis comparing Simpson's Rule and trapezoidal rule
        analysis_result = quantitative_analysis_simpsons_rule(smoothed_cognitive_load, h)
        if analysis_result:
            print("Quantitative Analysis of Simpson's Rule Impact:")
            print(f"Simpson's Integral: {analysis_result['simpson_integral']:.4f}")
            print(f"Trapezoidal Integral: {analysis_result['trapezoidal_integral']:.4f}")
            print(f"Absolute Difference: {analysis_result['absolute_difference']:.4f}")
            if analysis_result['relative_difference'] is not None:
                print(f"Relative Difference: {analysis_result['relative_difference']:.4f}")
            else:
                print("Relative Difference: Undefined (division by zero)")

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