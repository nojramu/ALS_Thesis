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

    # Use historical data to define the range if available
    if historical_integral_values is not None and len(historical_integral_values) > 1:
        min_range = np.min(historical_integral_values)
        max_range = np.max(historical_integral_values)
        print(f"Using historical data range for discretization: [{min_range:.2f}, {max_range:.2f}]")
    else:
        # If no historical data, use a dynamic range based on the input value
        min_range = 0
        max_range = simpsons_integral_value * 2 if simpsons_integral_value > 0 else 100
        print(f"Using dynamic range based on input value for discretization: [{min_range:.2f}, {max_range:.2f}]")

    # Ensure the range is valid
    if min_range >= max_range:
        print(f"Warning: Calculated range [{min_range:.2f}, {max_range:.2f}] is invalid. Using a default range based on input value.")
        min_range = 0
        max_range = simpsons_integral_value + 1
        if max_range <= min_range:
            max_range = min_range + 1

    # Create equally spaced bins for discretization
    bins = np.linspace(min_range, max_range, num_buckets + 1).tolist()
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
        bucket_index = bucket_index_category[0]
        bucket_index = int(np.clip(bucket_index, 0, num_buckets - 1))

    return bucket_index + 1  # Return 1-based bucket index

# Example usage:
# if __name__ == "__main__":
#     from data_handling import load_csv
#     # Load the smoothed predictions
#     df = load_csv('data/sample_predictions_smoothed.csv')
#     if df is not None and 'smoothed_cognitive_load' in df.columns:
#         y = df['smoothed_cognitive_load'].values  # Extract the relevant column as a numpy array
#         h = 3  # Example step size
#         simpsons_integral = simpsons_rule(y, h)
#         print(f"Simpson's Rule Integral (smoothed): {simpsons_integral}")

#         # Discretize the result into 5 buckets
#         bucket_number = discretize_simpsons_result(simpsons_integral, num_buckets=5)
#         print(f"Discretized Bucket Number: {bucket_number}") 

#         # Discretize with different number of buckets
#         bucket_number_alt = discretize_simpsons_result(simpsons_integral, num_buckets=7)
#         print(f"Discretized Bucket Number (7 buckets): {bucket_number_alt}")