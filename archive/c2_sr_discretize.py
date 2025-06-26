import pandas as pd
import numpy as np # Ensure numpy is imported

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
                                   the input value and estimated max is used (less robust).

    Returns:
        int: The bucket number (1-based index).
             Returns None if simpsons_integral_value is None or num_buckets < 2.
    """
    if simpsons_integral_value is None or num_buckets < 2:
        print("Error: Invalid input for discretization.")
        return None

    # --- IMPROVEMENT: Use a more robust method for defining the discretization range ---
    # Option A: Use historical data to define the range
    if historical_integral_values is not None and len(historical_integral_values) > 1:
        min_range = np.min(historical_integral_values)
        max_range = np.max(historical_integral_values)
        print(f"Using historical data range for discretization: [{min_range:.2f}, {max_range:.2f}]")
    # Option B: Define a fixed range based on domain knowledge or prior analysis
    # elif some_fixed_min is not None and some_fixed_max is not None:
    #    min_range = some_fixed_min
    #    max_range = some_fixed_max
    #    print(f"Using fixed range for discretization: [{min_range:.2f}, {max_range:.2f}]")
    # Option C (Fallback - original less robust dynamic range):
    else:
        # Use the range of 'smoothed_cognitive_load' as a hint for the integral's scale:
        # Assuming non-negative cognitive load, min possible integral is 0.
        min_range = 0

        # Attempt to use the global variable cognitive_load_values if available
        # This is not ideal as it relies on a global variable, but matches the original code's potential context
        global cognitive_load_values # Declare intent to use global variable
        cognitive_load_values = globals().get('cognitive_load_values', None)
        if cognitive_load_values is not None and len(cognitive_load_values) > 0:
             # A rough estimate of max integral: max_smoothed_load * number of steps * h
             # Need the step size 'h' as well. If 'h' is a global, try to use it.
             h = globals().get('h', None)
             if h is not None:
                 max_range_estimate = np.max(cognitive_load_values) * len(cognitive_load_values) * h
                 # If simpsons_integral_value exceeds this rough max, adjust the max
                 max_range = max_range_estimate * 1.1 if simpsons_integral_value > max_range_estimate else max_range_estimate
                 print(f"Using dynamic range based on global data/h for discretization: [{min_range:.2f}, {max_range:.2f}]")
             else:
                 # Fallback if h is not available globally
                 max_range = simpsons_integral_value * 2 if simpsons_integral_value > 0 else 100 # Ensure a positive max
                 print(f"Using dynamic range based on input value for discretization: [{min_range:.2f}, {max_range:.2f}]")
        else:
             # Fallback if cognitive_load_values is not available globally
             max_range = simpsons_integral_value * 2 if simpsons_integral_value > 0 else 100 # Ensure a positive max
             print(f"Using dynamic range based on input value for discretization: [{min_range:.2f}, {max_range:.2f}]")


    # Ensure min is less than max for bucket creation
    if min_range >= max_range:
        # Fallback if calculated range is invalid
        print(f"Warning: Calculated range [{min_range:.2f}, {max_range:.2f}] is invalid. Using a default range based on input value.")
        min_range = 0 # Assuming non-negative
        max_range = simpsons_integral_value + 1 # Use the value itself plus a small buffer
        if max_range <= min_range: # Ensure max is strictly greater than min
            max_range = min_range + 1


    # Define the bucket edges
    bins = np.linspace(min_range, max_range, num_buckets + 1)
    bins = bins.tolist()  # Convert numpy array to list for pd.cut compatibility

    # Find which bucket the integral value falls into
    # Use pd.cut to assign the value to a bin
    # We need to put the single value into a Series or DataFrame to use pd.cut
    integral_series = pd.Series([simpsons_integral_value])
    # labels argument can be False or None to just return the bin index
    # right=True means intervals are like (a, b], include_lowest=True handles the lowest bound.
    # If value == max_range, it falls into the last bin.
    bucket_index_category = pd.cut(integral_series, bins=bins, include_lowest=True, labels=False)

    # pd.cut returns a categorical type, get the index
    # Handle the case where the value is exactly at the edge or outside the range if include_lowest is False
    # With include_lowest=True, the first bin is inclusive of the lower bound.
    # If the value is outside the range, pd.cut might return NaN.
    if bucket_index_category.isnull().any():
        # This should ideally not happen with a well-defined range, but as a safeguard:
        print(f"Warning: Simpson's integral value {simpsons_integral_value} is outside the calculated range [{min_range:.2f}, {max_range:.2f}]. Assigning to the closest bucket.")
        # Assign to the closest bucket - either the first or the last
        if simpsons_integral_value < min_range:
            bucket_index = 0
        else: # Value is greater than max_range
             bucket_index = num_buckets - 1
    else:
        # Get the integer index of the bucket (0-based)
        bucket_index = bucket_index_category[0]

        # Ensure the index is within valid bounds [0, num_buckets - 1]
        bucket_index = int(np.clip(bucket_index, 0, num_buckets - 1))


    # Return the bucket number (1-based index)
    return bucket_index + 1