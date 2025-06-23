import numpy as np

def simpsons_rule(y, h):
    """
    Applies Simpson's Rule for numerical integration.

    Args:
        y (np.array): A 1D numpy array of function values (the data points).
        h (float): The step size (the distance between consecutive data points).

    Returns:
        float: The approximate value of the integral.
        None: If an error occurs (e.g., less than 3 points after dropping).
    """
    n = len(y)

    # Check if the number of data points is even and drop the first point if necessary
    if n % 2 == 0:
        print("Number of data points is even. Dropping the first data point to apply Simpson's Rule.")
        y = y[1:]
        n = len(y) # Update n after dropping the point

    if n < 3:
        print("Error: Simpson's Rule requires at least 3 points (after potential dropping).")
        return None

    integral = y[0] + y[n-1]
    for i in range(1, n - 1, 2):
        integral += 4 * y[i]
    for i in range(2, n - 2, 2):
        integral += 2 * y[i]

    integral = integral * h / 3
    return integral