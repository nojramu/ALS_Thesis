import numpy as np

def apply_kalman_filter(measurements, initial_covariance=1000.0, process_noise=0.1, measurement_noise=1.0):
    """
    Applies a Kalman filter to a sequence of measurements with adjustable parameters.

    Args:
        measurements (np.array): A 1D numpy array of measurements.
        initial_covariance (float, optional): The initial uncertainty in the state estimate (P). Defaults to 1000.0.
        process_noise (float, optional): The covariance of the process noise (Q). Defaults to 0.1.
        measurement_noise (float, optional): The covariance of the measurement noise (R). Defaults to 1.0.


    Returns:
        np.array: A 1D numpy array of smoothed values.
    """
    n_measurements = len(measurements)
    smoothed_values = np.zeros(n_measurements)

    # Initialize state estimate and covariance
    # State [x] - the estimated true value
    # Covariance [P] - the uncertainty in the state estimate
    x_hat = 0.0  # Initial state estimate
    P = initial_covariance # Initial covariance (high uncertainty)

    # Define system parameters (simplified for a static system with noise)
    # State transition matrix [A] - assumes the state doesn't change over time
    A = 1.0
    # Control input matrix [B] - no control input
    B = 0.0
    # Measurement matrix [H] - relates the state to the measurement
    H = 1.0
    # Process noise covariance [Q] - uncertainty in the system model
    Q = process_noise # Small value assuming the true value is relatively stable
    # Measurement noise covariance [R] - uncertainty in the measurements
    R = measurement_noise # Adjust based on expected measurement noise

    for k in range(n_measurements):
        # Prediction Step
        # Predicted state estimate
        x_hat_minus = A * x_hat + B * 0 # No control input (u=0)
        # Predicted covariance
        P_minus = A * P * A + Q

        # Update Step
        # Kalman Gain
        K = P_minus * H / (H * P_minus * H + R)

        # Updated state estimate
        x_hat = x_hat_minus + K * (measurements[k] - H * x_hat_minus)

        # Updated covariance
        P = (1 - K * H) * P_minus

        # Store the smoothed value
        smoothed_values[k] = x_hat

    return smoothed_values