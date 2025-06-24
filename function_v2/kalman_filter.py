import numpy as np
import pandas as pd

def apply_kalman_filter(
    measurements,
    initial_covariance=1000.0,
    process_noise=0.1,
    measurement_noise=1.0,
    initial_state=None
):
    """
    Applies a Kalman filter to a sequence of measurements.

    Args:
        measurements (array-like): Sequence of measurements (list, np.array, or pd.Series).
        initial_covariance (float): Initial uncertainty in the state estimate (P).
        process_noise (float): Covariance of the process noise (Q).
        measurement_noise (float): Covariance of the measurement noise (R).
        initial_state (float or None): Optional initial state estimate. If None, uses first measurement.

    Returns:
        np.array: Smoothed values.
    """
    measurements = np.asarray(measurements)
    n = len(measurements)
    smoothed = np.zeros(n)

    # Initialize state estimate and covariance
    x_hat = measurements[0] if initial_state is None else initial_state
    P = initial_covariance

    A = 1.0  # State transition
    H = 1.0  # Measurement matrix
    Q = process_noise
    R = measurement_noise

    for k in range(n):
        # Prediction
        x_hat_minus = A * x_hat
        P_minus = A * P * A + Q

        # Update
        K = P_minus * H / (H * P_minus * H + R)
        x_hat = x_hat_minus + K * (measurements[k] - H * x_hat_minus)
        P = (1 - K * H) * P_minus

        smoothed[k] = x_hat

    return smoothed

def add_kalman_column(df, col='cognitive_load', new_col='smoothed_cognitive_load', **kf_kwargs):
    """
    Adds a column with Kalman-filtered values to a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column to smooth.
        new_col (str): Name for the new column.
        **kf_kwargs: Additional arguments for apply_kalman_filter.

    Returns:
        pd.DataFrame: DataFrame with new column.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    df = df.copy()
    df[new_col] = apply_kalman_filter(df[col].values, **kf_kwargs)
    return df

# Example usage:
if __name__ == "__main__":
    from data_handling import load_csv, save_csv, display_csv_head
    from plot_responsive import plot_line_chart, save_figure_to_image_folder

    df = load_csv('data/sample_predictions.csv')
    if df is not None:
        df = add_kalman_column(df, col='cognitive_load', new_col='smoothed_cognitive_load')
        display_csv_head(df)
        save_csv(df, 'data/sample_predictions_smoothed.csv')

        # Plot original and filtered cognitive load using plot_responsive
        fig = plot_line_chart(
            x=df.index,
            y=[df['cognitive_load'].values, df['smoothed_cognitive_load'].values],
            xlabel='Row Number',
            ylabel='Cognitive Load',
            title='Original vs. Kalman-Filtered Cognitive Load',
            legend_labels=['Original', 'Kalman Filtered'],
            show=False
        )
        save_figure_to_image_folder(fig, prefix='kalman_filtered', image_dir='image')