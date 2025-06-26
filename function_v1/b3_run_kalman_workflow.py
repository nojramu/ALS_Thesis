from b0_data_utils import load_predictions
from b2_kf_filter import apply_kalman_filter
from b1_plot_utils import plot_cognitive_load

def main():
    df = load_predictions()
    if df is None or 'cognitive_load' not in df.columns:
        print("Error: Data not loaded or 'cognitive_load' column missing.")
        return

    # Plot original
    plot_cognitive_load(df, original_col='cognitive_load', smoothed_col=None, prefix='kf_unfiltered')

    # Apply Kalman filter
    df['smoothed_cognitive_load'] = apply_kalman_filter(df['cognitive_load'].values)

    # Plot smoothed
    plot_cognitive_load(df, original_col='cognitive_load', smoothed_col='smoothed_cognitive_load', prefix='kf_filtered')

if __name__ == "__main__":
    main()