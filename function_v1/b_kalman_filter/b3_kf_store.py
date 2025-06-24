import pandas as pd
from b2_kf_filter import apply_kalman_filter

def load_predictions(csv_path='data/sample_predictions.csv'):
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

df_predictions = load_predictions()
if df_predictions is None:
    exit(1)

cognitive_load_measurements = df_predictions['cognitive_load'].values
df_predictions['smoothed_cognitive_load'] = apply_kalman_filter(cognitive_load_measurements)
print(df_predictions.head())