import pandas as pd
import os

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

def ensure_dir_exists(directory):
    os.makedirs(directory, exist_ok=True)