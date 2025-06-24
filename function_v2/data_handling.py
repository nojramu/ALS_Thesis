import pandas as pd
import os

def load_csv(csv_path):
    """Load a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def save_csv(df, csv_path):
    """Save a DataFrame to a CSV file."""
    try:
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

def ensure_dir_exists(directory):
    """Ensure a directory exists."""
    os.makedirs(directory, exist_ok=True)

def preprocess_data(df, required_features, is_training_data=False):
    """
    Preprocess DataFrame: check columns, convert types, fill missing values.
    """
    if df is None:
        return None

    # Check required columns
    if is_training_data:
        required_cols = required_features + ['engagement_level', 'cognitive_load']
    else:
        required_cols = required_features

    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing columns: {missing}")
        return None

    # Numeric columns
    numeric_cols = [
        'engagement_rate', 'time_on_task_s', 'hint_ratio', 'interaction_count',
        'quiz_score', 'difficulty', 'error_rate', 'time_before_hint_used'
    ]
    if is_training_data:
        numeric_cols.append('cognitive_load')

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # Integer columns
    int_cols = ['task_completed', 'task_timed_out']
    if is_training_data:
        int_cols.append('engagement_level')
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col in ['task_completed', 'task_timed_out']:
                df[col] = df[col].fillna(0).astype(int)
            elif col == 'engagement_level':
                df[col] = df[col].fillna(df[col].median()).astype(int)

    return df

'''Example usage:
if __name__ == "__main__":
    # Load data
    df = load_csv('data/sample_data.csv')
    
    # Preprocess data
    required_features = ['engagement_rate', 'time_on_task_s', 'hint_ratio', 'interaction_count']
    df_processed = preprocess_data(df, required_features, is_training_data=True)
    
    if df_processed is not None:
        # Save processed data
        save_csv(df_processed, 'data/processed_data.csv')
'''