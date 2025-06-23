import pandas as pd

def load_and_preprocess_data(csv_file_path=None, data=None):
    """
    Loads data from a CSV file or accepts a DataFrame and performs preprocessing,
    including initial cleaning.

    Args:
      csv_file_path (str, optional): The path to the CSV file containing the data.
                                     If None, 'data' must be provided.
      data (pd.DataFrame, optional): A DataFrame containing the data.
                                   If None, 'csv_file_path' must be provided.

    Returns:
      Cleaned and preprocessed pandas DataFrame, or None if an error occurs,
      required columns are missing, or both csv_file_path and data are None.
    """
    df = None
    if csv_file_path:
        # Load the data from CSV
        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {csv_file_path}")
            return None
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
    elif data is not None:
        # Use the provided DataFrame
        df = data.copy() # Work on a copy to avoid modifying the original
    else:
        print("Error: Either csv_file_path or data must be provided.")
        return None


    if df is None:
        return None

    # Check if required columns exist
    required_features = ['engagement_rate', 'time_on_task_s', 'hint_ratio', 'interaction_count',
                         'task_completed', 'quiz_score', 'difficulty', 'error_rate',
                         'task_timed_out', 'time_before_hint_used']

    # If it's training data, also check for target columns
    is_training_data = 'engagement_level' in df.columns and 'cognitive_load' in df.columns
    if is_training_data:
         required_cols = required_features + ['engagement_level', 'cognitive_load']
    else:
         required_cols = required_features # Only features are required for new data


    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing required columns in the data: {missing_cols}")
        return None

    # Preprocess the data: Convert to numeric and fill missing values
    # Apply to columns expected to be numeric
    numeric_cols_to_fill = [
        'engagement_rate', 'time_on_task_s', 'hint_ratio', 'interaction_count',
        'quiz_score', 'difficulty', 'error_rate', 'time_before_hint_used'
    ]
    if is_training_data:
        numeric_cols_to_fill.append('cognitive_load') # Include cognitive_load for training data


    for col in numeric_cols_to_fill:
         if col in df.columns: # Check if column exists before processing
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median()) # Use median for robustness

    # Convert boolean/integer columns to integer (if they exist and are not already numeric)
    int_cols = ['task_completed', 'task_timed_out']
    if is_training_data:
        int_cols.append('engagement_level') # Include engagement_level for training data

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col in ['task_completed', 'task_timed_out']:
                 df[col] = df[col].fillna(0).astype(int)
            elif col == 'engagement_level':
                 df[col] = df[col].fillna(df[col].median()).astype(int)

    return df