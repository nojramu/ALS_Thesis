import pandas as pd
csv_file_path2 = '/content/drive/MyDrive/The Paper/Numerical/Code/sample_predictions.csv'
try:
    df_predictions = pd.read_csv(csv_file_path2)
    display(df_predictions.head())
    df_predictions.info()
except FileNotFoundError:
    print(f"Error: File not found at {csv_file_path2}")
except Exception as e:
    print(f"Error loading CSV file: {e}")