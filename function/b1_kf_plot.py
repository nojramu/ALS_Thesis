import matplotlib.pyplot as plt # Import matplotlib
import pandas as pd # Import pandas
import os
from datetime import datetime

# Load sample predictions data (assuming the file path is correct and accessible)
csv_file_path2 = 'data/sample_predictions.csv'  # Adjust the path as needed
try:
    df_predictions = pd.read_csv(csv_file_path2)
except FileNotFoundError:
    print(f"Error: File not found at {csv_file_path2}")
    df_predictions = None # Set to None if file not found
except Exception as e:
    print(f"Error loading CSV file: {e}")
    df_predictions = None # Set to None if other error occurs

if df_predictions is not None:
    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots()

    # Add a scatter trace for the original 'cognitive_load' data
    ax.plot(df_predictions.index, df_predictions['cognitive_load'], label='Cognitive Load')

    # Update the layout with title and axis labels
    ax.set_title('Cognitive Load by Row')
    ax.set_xlabel('Row')
    ax.set_ylabel('Cognitive Load')

    # Add a legend
    ax.legend()

    # Ensure image directory exists
    image_dir = os.path.join(os.path.dirname(__file__), '..', 'image')
    os.makedirs(image_dir, exist_ok=True)

    # Save the figure with timestamp
    filename = f"kf_unfiltered_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    filepath = os.path.join(image_dir, filename)
    plt.savefig(filepath)
    plt.close()

    print(f"Plot saved to {filepath}")
else:
    print("Skipping plot due to data loading error.")