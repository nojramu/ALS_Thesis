import matplotlib.pyplot as plt # Import matplotlib
import pandas as pd # Import pandas
import numpy as np # Import numpy for apply_kalman_filter
import os
from datetime import datetime
from b2_kf_filter import apply_kalman_filter # Import apply_kalman_filter function
from b5_data_loader import load_predictions

df_predictions = load_predictions()
if df_predictions is None:
    exit(1)

# Assuming apply_kalman_filter is defined in a previous cell
# Make sure to run the cell with apply_kalman_filter function definition first

# Load Measurement and Store Smooth values (from d05b86fc)
# This part was originally in cell d05b86fc, moving it here to ensure data exists
if 'cognitive_load' in df_predictions.columns:
    cognitive_load_measurements = df_predictions['cognitive_load'].values
    # Check if apply_kalman_filter is defined before calling
    if 'apply_kalman_filter' in globals():
        smoothed_cognitive_load = apply_kalman_filter(cognitive_load_measurements)
        df_predictions['smoothed_cognitive_load'] = smoothed_cognitive_load

        # Create a Matplotlib figure and axes
        fig, ax = plt.subplots()

        # Add a scatter trace for the original 'cognitive_load' data
        ax.plot(df_predictions.index, df_predictions['cognitive_load'], label='Original Cognitive Load')

        # Add a scatter trace for the 'smoothed_cognitive_load' data
        ax.plot(df_predictions.index, df_predictions['smoothed_cognitive_load'], label='Smoothed Cognitive Load')

        # Update the layout with title and axis labels
        ax.set_title('Original vs. Smoothed Cognitive Load')
        ax.set_xlabel('Row Number')
        ax.set_ylabel('Cognitive Load')

        # Add a legend
        ax.legend()

        # Ensure image directory exists
        image_dir = os.path.join(os.path.dirname(__file__), '..', 'image')
        os.makedirs(image_dir, exist_ok=True)

        # Save the figure with timestamp
        filename = f"kf_filtered_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        filepath = os.path.join(image_dir, filename)
        plt.savefig(filepath)
        plt.close()

        print(f"Plot saved to {filepath}")
    else:
        print("Error: 'apply_kalman_filter' function not found. Please ensure the Kalman Filter cell is executed.")
else:
    print("Error: 'cognitive_load' column not found in the loaded data.")