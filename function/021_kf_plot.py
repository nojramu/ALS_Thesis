import matplotlib.pyplot as plt # Import matplotlib
import pandas as pd # Import pandas

# Load sample predictions data (assuming the file path is correct and accessible)
csv_file_path2 = '/content/drive/MyDrive/The Paper/Numerical/Code/sample_predictions.csv'
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

    # Display the Matplotlib figure
    plt.show()
else:
    print("Skipping plot due to data loading error.")