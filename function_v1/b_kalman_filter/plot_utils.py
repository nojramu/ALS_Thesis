import matplotlib.pyplot as plt
import os
from datetime import datetime
from data_utils import ensure_dir_exists

def plot_cognitive_load(df, original_col, smoothed_col=None, image_dir='../image', prefix='kf'):
    fig, ax = plt.subplots()
    ax.plot(df.index, df[original_col], label='Original Cognitive Load')
    if smoothed_col and smoothed_col in df.columns:
        ax.plot(df.index, df[smoothed_col], label='Smoothed Cognitive Load')
    ax.set_title('Cognitive Load')
    ax.set_xlabel('Row Number')
    ax.set_ylabel('Cognitive Load')
    ax.legend()
    # Ensure image directory exists
    image_dir = os.path.join(os.path.dirname(__file__), image_dir)
    ensure_dir_exists(image_dir)
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    filepath = os.path.join(image_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved to {filepath}")