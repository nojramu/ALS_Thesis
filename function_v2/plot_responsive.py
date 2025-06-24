import matplotlib.pyplot as plt
import os
from datetime import datetime
from io import BytesIO
from PIL import Image

def plot_line_chart(
    x, y, xlabel='X', ylabel='Y', title='Line Chart', legend_labels=None,
    save_path=None, show=False
):
    """
    Plots a line chart and optionally saves or returns the figure.

    Args:
        x (array-like): X-axis data.
        y (array-like or list of arrays): Y-axis data or list of series.
        xlabel (str): Label for X-axis.
        ylabel (str): Label for Y-axis.
        title (str): Plot title.
        legend_labels (list): Labels for each line (if y is a list).
        save_path (str): If provided, saves the plot to this path.
        show (bool): If True, displays the plot interactively.

    Returns:
        fig: The Matplotlib figure object.
    """
    fig, ax = plt.subplots()
    if isinstance(y, list) and legend_labels:
        for yi, label in zip(y, legend_labels):
            ax.plot(x, yi, label=label)
        ax.legend()
    elif isinstance(y, list):
        for yi in y:
            ax.plot(x, yi)
    else:
        ax.plot(x, y, label=legend_labels[0] if legend_labels else None)
        if legend_labels:
            ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig

def plot_bar_chart(
    x, y, xlabel='X', ylabel='Y', title='Bar Chart', save_path=None, show=False, rotation=45
):
    """
    Plots a bar chart and optionally saves or returns the figure.

    Args:
        x (array-like): X-axis categories.
        y (array-like): Y-axis values.
        xlabel (str): Label for X-axis.
        ylabel (str): Label for Y-axis.
        title (str): Plot title.
        save_path (str): If provided, saves the plot to this path.
        show (bool): If True, displays the plot interactively.
        rotation (int): Rotation for x-tick labels.

    Returns:
        fig: The Matplotlib figure object.
    """
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=rotation)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig

def fig_to_image(fig, as_bytes=False, save_path=None):
    """
    Converts a Matplotlib figure to a PNG image.

    Args:
        fig: Matplotlib figure.
        as_bytes (bool): If True, returns image as bytes.
        save_path (str): If provided, saves the image to this path.

    Returns:
        bytes or None: PNG image bytes if as_bytes is True, else None.
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    if save_path:
        with open(save_path, 'wb') as f:
            f.write(buf.getbuffer())
    if as_bytes:
        return buf.getvalue()
    else:
        return None

def image_bytes_to_pil(image_bytes):
    """
    Converts image bytes to a PIL Image object.

    Args:
        image_bytes (bytes): Image in bytes.

    Returns:
        PIL.Image.Image: PIL Image object.
    """
    return Image.open(BytesIO(image_bytes))

def save_figure_to_image_folder(fig, prefix='plot', image_dir='image', custom_name=None):
    """
    Saves a Matplotlib figure to the specified image folder with a timestamped or custom filename.

    Args:
        fig: Matplotlib figure.
        prefix (str): Prefix for the filename.
        image_dir (str): Directory to save the image in.
        custom_name (str): Custom filename (without extension). If provided, overrides prefix+timestamp.

    Returns:
        str: The path to the saved image.
    """
    os.makedirs(image_dir, exist_ok=True)
    if custom_name:
        filename = f"{custom_name}.png"
    else:
        filename = f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    save_path = os.path.join(image_dir, filename)
    fig.savefig(save_path)
    print(f"Plot saved to {save_path}")
    return save_path

'''Example usage:'''
# if __name__ == "__main__":
#     # Line chart example
#     fig = plot_line_chart(
#         x=[1, 2, 3, 4],
#         y=[[1, 2, 3, 4], [2, 3, 4, 5]],
#         xlabel='Time',
#         ylabel='Value',
#         title='Sample Line Chart',
#         legend_labels=['Series 1', 'Series 2'],
#         show=False
#     )
#     save_figure_to_image_folder(fig, prefix='line_chart', image_dir='image', custom_name='my_custom_plot')

#     # Bar chart example
#     fig = plot_bar_chart(
#         x=['A', 'B', 'C', 'D'],
#         y=[3, 7, 2, 5],
#         xlabel='Category',
#         ylabel='Values',
#         title='Sample Bar Chart',
#         show=False
#     )
#     save_figure_to_image_folder(fig, prefix='bar_chart', image_dir='image', custom_name='my_bar_chart')