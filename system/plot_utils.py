import matplotlib.pyplot as plt
import os
from datetime import datetime
from io import BytesIO
from PIL import Image

import plotly.graph_objs as go
import plotly.express as px

# --- Matplotlib Utilities ---

def plot_line_chart(
    x, y, xlabel='X', ylabel='Y', title='Line Chart', legend_labels=None,
    save_path=None, show=False
):
    """
    Plots a line chart using Matplotlib and optionally saves or returns the figure.

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
    Plots a bar chart using Matplotlib and optionally saves or returns the figure.

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

# --- Plotly Utilities ---

def plotly_line_chart(x, y, xlabel='X', ylabel='Y', title='Line Chart', legend_labels=None, save_path=None, show=False):
    """
    Plots a line chart using Plotly and optionally saves or returns the figure.
    """
    fig = go.Figure()
    if isinstance(y, list) and legend_labels:
        for yi, label in zip(y, legend_labels):
            fig.add_trace(go.Scatter(x=x, y=yi, mode='lines', name=label))
    elif isinstance(y, list):
        for yi in y:
            fig.add_trace(go.Scatter(x=x, y=yi, mode='lines'))
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=legend_labels[0] if legend_labels else None))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    if save_path:
        fig.write_image(save_path)
        print(f"Plotly plot saved to {save_path}")
    if show:
        fig.show()
    return fig

def plotly_bar_chart(x, y, xlabel='X', ylabel='Y', title='Bar Chart', color='blue', save_path=None, show=False):
    """
    Plots a bar chart using Plotly and optionally saves or returns the figure.
    """
    fig = go.Figure([go.Bar(x=x, y=y, marker_color=color)])
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    if save_path:
        fig.write_image(save_path)
        print(f"Plotly bar chart saved to {save_path}")
    if show:
        fig.show()
    return fig

def plotly_heatmap(z, x=None, y=None, xlabel='X', ylabel='Y', title='Heatmap', show=False, colorbar_title="Value"):
    """
    Plots a heatmap using Plotly (for interactive display only).
    """
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorbar=dict(title=colorbar_title)
    ))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    if show:
        fig.show()
    return fig

def plotly_qtable_heatmap(q_table, index_to_state=None, index_to_action=None, show=False):
    """
    Plots a Q-table heatmap using Plotly with labeled axes.
    """
    import plotly.express as px

    # Create axis labels if mappings are provided
    y_labels = [str(index_to_state[i]) for i in range(q_table.shape[0])] if index_to_state else [str(i) for i in range(q_table.shape[0])]
    x_labels = [str(index_to_action[j]) for j in range(q_table.shape[1])] if index_to_action else [str(j) for j in range(q_table.shape[1])]

    fig = px.imshow(
        q_table,
        color_continuous_scale='Viridis',
        aspect='auto',
        labels=dict(x="Action (Task, Difficulty)", y="State (Simpson, Engagement, Completed, PrevTask)", color="Q-value"),
        title="Q-table Heatmap",
        x=x_labels,
        y=y_labels
    )
    if show:
        fig.show()
    return fig

def plot_visit_counts_heatmap(visit_counts, save_path="image/state_action_visit_counts.png", use_plotly=False, show=False):
    """
    Plots a heatmap of state-action visit counts using either Matplotlib or Plotly.
    """
    if use_plotly:
        plotly_heatmap(
            z=visit_counts,
            xlabel="Action Index",
            ylabel="State Index",
            title="State-Action Visit Counts (Unvisited = 0)",
            show=show,
            colorbar_title="Visit Count"
        )
    else:
        plt.figure(figsize=(12, 8))
        plt.imshow(visit_counts, aspect='auto', cmap='viridis')
        plt.colorbar(label='Visit Count')
        plt.xlabel("Action Index")
        plt.ylabel("State Index")
        plt.title("State-Action Visit Counts (Unvisited = 0)")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
        print(f"State-action visit counts heatmap saved to {save_path}")