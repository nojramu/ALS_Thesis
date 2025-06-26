import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import random
import plotly.graph_objs as go

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Input(id='seconds-input', type='number', value=10, min=1, step=1, placeholder='Seconds'),  # Input for number of seconds
    html.Button('Start', id='start-btn', n_clicks=0),  # Button to start the simulation
    dcc.Graph(id='live-graph'),  # Graph to display the progress
    dcc.Interval(id='interval', interval=300, n_intervals=0, disabled=True),  # Interval for updating the graph every second
    html.Div(id='hidden-div', style={'display': 'none'})  # Hidden div (not used, placeholder)
])

# Store progress data globally
progress_data = []

@app.callback(
    Output('live-graph', 'figure'),
    Output('interval', 'disabled'),
    Output('interval', 'n_intervals'),
    Input('start-btn', 'n_clicks'),
    Input('interval', 'n_intervals'),
    State('seconds-input', 'value'),
    prevent_initial_call=True
)
def update_graph(n_clicks, n_intervals, seconds):
    """
    Callback to update the graph and control the interval component.
    - Resets data and starts interval when 'Start' is clicked.
    - Appends random data each second while interval is running.
    - Keeps only the last 10 values in the plot.
    """
    global progress_data
    ctx = dash.callback_context

    # If triggered by the start button, reset everything
    if ctx.triggered and ctx.triggered[0]['prop_id'].startswith('start-btn'):
        progress_data = []
        fig = go.Figure(
            data=[go.Scatter(y=progress_data, mode='lines+markers')],
            layout=go.Layout(
                xaxis=dict(title='Seconds'),
                yaxis=dict(title='Random Value', range=[0, 1]),
                title='Random Progress per Second'
            )
        )
        return fig, False, 0  # Enable interval, reset n_intervals

    # If triggered by the interval, update the data
    if n_intervals is not None and seconds:
        if n_intervals == 0:
            progress_data = []
        if n_intervals <= seconds:
            progress_data.append(random.uniform(0, 1))  # Append a new random value
            # Keep only the last 10 values
            if len(progress_data) > 10:
                progress_data = progress_data[-10:]
        disabled = n_intervals >= seconds  # Disable interval after reaching the limit
        fig = go.Figure(
            data=[go.Scatter(y=progress_data, mode='lines+markers')],
            layout=go.Layout(
                xaxis=dict(title='Seconds'),
                yaxis=dict(title='Random Value', range=[0, 1]),
                title='Random Progress per Second'
            )
        )
        return fig, disabled, dash.no_update

    # Default: do not update anything
    return dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run(debug=True)