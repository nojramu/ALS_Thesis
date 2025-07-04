import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from shewhart_control import initialize_control_chart
from simpsons_rule import quantitative_analysis_simpsons_rule
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# --- Layout Components ---
sidebar = dbc.Nav(
    [
        dbc.NavLink("Preprocessing", href="/", active="exact"),
        dbc.NavLink("Random Forest", href="/rf", active="exact"),
        dbc.NavLink("Kalman Filter", href="/kalman", active="exact"),
        dbc.NavLink("Simpson's Rule", href="/simpson", active="exact"),
        dbc.NavLink("Q-Learning", href="/qlearning", active="exact"),
        dbc.NavLink("Shewhart Control", href="/shewhart", active="exact"),
        dbc.NavLink("System Simulation", href="/sys_simulation", active="exact"),
    ],
    vertical=True,
    pills=True,
    className="bg-light",
    style={"height": "100vh", "padding": "10px"}
)

def make_main_panel():
    return html.Div(id="main-panel", style={"padding": "2rem"})

app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        dcc.Store(id="shewhart-chart-state", data=initialize_control_chart()),
        dbc.Row([
            dbc.Col(sidebar, width=2),
            dbc.Col(make_main_panel(), width=10)
        ])
    ],
    fluid=True
)

# --- Callbacks for Navigation and Content ---
@app.callback(
    Output("main-panel", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    if pathname == "/" or pathname == "/preprocessing":
        return preprocessing_panel()
    elif pathname == "/rf":
        return rf_panel()
    elif pathname == "/kalman":
        return kalman_panel()
    elif pathname == "/simpson":
        return simpson_panel()
    elif pathname == "/qlearning":
        return qlearning_panel()
    elif pathname == "/shewhart":
        return shewhart_panel()
    elif pathname == "/sys_simulation":
        return sys_simulation_panel()
    else:
        return html.H3("404: Page not found")

# --- Panel Definitions ---

def preprocessing_panel():
    return html.Div([
        html.H2("Preprocessing"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select CSV File")]),
            style={
                "width": "100%", "height": "60px", "lineHeight": "60px",
                "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                "textAlign": "center", "margin": "10px"
            },
            multiple=False
        ),
        dbc.Button("Load Sample Data", id="load-default-btn", n_clicks=0, color="primary", className="mb-2"),
        html.Div(id="preprocessing-output"),
    ])

def rf_panel():
    # global rf_models
    sample = {
        'engagement_rate': 0.8, 'time_on_task_s': 450, 'hint_ratio': 0.5, 'interaction_count': 12,
        'task_completed': 1, 'quiz_score': 92, 'difficulty': 3, 'error_rate': 0.2,
        'task_timed_out': 0, 'time_before_hint_used': 120
    }
    feature_notes = {
        'engagement_rate': "(0.0 - 1.0)",
        'time_on_task_s': "(positive integer, seconds)",
        'hint_ratio': "(0.0 - 1.0)",
        'interaction_count': "(positive integer)",
        'task_completed': "(0 or 1)",
        'quiz_score': "(0 - 100)",
        'difficulty': "(positive integer, e.g. 1-10)",
        'error_rate': "(0.0 - 1.0)",
        'task_timed_out': "(0 or 1)",
        'time_before_hint_used': "(positive integer, seconds)"
    }
    feature_inputs = [
        dbc.Row([
            dbc.Col(html.Label(f"{f} {feature_notes[f]}"), width=5),
            dbc.Col(dcc.Input(
                id=f"rf-test-{f}",
                type="number",
                value=sample[f],
                debounce=True,
                style={"width": "100%"}
            ), width=7)
        ], className="mb-2")
        for f in sample
    ]

    # Training parameter inputs
    param_inputs = [
        dbc.Row([
            dbc.Col(html.Label("Test Size:"), width=4),
            dbc.Col(dcc.Input(id="rf-test-size", type="number", value=0.2, min=0.05, max=0.5, step=0.01, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Random State:"), width=4),
            dbc.Col(dcc.Input(id="rf-random-state", type="number", value=25, min=0, step=1, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("N Estimators:"), width=4),
            dbc.Col(dcc.Input(id="rf-n-estimators", type="number", value=100, min=10, step=1, style={"width": "100%"}), width=8)
        ], className="mb-2"),
    ]

    btn_text = "Train Models"
    btn_color = "primary"

    return html.Div([
        html.H2("Random Forest"),
        html.P("Configure training parameters and train the models."),
        html.Div(param_inputs),
        dbc.Button(btn_text, id="train-rf-btn", n_clicks=0, color=btn_color, className="mt-2"),
        html.Div(id="rf-output"),
        html.Hr(),
        html.H4("Cross-Validation Metrics"),
        html.Div([
            html.P(id="cv-mae"),
            html.P(id="cv-mse"),
            html.P(id="cv-f1"),
            html.P(id="cv-precision"),
            html.P(id="cv-recall"),
        ]),
        html.Hr(),
        html.H4("Test Model with Custom Input"),
        html.Div(feature_inputs),
        dbc.Button("Test", id="rf-test-btn", n_clicks=0, color="primary", className="mt-2"),
        html.Div(id="rf-test-output", className="mt-3")
    ])


def kalman_panel():
    import pandas as pd
    from dash.dash_table import DataTable

    # UI for uploading or loading predictions
    upload_section = html.Div([
        dcc.Upload(
            id="upload-predictions",
            children=html.Div(["Drag and Drop or ", html.A("Select CSV File")]),
            style={
                "width": "100%", "height": "60px", "lineHeight": "60px",
                "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                "textAlign": "center", "margin": "10px"
            },
            multiple=False
        ),
        dbc.Button("Load Sample Predictions", id="load-sample-pred-btn", n_clicks=0, color="primary", className="mb-2"),
        html.Div(id="kalman-pred-table")
    ])

    return html.Div([
        html.H2("Kalman Filter"),
        html.P("Add your predictions or load sample predictions to proceed."),
        upload_section,
        html.Br(),
        dbc.Row([
            dbc.Col(html.Label("Process Noise:"), width=4),
            dbc.Col(dcc.Input(id="kalman-process-noise", type="number", value=0.1, min=0, step=0.01, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Measurement Noise:"), width=4),
            dbc.Col(dcc.Input(id="kalman-measurement-noise", type="number", value=1.0, min=0, step=0.01, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Button("Apply Kalman Filter", id="kalman-btn", n_clicks=0, color="primary", className="mt-2"),
        html.Div(id="kalman-output"),
        html.Div(id="kalman-mse-output", style={"marginTop": "10px", "fontWeight": "bold"}),
        html.Hr(),
    ])

def simpson_panel():
    return html.Div([
        html.H2("Simpson's Rule"),
        html.Label("Number of Buckets:"),
        dcc.Slider(
            id="simpson-bucket-num",
            min=3,
            max=10,
            step=1,
            value=5,
            marks={i: str(i) for i in range(3, 11)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        dbc.Button("Integrate Cognitive Load", id="simpson-btn", n_clicks=0, color="primary", className="mt-2"),
        html.Div(id="simpson-output"),
        html.Hr(),
        dbc.Button("Run Quantitative Analysis", id="simpson-quant-analysis-btn", n_clicks=0, color="secondary", className="mt-2"),
        html.Div(id="simpson-quant-analysis-output", style={"whiteSpace": "pre-wrap", "marginTop": "10px"}),
    ])

# Add callback for quantitative analysis button
from dash.exceptions import PreventUpdate
from dash import callback_context

@app.callback(
    Output("simpson-quant-analysis-output", "children"),
    Input("simpson-quant-analysis-btn", "n_clicks"),
    State("url", "pathname"),
)
def run_simpson_quantitative_analysis(n_clicks, pathname):
    if not n_clicks or pathname != "/simpson":
        raise PreventUpdate
    from simpsons_rule import quantitative_analysis_simpsons_rule
    from data_handling import load_csv

    # Load data (same as used in simpsons_rule.py main)
    df = load_csv("data/sample_predictions_with_smoothed.csv")
    if df is None or "smoothed_cognitive_load" not in df.columns:
        return "Data not available or missing 'smoothed_cognitive_load' column."

    y = df["smoothed_cognitive_load"].values
    h = 3  # Step size consistent with simpsons_rule usage

    result = quantitative_analysis_simpsons_rule(y, h)
    if result is None:
        return "Quantitative analysis failed."

    output_text = (
        f"Simpson's Integral: {result['simpson_integral']:.4f}\n"
        f"Trapezoidal Integral: {result['trapezoidal_integral']:.4f}\n"
        f"Absolute Difference: {result['absolute_difference']:.4f}\n"
    )
    if result['relative_difference'] is not None:
        output_text += f"Relative Difference: {result['relative_difference']:.4f}\n"
    else:
        output_text += "Relative Difference: Undefined (division by zero)\n"

    return output_text

# Add callback for anomaly detection performance display
@app.callback(
    Output("shewhart-anomaly-performance", "children"),
    Input("shewhart-sim-interval", "n_intervals"),
    State("shewhart-chart-state", "data"),
    prevent_initial_call=True
)
def update_anomaly_detection_performance(n_intervals, chart_state):
    if not chart_state:
        raise PreventUpdate
    from shewhart_control import evaluate_anomaly_detection_performance

    # For demonstration, simulate true labels and predicted anomalies from chart_state
    # In real use, true labels should come from ground truth data
    true_labels = [0] * (len(chart_state.get('engagement_data', [])) - len(chart_state.get('anomaly_buffer', []))) + \
                  [1 if anomaly else 0 for anomaly in chart_state.get('anomaly_buffer', [])]
    predicted_anomalies = [1 if i in chart_state.get('anomalies', []) else 0 for i in range(len(chart_state.get('engagement_data', [])))]

    try:
        performance = evaluate_anomaly_detection_performance(true_labels, predicted_anomalies)
        sensitivity = performance.get('sensitivity', 0.0)
        specificity = performance.get('specificity', 0.0)
        return f"Sensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}"
    except Exception as e:
        return f"Error computing performance: {e}"

def qlearning_panel():
    # Training parameter inputs
    param_rows = [
        dbc.Row([
            dbc.Col(html.Label("Number of Episodes:"), width=4),
            dbc.Col(dcc.Input(id="qlearn-episodes", type="number", value=200, min=1, step=1, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Max Steps per Episode:"), width=4),
            dbc.Col(dcc.Input(id="qlearn-max-steps", type="number", value=30, min=1, step=1, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Learning Rate (alpha):"), width=4),
            dbc.Col(dcc.Input(id="qlearn-lr", type="number", value=0.1, min=0, max=1, step=0.01, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Discount Factor (gamma):"), width=4),
            dbc.Col(dcc.Input(id="qlearn-gamma", type="number", value=0.9, min=0, max=1, step=0.01, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Epsilon (exploration):"), width=4),
            dbc.Col(dcc.Input(id="qlearn-epsilon", type="number", value=1.0, min=0, max=1, step=0.01, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Epsilon Decay Rate:"), width=4),
            dbc.Col(dcc.Input(id="qlearn-eps-decay", type="number", value=0.005, min=0, max=1, step=0.001, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Min Epsilon:"), width=4),
            dbc.Col(dcc.Input(id="qlearn-min-epsilon", type="number", value=0.05, min=0, max=1, step=0.01, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Number of Buckets (Simpson):"), width=4),
            dbc.Col(dcc.Input(id="qlearn-buckets", type="number", value=5, min=3, max=10, step=1, style={"width": "100%"}), width=8)
        ], className="mb-2"),
    ]
    # Testing input fields
    test_rows = [
        dbc.Row([
            dbc.Col(html.Label("Simpson's Integral Level (1-N):"), width=4),
            dbc.Col(dcc.Input(id="ql-test-simpson", type="number", value=3, min=1, max=10, step=1, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Engagement Level (1-5):"), width=4),
            dbc.Col(dcc.Input(id="ql-test-engagement", type="number", value=3, min=1, max=5, step=1, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Task Completed (0 or 1):"), width=4),
            dbc.Col(dcc.Input(id="ql-test-completed", type="number", value=1, min=0, max=1, step=1, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Previous Task Type (A/B/C/D):"), width=4),
            dbc.Col(dcc.Input(id="ql-test-prevtype", type="text", value='A', maxLength=1, style={"width": "100%"}), width=8)
        ], className="mb-2"),
    ]
    return html.Div([
        html.H2("Q-Learning"),
        html.Div(param_rows),
        dbc.Button("Train Q-Learning Agent", id="qlearn-btn", n_clicks=0, color="primary", className="mt-2"),
        html.Div(id="qlearn-output"),
        html.Hr(),
        html.H4("Q-Table Heatmap"),
        dcc.Loading(dcc.Graph(id="qlearn-heatmap"), type="circle"),
        html.Hr(),
        html.H4("Test Q-Learning Policy"),
        html.Div(test_rows),
        dbc.Button("Get Recommended Action", id="ql-test-btn", n_clicks=0, color="primary", className="mt-2"),
        html.Div(id="ql-test-output"),
    ])

def shewhart_panel():
    # Create blue buttons for engagement rates 0.0 to 1.0
    rate_buttons = [
        dbc.Button(
            f"{v:.1f}",
            id={"type": "shewhart-rate-btn", "index": f"{v:.1f}"},
            color="primary",  # Blue
            outline=False,
            size="sm",
            style={"marginRight": "4px", "marginBottom": "4px"}
        )
        for v in [i / 10 for i in range(11)]
    ]
    return html.Div([
        html.H2("Shewhart Control"),
        dbc.Row([
            dbc.Col(html.Label("Window Size:"), width=4),
            dbc.Col(dcc.Input(id="shewhart-window", type="number", value=20, min=2, max=100, step=1, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Num Std Dev (Threshold):"), width=4),
            dbc.Col(dcc.Input(id="shewhart-num-stddev", type="number", value=3, min=1, max=5, step=0.1, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        html.Label("Sample Engagement Rate:"),
        html.Div(rate_buttons, style={"display": "flex", "flexWrap": "wrap", "marginBottom": "8px"}),
        dbc.Button("Simulate", id="shewhart-sim-btn", n_clicks=0, color="info", className="mt-2", style={"marginRight": "10px"}),
        dcc.Interval(id="shewhart-sim-interval", interval=1000, n_intervals=0, disabled=True),
        dbc.Button("Reset Chart", id="shewhart-reset-btn", n_clicks=0, color="secondary", className="mt-2"),
        html.Div(id="shewhart-notification", className="mt-2"),
        # Feedback input area, initially hidden
        html.Div([
            dbc.Label("Adjust Difficulty:", className="mb-2"),
            dcc.Input(id="shewhart-feedback-input", type="number", min=0, max=10, step=1, placeholder="Enter difficulty (0-10)", className="mb-2", style={"width": "100%"}),
            html.Br(),
            dbc.Button("Submit Feedback", id="shewhart-feedback-submit", n_clicks=0, color="primary", className="mb-3"),
            html.Div(id="shewhart-feedback-response", className="mt-2")
        ], id="shewhart-feedback-container", style={"display": "none"}),
        dcc.Graph(id="shewhart-plotly-fig", style={"height": "400px"}),
        html.Hr(),
        html.H4("Anomaly Detection Performance"),
        html.Div(id="shewhart-anomaly-performance", style={"whiteSpace": "pre-wrap", "marginTop": "10px"}),
    ])

def sys_simulation_panel():
    # Define input fields, labels, and info
    param_fields = [
        ('engagement_rate', "Engagement Rate", "Value between 0.0 and 1.0", 0.8),
        ('time_on_task_s', "Time on Task (s)", "Positive integer (seconds)", 450),
        ('hint_ratio', "Hint Ratio", "Value between 0.0 and 1.0", 0.5),
        ('interaction_count', "Interaction Count", "Positive integer", 12),
        ('task_completed', "Task Completed", "0 or 1", 1),
        ('quiz_score', "Quiz Score", "Integer between 0 and 100", 92),
        ('difficulty', "Difficulty", "Positive integer (e.g. 1-10)", 3),
        ('error_rate', "Error Rate", "Value between 0.0 and 1.0", 0.2),
        ('task_timed_out', "Task Timed Out", "0 or 1", 0),
        ('time_before_hint_used', "Time Before Hint Used", "Positive integer (seconds)", 120),
        ('prev_type', "Previous Task Type (A/B/C/D)", "Single letter: A, B, C, or D", 'A'),
        ('num_buckets', "Number of buckets", "Number of bucket use for Simpson's Rule", 5),
    ]
    default_values = {field: default for field, _, _, default in param_fields}
    # Arrange inputs in two columns with info
    left_col = []
    right_col = []
    for i, (field, label, info, default_val) in enumerate(param_fields):
        input_type = "number" if field != "prev_type" else "text"
        input_box = dbc.Row([
            dbc.Col([
                html.Label(label),
                html.Br(),
                html.Small(info, style={"color": "#888", "fontSize": "80%"})
            ], width=6),
            dbc.Col(dcc.Input(id=f"sys-sim-{field}", type=input_type, value=default_val, style={"width": "100%"}), width=6)
        ], className="mb-2")
        if i % 2 == 0:
            left_col.append(input_box)
        else:
            right_col.append(input_box)

    return html.Div([
        html.H2("System Simulation"),
        html.P("Enter simulation parameters below. Please ensure all values are within the specified ranges."),
        dcc.Store(id="sys-sim-params", data=default_values),
        dbc.Button("Initialize", id="sys-sim-init-btn", n_clicks=0, color="primary", className="mb-2"),
        dbc.Button("Append", id="sys-sim-append-btn", n_clicks=0, color="success", className="mb-2", style={"marginLeft": "10px"}),
        html.Div(id="sys-simulation-output", className="mt-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="sys-simulation-graph", figure=go.Figure()), width=6),
            dbc.Col(dcc.Graph(id="sys-engagement-graph", figure=go.Figure()), width=6)
        ]),
        dbc.Row([
            dbc.Col(left_col, width=6),
            dbc.Col(right_col, width=6)
    ]),
    ])
# Move this import here, after all app/layout definitions
import callback  # Register all callbacks

if __name__ == "__main__":
    app.run(debug=True)