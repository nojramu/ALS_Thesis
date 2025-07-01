import dash
from dash import dcc, html, Input, Output, State, dash_table
from dash.dependencies import ALL
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from data_handling import load_csv, preprocess_data
from random_forest import train_models, predict
from kalman_filter import add_kalman_column
from simpsons_rule import simpsons_rule, discretize_simpsons_result
from ql_setup import define_state_space, define_action_space, initialize_q_table
from ql_training import train_q_learning_agent
from ql_analysis import get_optimal_action_for_state, get_top_n_actions_for_state, plotly_qtable_heatmap
from shewhart_control import (
    initialize_control_chart, add_engagement_data, check_for_engagement_anomaly,
    get_control_chart_data, enough_anomalies
)
from plot_utils import plot_line_chart, plotly_bar_chart

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# --- Globals for session state ---
DATA_PATH = "data/sample_training_data.csv"
df_global = None
rf_models = None
features = None
metrics = None
q_table = None
chart_state = initialize_control_chart()
simpsons_integral = None
simpsons_bucket = None

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

@app.callback(
    Output("preprocessing-output", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    Input("load-default-btn", "n_clicks"),
    prevent_initial_call=True
)
def handle_preprocessing(uploaded_contents, uploaded_filename, _):
    global df_global
    import base64, io

    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    # Columns to check for missing values
    feature_cols = [
        'engagement_rate', 'time_on_task_s', 'hint_ratio', 'interaction_count',
        'task_completed', 'quiz_score', 'difficulty', 'error_rate',
        'task_timed_out', 'time_before_hint_used'
    ]
    target_cols = ['cognitive_load', 'engagement_level']

    if trigger == "upload-data" and uploaded_contents:
        _, content_string = uploaded_contents.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        df_global = df
        info = []
        missing = df[feature_cols + target_cols].isnull().sum()
        total_missing = missing.sum()
        if total_missing > 0:
            info.append(html.P(f"Missing values detected: {dict(missing[missing > 0])}"))
        else:
            info.append(html.P("No missing values detected."))
        info.append(html.P(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns."))
        df_display = df.head(10)
        df_display.columns = df_display.columns.map(str)
        return html.Div([
            html.H5(f"Uploaded: {uploaded_filename}"),
            *info,
            dash_table.DataTable(
                data=[{str(k): v for k, v in row.items()} for row in df_display.to_dict("records")],
                columns=[{"name": str(i), "id": str(i)} for i in df_display.columns],
                page_size=10,
                style_table={
                    'overflowX': 'auto',
                    'overflowY': 'auto',
                    'maxHeight': '400px',
                    'minWidth': '100%',
                    'maxWidth': '100%',
                    'margin': '0 auto',
                    'resize': 'both'
                },
                style_cell={
                    'minWidth': '120px', 'width': '120px', 'maxWidth': '180px',
                    'whiteSpace': 'normal',
                    'textAlign': 'left'
                }
            )
        ])
    elif trigger == "load-default-btn":
        df = load_csv(DATA_PATH)
        df_global = df
        if df is not None:
            info = []
            missing = df[feature_cols + target_cols].isnull().sum()
            total_missing = missing.sum()
            if total_missing > 0:
                info.append(html.P(f"Missing values detected: {dict(missing[missing > 0])}"))
            else:
                info.append(html.P("No missing values detected."))
            info.append(html.P(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns."))
            df_display = df.head(10)
            df_display.columns = df_display.columns.map(str)
            return html.Div([
                html.H5("Loaded sample training data"),
                *info,
                dash_table.DataTable(
                    data=[{str(k): v for k, v in row.items()} for row in df_display.to_dict("records")],
                    columns=[{"name": str(i), "id": str(i)} for i in df_display.columns],
                    page_size=10,
                    style_table={
                        'overflowX': 'auto',
                        'overflowY': 'auto',
                        'maxHeight': '400px',
                        'minWidth': '100%',
                        'maxWidth': '100%',
                        'margin': '0 auto',
                        'resize': 'both'
                    },
                    style_cell={
                        'minWidth': '120px', 'width': '120px', 'maxWidth': '180px',
                        'whiteSpace': 'normal',
                        'textAlign': 'left'
                    }
                )
            ])
        else:
            return html.Div("Failed to load data.")
    return ""

def rf_panel():
    # Sample values for the input fields
    sample = {
        'engagement_rate': 0.8,         # 0.0 - 1.0
        'time_on_task_s': 450,          # positive integer (seconds)
        'hint_ratio': 0.5,              # 0.0 - 1.0
        'interaction_count': 12,        # positive integer
        'task_completed': 1,            # 0 or 1
        'quiz_score': 92,               # 0 - 100
        'difficulty': 3,                # positive integer (e.g., 1-10)
        'error_rate': 0.2,              # 0.0 - 1.0
        'task_timed_out': 0,            # 0 or 1
        'time_before_hint_used': 120    # positive integer (seconds)
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
    return html.Div([
        html.H2("Random Forest"),
        html.P("Click 'Train Models' after loading and preprocessing data."),
        dbc.Button("Train Models", id="train-rf-btn", n_clicks=0, color="primary", className="mt-2"),
        dbc.Button("Retrain Models", id="retrain-rf-btn", n_clicks=0, color="warning", className="mt-2", style={"marginLeft": "10px"}),
        html.Div(id="rf-output"),
        html.Hr(),
        html.H4("Test Model with Custom Input"),
        html.Div(feature_inputs),
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
        dbc.Button("Test", id="rf-test-btn", n_clicks=0, color="primary", className="mt-2"),
        html.Div(id="rf-test-output", className="mt-3")
    ])

@app.callback(
    Output("rf-output", "children"),
    [Input("train-rf-btn", "n_clicks"),
     Input("retrain-rf-btn", "n_clicks")],
    State("rf-test-size", "value"),
    State("rf-random-state", "value"),
    State("rf-n-estimators", "value"),
    prevent_initial_call=True
)
def handle_rf_train(train_clicks, retrain_clicks, test_size, random_state, n_estimators):
    global df_global, rf_models, features, metrics
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if df_global is None or df_global.empty:
        return html.Div("Please load and preprocess data first.")

    feature_cols = [
        'engagement_rate', 'time_on_task_s', 'hint_ratio', 'interaction_count',
        'task_completed', 'quiz_score', 'difficulty', 'error_rate',
        'task_timed_out', 'time_before_hint_used'
    ]
    target_cols = ['cognitive_load', 'engagement_level']

    df = preprocess_data(df_global, feature_cols, is_training_data=True)
    if df is None or df.empty:
        return html.Div("Data preprocessing failed. Please check your data.")

    reg, clf, features, metrics = train_models(df, feature_cols, target_cols,
                                               test_size=test_size,
                                               random_state=random_state,
                                               n_estimators=n_estimators)
    rf_models = (reg, clf)

    reg_importance = reg.feature_importances_
    reg_fig = plotly_bar_chart(
        x=feature_cols,
        y=reg_importance,
        xlabel="Feature",
        ylabel="Importance",
        title="Random Forest Feature Importance (Regression)",
        color='indigo'
    )
    clf_importance = clf.feature_importances_
    clf_fig = plotly_bar_chart(
        x=feature_cols,
        y=clf_importance,
        xlabel="Feature",
        ylabel="Importance",
        title="Random Forest Feature Importance (Classification)",
        color='green'
    )

    retrain_msg = ""
    if trigger == "retrain-rf-btn":
        retrain_msg = html.P("Models retrained due to engagement anomalies.", style={"color": "orange", "fontWeight": "bold"})

    return html.Div([
        retrain_msg,
        html.P(f"Regression MSE: {metrics['mse']:.4f}"),
        html.P(f"Classification Accuracy: {metrics['accuracy']:.4f}"),
        html.P("Models trained and stored in memory."),
        html.H5("Regression Feature Importance"),
        dcc.Graph(figure=reg_fig),
        html.H5("Classification Feature Importance"),
        dcc.Graph(figure=clf_fig)
    ])

@app.callback(
    Output("rf-test-output", "children"),
    Input("rf-test-btn", "n_clicks"),
    [State(f"rf-test-{f}", "value") for f in [
        'engagement_rate', 'time_on_task_s', 'hint_ratio', 'interaction_count',
        'task_completed', 'quiz_score', 'difficulty', 'error_rate',
        'task_timed_out', 'time_before_hint_used'
    ]],
    prevent_initial_call=True
)
def handle_rf_test(n_clicks, *values):
    global rf_models, features
    feature_cols = [
        'engagement_rate', 'time_on_task_s', 'hint_ratio', 'interaction_count',
        'task_completed', 'quiz_score', 'difficulty', 'error_rate',
        'task_timed_out', 'time_before_hint_used'
    ]
    if rf_models is None:
        return html.Div("Please train the models first.")
    # Prepare input DataFrame
    import pandas as pd
    input_dict = dict(zip(feature_cols, values))
    new_data = pd.DataFrame([input_dict])
    try:
        reg_pred, clf_pred = predict(rf_models, feature_cols, new_data)
        return html.Div([
            html.P(f"Predicted Cognitive Load: {reg_pred[0]:.2f}"),
            html.P(f"Predicted Engagement Level: {int(clf_pred[0])}")
        ])
    except Exception as e:
        return html.Div(f"Prediction failed: {e}")
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
        dbc.Button("Apply Kalman Filter", id="kalman-btn", n_clicks=0, color="primary", className="mt-2"),
        html.Div(id="kalman-output"),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.Label("Process Noise:"), width=4),
            dbc.Col(dcc.Input(id="kalman-process-noise", type="number", value=0.1, min=0, step=0.01, style={"width": "100%"}), width=8)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Label("Measurement Noise:"), width=4),
            dbc.Col(dcc.Input(id="kalman-measurement-noise", type="number", value=1.0, min=0, step=0.01, style={"width": "100%"}), width=8)
        ], className="mb-2"),
    ])

@app.callback(
    Output("kalman-pred-table", "children"),
    Input("upload-predictions", "contents"),
    State("upload-predictions", "filename"),
    Input("load-sample-pred-btn", "n_clicks"),
    prevent_initial_call=True
)
def handle_kalman_predictions(uploaded_contents, uploaded_filename, n_clicks_sample):
    import pandas as pd
    from dash.dash_table import DataTable
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "upload-predictions" and uploaded_contents:
        _, content_string = uploaded_contents.split(",")
        import base64, io
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        table = DataTable(
            data=[{str(k): v for k, v in row.items()} for row in df.head(10).to_dict("records")],
            columns=[{"name": i, "id": i} for i in df.columns],
            page_size=10,
            style_table={
                'overflowX': 'auto',
                'overflowY': 'auto',
                'maxHeight': '400px',
                'minWidth': '100%',
                'maxWidth': '100%',
                'margin': '0 auto',
                'resize': 'both'
            },
            style_cell={
                'minWidth': '120px', 'width': '120px', 'maxWidth': '180px',
                'whiteSpace': 'normal',
                'textAlign': 'left'
            }
        )
        return html.Div([
            html.H5(f"Uploaded: {uploaded_filename}"),
            table
        ])
    elif trigger == "load-sample-pred-btn":
        sample_pred_path = "data/sample_predictions.csv"
        try:
            df = pd.read_csv(sample_pred_path)
        except Exception as e:
            return html.Div([html.P(f"Failed to load sample predictions: {e}")])
        table = DataTable(
            data=[{str(k): v for k, v in row.items()} for row in df.head(10).to_dict("records")],
            columns=[{"name": i, "id": i} for i in df.columns],
            page_size=10,
            style_table={
                'overflowX': 'auto',
                'overflowY': 'auto',
                'maxHeight': '400px',
                'minWidth': '100%',
                'maxWidth': '100%',
                'margin': '0 auto',
                'resize': 'both'
            },
            style_cell={
                'minWidth': '120px', 'width': '120px', 'maxWidth': '180px',
                'whiteSpace': 'normal',
                'textAlign': 'left'
            }
        )
        return html.Div([
            html.H5("Loaded sample predictions"),
            table
        ])
    return ""

@app.callback(
    Output("kalman-output", "children"),
    Input("kalman-btn", "n_clicks"),
    State("kalman-pred-table", "children"),
    State("kalman-process-noise", "value"),
    State("kalman-measurement-noise", "value"),
    prevent_initial_call=True
)
def handle_kalman(n_clicks, pred_table_children, process_noise, measurement_noise):
    global df_global
    import pandas as pd
    from plotly.graph_objs import Figure
    from kalman_filter import add_kalman_column

    if not pred_table_children:
        return html.Div("Please upload or load predictions first.")

    sample_pred_path = "data/sample_predictions.csv"
    try:
        df = pd.read_csv(sample_pred_path)
    except Exception as e:
        return html.Div([f"Failed to load predictions: {e}"])

    try:
        df = add_kalman_column(df, col="cognitive_load", new_col="smoothed_cognitive_load",
                               process_noise=process_noise, measurement_noise=measurement_noise)
        df_global = df  # <-- Save to global so Simpson panel can access
    except Exception as e:
        return html.Div([f"Kalman filter failed: {e}"])

    import plotly.graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["cognitive_load"],
        mode="lines+markers", name="Original Cognitive Load"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["smoothed_cognitive_load"],
        mode="lines+markers", name="Kalman Smoothed"
    ))
    fig.update_layout(
        title="Kalman Filtered Cognitive Load",
        xaxis_title="Index",
        yaxis_title="Cognitive Load"
    )

    return dcc.Graph(figure=fig)

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
    ])

@app.callback(
    Output("simpson-output", "children"),
    Input("simpson-btn", "n_clicks"),
    State("simpson-bucket-num", "value"),
)
def handle_simpson(_, bucket_num):
    global df_global, simpsons_integral, simpsons_bucket
    if df_global is None or "smoothed_cognitive_load" not in df_global.columns:
        return html.Div("Apply Kalman filter first.")
    h = 3
    simpsons_integral = simpsons_rule(df_global["smoothed_cognitive_load"], h)
    simpsons_bucket = discretize_simpsons_result(simpsons_integral, num_buckets=bucket_num or 5)
    return html.Div([
        html.P(f"Simpson's Integral: {simpsons_integral:.2f}"),
        html.P(f"Discretized Bucket (num_buckets={bucket_num or 5}): {simpsons_bucket}")
    ])

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
        html.H4("Test Q-Learning Policy"),
        html.Div(test_rows),
        dbc.Button("Get Recommended Action", id="ql-test-btn", n_clicks=0, color="primary", className="mt-2"),
        html.Div(id="ql-test-output"),
        html.Hr(),
        html.H4("Q-Table Heatmap"),
        dcc.Loading(dcc.Graph(id="qlearn-heatmap"), type="circle"),
    ])

@app.callback(
    [Output("qlearn-output", "children"),
     Output("qlearn-heatmap", "figure")],
    Input("qlearn-btn", "n_clicks"),
    State("qlearn-episodes", "value"),
    State("qlearn-max-steps", "value"),
    State("qlearn-lr", "value"),
    State("qlearn-gamma", "value"),
    State("qlearn-epsilon", "value"),
    State("qlearn-eps-decay", "value"),
    State("qlearn-min-epsilon", "value"),
    State("qlearn-buckets", "value"),
    prevent_initial_call=True
)
def handle_qlearning(n_clicks, episodes, max_steps, lr, gamma, epsilon, eps_decay, min_epsilon, num_buckets):
    global q_table, state_to_index, index_to_action, action_to_index, index_to_state
    states, state_to_index, index_to_state, num_states = define_state_space(num_buckets)
    actions, action_to_index, index_to_action, num_actions = define_action_space()
    q_table = initialize_q_table(num_states, num_actions)

    # Get logs from training
    q_table, rewards, max_q_values, policy_evolution, logs = train_q_learning_agent(
        num_episodes=episodes,
        max_steps_per_episode=max_steps,
        learning_rate=lr,
        discount_factor=gamma,
        epsilon=epsilon,
        epsilon_decay_rate=eps_decay,
        min_epsilon=min_epsilon,
        reward_mode="state",
        progress_interval=20,
        num_buckets=num_buckets
    )
    # Plotly heatmap
    heatmap_fig = plotly_qtable_heatmap(q_table, index_to_state=index_to_state, index_to_action=index_to_action, show=False)
    return (
        html.Div([
            html.P("Q-Learning training complete."),
            html.P(f"Final Q-table shape: {q_table.shape}"),
            html.Pre("\n".join(logs), style={"maxHeight": "300px", "overflowY": "auto", "background": "#f8f8f8"})
        ]),
        heatmap_fig
    )

@app.callback(
    Output("ql-test-output", "children"),
    Input("ql-test-btn", "n_clicks"),
    State("ql-test-simpson", "value"),
    State("ql-test-engagement", "value"),
    State("ql-test-completed", "value"),
    State("ql-test-prevtype", "value"),
    prevent_initial_call=True
)
def handle_ql_test(n_clicks, simpson_level, engagement_level, completed, prev_type):
    global q_table, state_to_index, index_to_action, action_to_index
    if q_table is None:
        return html.Div("Please train the Q-learning agent first.")
    try:
        state = (int(simpson_level), int(engagement_level), int(completed), str(prev_type).upper())
    except Exception:
        return html.Div("Invalid input. Please check your values.")
    action = get_optimal_action_for_state(state, q_table, state_to_index, index_to_action, action_to_index)
    if action is None:
        return html.Div("No valid action found for this state.")

    # Show top 5 actions
    top_actions = get_top_n_actions_for_state(state, q_table, state_to_index, index_to_action, action_to_index, n=5)
    top_actions_html = [
        html.Li(f"{i+1}. Task: {a[0][0]}, Difficulty: {a[0][1]}, Q-value: {a[1]:.4f}")
        for i, a in enumerate(top_actions)
    ]

    return html.Div([
        html.P(f"Recommended Next Task: {action[0]}"),
        html.P(f"Recommended Difficulty: {action[1]}"),
        html.Hr(),
        html.H5("Top 5 Actions:"),
        html.Ul(top_actions_html)
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
        dcc.Graph(id="shewhart-plotly-fig", style={"height": "400px"}),
    ])

@app.callback(
    Output("shewhart-sim-interval", "disabled"),
    Output("shewhart-sim-btn", "children"),
    Input("shewhart-sim-btn", "n_clicks"),
    State("shewhart-sim-interval", "disabled"),
    prevent_initial_call=True
)
def toggle_simulation(n_clicks, disabled):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    running = not disabled
    return running, ("Stop" if not disabled else "Simulate")

@app.callback(
    [Output("shewhart-plotly-fig", "figure"),
     Output("shewhart-notification", "children"),
     Output("shewhart-chart-state", "data")],
    [Input({"type": "shewhart-rate-btn", "index": ALL}, "n_clicks"),
     Input("shewhart-reset-btn", "n_clicks"),
     Input("shewhart-sim-interval", "n_intervals")],
    State("shewhart-window", "value"),
    State("shewhart-num-stddev", "value"),
    State("shewhart-chart-state", "data"),
    State("shewhart-sim-interval", "disabled"),
    prevent_initial_call=True
)
def update_shewhart_chart(rate_btn_clicks, reset_clicks, sim_intervals, window_size, num_stddev, chart_state, sim_disabled):
    import dash
    import json
    import re
    import random

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    triggered = ctx.triggered[0]["prop_id"]

    # Handle reset: add two median values for control limits
    if "shewhart-reset-btn" in triggered:
        chart_state = initialize_control_chart(window_size=window_size)
        add_engagement_data(chart_state, 0.5, num_stddev=num_stddev)
        add_engagement_data(chart_state, 0.5, num_stddev=num_stddev)
        chart_data = get_control_chart_data(chart_state)
        engagement_rates = chart_data.get("engagement_rates", [])
        cl = chart_data.get("cl")
        ucl = chart_data.get("ucl")
        lcl = chart_data.get("lcl")
        anomalies = chart_data.get("anomalies", [])
        x = list(range(len(engagement_rates)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=engagement_rates, mode='lines+markers', name='Engagement Rate'))
        fig.add_trace(go.Scatter(x=x, y=[cl] * len(x), mode='lines', name='CL'))
        fig.add_trace(go.Scatter(x=x, y=[ucl] * len(x), mode='lines', name='UCL'))
        fig.add_trace(go.Scatter(x=x, y=[lcl] * len(x), mode='lines', name='LCL'))
        notification = "Chart has been reset. Two median values (0.5) added."
        return fig, notification, chart_state

    # Handle simulation interval
    if "shewhart-sim-interval" in triggered and not sim_disabled:
        engagement_rate = round(random.uniform(0, 1), 2)
    # Handle engagement rate button click
    elif "shewhart-rate-btn" in triggered:
        match = re.match(r'(\{.*\})\.n_clicks', triggered)
        if match:
            btn_id = json.loads(match.group(1))
            engagement_rate = float(btn_id["index"])
        else:
            raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate

    if not chart_state or 'window_size' not in chart_state:
        chart_state = initialize_control_chart(window_size=window_size)
    else:
        chart_state['window_size'] = window_size

    add_engagement_data(chart_state, engagement_rate, num_stddev=num_stddev)
    chart_data = get_control_chart_data(chart_state)
    engagement_rates = chart_data.get("engagement_rates", [])
    cl = chart_data.get("cl")
    ucl = chart_data.get("ucl")
    lcl = chart_data.get("lcl")
    anomalies = chart_data.get("anomalies", [])

    if not engagement_rates or cl is None or ucl is None or lcl is None:
        fig = go.Figure()
        notification = "Not enough data to plot control chart."
        return fig, notification, chart_state

    x = list(range(len(engagement_rates)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=engagement_rates, mode='lines+markers', name='Engagement Rate'))
    fig.add_trace(go.Scatter(x=x, y=[cl] * len(x), mode='lines', name='CL'))
    fig.add_trace(go.Scatter(x=x, y=[ucl] * len(x), mode='lines', name='UCL'))
    fig.add_trace(go.Scatter(x=x, y=[lcl] * len(x), mode='lines', name='LCL'))
    if anomalies:
        anomaly_x = [i for i in anomalies]
        anomaly_y = [engagement_rates[i] for i in anomaly_x]
        fig.add_trace(go.Scatter(x=anomaly_x, y=anomaly_y, mode='markers', marker=dict(color='red', size=12, symbol='x'), name='Anomaly'))

    notification = ""
    # Natural language explanation for anomaly
    if anomalies and anomalies[-1] == len(engagement_rates) - 1:
        latest = engagement_rates[-1]
        explanation = f"Latest engagement rate ({latest:.2f}) is {'above' if latest > ucl else 'below'} the control limits (CL={cl:.2f}, UCL={ucl:.2f}, LCL={lcl:.2f})."
        notification = html.Div([
            "⚠️ Anomaly detected in the latest engagement rate!",
            html.Br(),
            explanation
        ], style={"color": "red", "fontWeight": "bold"})

    # Check for enough anomalies to suggest retraining/adaptation
    if enough_anomalies(chart_state, threshold=2):
        retrain_msg = html.Div(
            "⚠️ Multiple anomalies detected! Please retrain the Random Forest models.",
            style={"color": "orange", "fontWeight": "bold", "marginTop": "10px"}
        )
        if notification:
            notification = html.Div([notification, html.Br(), retrain_msg])
        else:
            notification = retrain_msg

    return fig, notification, chart_state

def sys_simulation_panel():
    # Define input fields and their labels
    param_fields = [
        ('engagement_rate', "Engagement Rate"),
        ('time_on_task_s', "Time on Task (s)"),
        ('hint_ratio', "Hint Ratio"),
        ('interaction_count', "Interaction Count"),
        ('task_completed', "Task Completed"),
        ('quiz_score', "Quiz Score"),
        ('difficulty', "Difficulty"),
        ('error_rate', "Error Rate"),
        ('task_timed_out', "Task Timed Out"),
        ('time_before_hint_used', "Time Before Hint Used"),
        ('prev_type', "Previous Task Type (A/B/C/D)")
    ]

    # Arrange inputs in two columns
    left_col = []
    right_col = []
    for i, (field, label) in enumerate(param_fields):
        input_type = "number" if field != "prev_type" else "text"
        default_val = 0 if input_type == "number" else "A"
        input_box = dbc.Row([
            dbc.Col(html.Label(label), width=6),
            dbc.Col(dcc.Input(id=f"sys-sim-{field}", type=input_type, value=default_val, style={"width": "100%"}), width=6)
        ], className="mb-2")
        if i % 2 == 0:
            left_col.append(input_box)
        else:
            right_col.append(input_box)

    return html.Div([
        html.H2("System Simulation"),
        html.P("This panel allows you to run a full pipeline simulation."),
        dbc.Button("Initialize Simulation", id="sys-sim-init-btn", n_clicks=0, color="primary", className="mb-2"),
        dbc.Row([
            dbc.Col(left_col, width=6),
            dbc.Col(right_col, width=6)
        ]),
        html.Div(id="sys-simulation-output", className="mt-3")
    ])

@app.callback(
    Output("sys-simulation-output", "children"),
    Input("sys-sim-init-btn", "n_clicks"),
    prevent_initial_call=True
)
def handle_sys_sim_init(n_clicks):
    global rf_models, q_table
    missing = []
    if rf_models is None:
        missing.append("Random Forest models")
    if q_table is None:
        missing.append("Q-Learning agent")
    if missing:
        return html.Div(
            f"Please train the following before initializing the simulation: {', '.join(missing)}.",
            style={"color": "red", "fontWeight": "bold"}
        )
    # If both models are trained, proceed with simulation initialization
    return html.Div(
        "Simulation initialized! (You can now proceed with your simulation steps.)",
        style={"color": "green", "fontWeight": "bold"}
    )


if __name__ == "__main__":
    app.run(debug=True)