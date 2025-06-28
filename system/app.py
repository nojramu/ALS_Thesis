import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import os

from data_handling import load_csv, preprocess_data
from random_forest import train_models
# from kalman_filter import add_kalman_column
# from simpsons_rule import simpsons_rule, discretize_simpsons_result
# # from ql_setup import define_state_space, define_action_space, initialize_q_table
# from ql_training import train_q_learning_agent
# from shewhart_control import initialize_control_chart, add_engagement_data, check_for_engagement_anomaly
from plot_utils import plot_line_chart

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
# chart_state = initialize_control_chart()
simpsons_integral = None
simpsons_bucket = None

# --- Layout Components ---
sidebar = dbc.Nav(
    [
        dbc.NavLink("Preprocessing", href="/", active="exact"),
        dbc.NavLink("Random Forest", href="/rf", active="exact"),
        # dbc.NavLink("Kalman Filter", href="/kalman", active="exact"),
        # dbc.NavLink("Simpson's Rule", href="/simpson", active="exact"),
        # dbc.NavLink("Q-Learning", href="/qlearning", active="exact"),
        # dbc.NavLink("Shewhart Control", href="/shewhart", active="exact"),
        # dbc.NavLink("Visualization", href="/viz", active="exact"),
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
    # elif pathname == "/kalman":
    #     return kalman_panel()
    # elif pathname == "/simpson":
    #     return simpson_panel()
    # elif pathname == "/qlearning":
    #     return qlearning_panel()
    # elif pathname == "/shewhart":
    #     return shewhart_panel()
    # elif pathname == "/viz":
    #     return viz_panel()
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
        html.Button("Load Sample Data", id="load-default-btn", n_clicks=0),
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
    return html.Div([
        html.H2("Random Forest"),
        html.P("Click 'Train Models' after loading and preprocessing data."),
        html.Button("Train Models", id="train-rf-btn", n_clicks=0),
        html.Div(id="rf-output"),
    ])

@app.callback(
    Output("rf-output", "children"),
    Input("train-rf-btn", "n_clicks"),
    prevent_initial_call=True
)
def handle_rf_train(n_clicks):
    global df_global, rf_models, features, metrics
    if not n_clicks:
        return ""  # Don't show anything until button is clicked

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

    reg, clf, features, metrics = train_models(df, feature_cols, target_cols)
    rf_models = (reg, clf)

    # Plot feature importance for regression model
    reg_importance = reg.feature_importances_
    reg_fig = go.Figure([go.Bar(
        x=feature_cols,
        y=reg_importance,
        marker_color='indigo'
    )])
    reg_fig.update_layout(
        title="Random Forest Feature Importance (Regression)",
        xaxis_title="Feature",
        yaxis_title="Importance",
        yaxis=dict(range=[0, max(reg_importance)*1.1])
    )

    # Plot feature importance for classification model
    clf_importance = clf.feature_importances_
    clf_fig = go.Figure([go.Bar(
        x=feature_cols,
        y=clf_importance,
        marker_color='darkgreen'
    )])
    clf_fig.update_layout(
        title="Random Forest Feature Importance (Classification)",
        xaxis_title="Feature",
        yaxis_title="Importance",
        yaxis=dict(range=[0, max(clf_importance)*1.1])
    )

    return html.Div([
        html.P(f"Regression MSE: {metrics['mse']:.4f}"),
        html.P(f"Classification Accuracy: {metrics['accuracy']:.4f}"),
        html.P("Models trained and stored in memory."),
        html.H5("Regression Feature Importance"),
        dcc.Graph(figure=reg_fig),
        html.H5("Classification Feature Importance"),
        dcc.Graph(figure=clf_fig)
    ])

# def kalman_panel():
#     return html.Div([
#         html.H2("Kalman Filter"),
#         html.Button("Apply Kalman Filter", id="kalman-btn", n_clicks=0),
#         html.Div(id="kalman-output"),
#     ])

# @app.callback(
#     Output("kalman-output", "children"),
#     Input("kalman-btn", "n_clicks"),
# )
# def handle_kalman(_):
#     global df_global
#     if df_global is None or "predicted_cognitive_load" not in df_global.columns:
#         return html.Div("Run Random Forest prediction first.")
#     df = add_kalman_column(df_global, col="predicted_cognitive_load", new_col="smoothed_cognitive_load")
#     df_global["smoothed_cognitive_load"] = df["smoothed_cognitive_load"]
#     fig = px.line(df, y=["predicted_cognitive_load", "smoothed_cognitive_load"], title="Kalman Smoothing")
#     return dcc.Graph(figure=fig)

# def simpson_panel():
#     return html.Div([
#         html.H2("Simpson's Rule"),
#         html.Button("Integrate Cognitive Load", id="simpson-btn", n_clicks=0),
#         html.Div(id="simpson-output"),
#     ])

# @app.callback(
#     Output("simpson-output", "children"),
#     Input("simpson-btn", "n_clicks"),
# )
# def handle_simpson(_):
#     global df_global, simpsons_integral, simpsons_bucket
#     if df_global is None or "smoothed_cognitive_load" not in df_global.columns:
#         return html.Div("Apply Kalman filter first.")
#     h = 3
#     simpsons_integral = simpsons_rule(df_global["smoothed_cognitive_load"], h)
#     simpsons_bucket = discretize_simpsons_result(simpsons_integral)
#     return html.Div([
#         html.P(f"Simpson's Integral: {simpsons_integral:.2f}"),
#         html.P(f"Discretized Bucket: {simpsons_bucket}")
#     ])

# def qlearning_panel():
#     return html.Div([
#         html.H2("Q-Learning"),
#         html.Button("Train Q-Learning Agent", id="qlearn-btn", n_clicks=0),
#         html.Div(id="qlearn-output"),
#     ])

# @app.callback(
#     Output("qlearn-output", "children"),
#     Input("qlearn-btn", "n_clicks"),
# )
# def handle_qlearning(_):
#     global q_table
#     q_table, _, *_ = train_q_learning_agent(num_episodes=100, max_steps_per_episode=20)
#     return html.Div([
#         html.P("Q-Learning training complete."),
#         html.P(f"Final Q-table shape: {q_table.shape}")
#     ])

# def shewhart_panel():
#     return html.Div([
#         html.H2("Shewhart Control"),
#         html.Button("Add Engagement Data", id="shewhart-btn", n_clicks=0),
#         html.Div(id="shewhart-output"),
#     ])

# @app.callback(
#     Output("shewhart-output", "children"),
#     Input("shewhart-btn", "n_clicks"),
# )
# def handle_shewhart(_):
#     global chart_state, df_global
#     if df_global is None or "predicted_engagement_level" not in df_global.columns:
#         return html.Div("Run Random Forest prediction first.")
#     engagement_rate = df_global["predicted_engagement_level"].mean() / df_global["predicted_engagement_level"].max()
#     add_engagement_data(chart_state, engagement_rate)
#     anomaly = check_for_engagement_anomaly(chart_state)
#     return html.Div([
#         html.P(f"Engagement Rate: {engagement_rate:.2f}"),
#         html.P(f"Anomaly Detected: {'Yes' if anomaly else 'No'}")
#     ])

if __name__ == "__main__":
    app.run(debug=True)