# callbacks.py
from dash import Input, Output, html, dcc
import sys
sys.path.append("/workspaces/ALS_Thesis")

# Import all needed functions from your backend modules
from adaptive_learning_system.data_handling import load_csv, preprocess_data, save_csv, display_csv_head
from adaptive_learning_system.random_forest import train_models, predict, save_models
from adaptive_learning_system.kalman_filter import add_kalman_column
from adaptive_learning_system.simpsons_rule import simpsons_rule, discretize_simpsons_result
from adaptive_learning_system.ql_setup import define_state_space, define_action_space, initialize_q_table, is_valid_state
from adaptive_learning_system.ql_training import train_q_learning_agent
from adaptive_learning_system.shewhart_control import (
    initialize_control_chart, add_engagement_data, check_for_engagement_anomaly, plot_shewhart_chart
)
from adaptive_learning_system.plot_utils import (
    plot_line_chart, plot_q_table_heatmap, plot_visit_counts_heatmap, plot_learning_curve
)

def register_callbacks(app):
    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        if pathname == "/preprocessing":
            return html.Div([
                html.H2("Preprocessing"),
                html.P("Upload, clean, and validate your CSV data here."),
                dcc.Upload(id="upload-data", children=html.Button("Upload CSV")),
                html.Div(id="preprocessing-output"),
            ])
        elif pathname == "/random-forest":
            return html.Div([
                html.H2("Random Forest"),
                html.P("Train and evaluate Random Forest models for cognitive load and engagement."),
                html.Button("Train Models", id="train-models-btn"),
                html.Div(id="rf-output"),
            ])
        elif pathname == "/kalman-filter":
            return html.Div([
                html.H2("Kalman Filter"),
                html.P("Apply Kalman filter smoothing to cognitive load predictions."),
                html.Div(id="kalman-output"),
            ])
        elif pathname == "/simpsons-rule":
            return html.Div([
                html.H2("Simpson’s Rule"),
                html.P("Integrate smoothed cognitive load using Simpson’s Rule."),
                html.Div(id="simpsons-output"),
            ])
        elif pathname == "/q-learning":
            return html.Div([
                html.H2("Q-Learning"),
                html.P("Train and analyze the Q-learning agent."),
                html.Button("Train Q-Learning Agent", id="train-ql-btn"),
                html.Div(id="ql-output"),
            ])
        elif pathname == "/shewhart-control":
            return html.Div([
                html.H2("Shewhart Control"),
                html.P("Monitor engagement and detect anomalies using Shewhart control charts."),
                html.Div(id="shewhart-output"),
            ])
        elif pathname == "/visualization":
            return html.Div([
                html.H2("Visualization"),
                html.P("View pipeline outputs and learning curves."),
                html.Div(id="visualization-output"),
            ])
        else:
            return html.Div([
                html.H2("Welcome to the Adaptive Learning System Dashboard"),
                html.P("Select a pipeline stage from the sidebar to begin."),
            ])