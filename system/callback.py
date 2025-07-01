from app import app
import dash
from dash import html, dcc, Input, Output, State, dash_table, ctx
from dash.dependencies import ALL
import pandas as pd
import base64
import io

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
import plotly.graph_objs as go

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

# --- All callbacks go here ---

@app.callback(
    Output("preprocessing-output", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    Input("load-default-btn", "n_clicks"),
    prevent_initial_call=True
)
def handle_preprocessing(uploaded_contents, uploaded_filename, _):
    global df_global
    ctx = ctx or dash.callback_context
    if not ctx.triggered:
        return ""
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
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

@app.callback(
    Output("rf-output", "children"),
    Input("train-rf-btn", "n_clicks"),
    State("rf-test-size", "value"),
    State("rf-random-state", "value"),
    State("rf-n-estimators", "value"),
    prevent_initial_call=True
)
def handle_rf_train(train_clicks, test_size, random_state, n_estimators):
    global df_global, rf_models, features, metrics
    ctx = ctx or dash.callback_context
    if not ctx.triggered:
        return ""
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
    if rf_models is not None and train_clicks > 1:
        retrain_msg = html.P("Models retrained.", style={"color": "orange", "fontWeight": "bold"})
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

@app.callback(
    Output("sys-simulation-output", "children"),
    Input("sys-sim-init-btn", "n_clicks"),
    Input("sys-sim-append-btn", "n_clicks"),
    *[State(f"sys-sim-{f}", "value") for f in [
        'engagement_rate', 'time_on_task_s', 'hint_ratio', 'interaction_count',
        'task_completed', 'quiz_score', 'difficulty', 'error_rate',
        'task_timed_out', 'time_before_hint_used', 'prev_type'
    ]],
    prevent_initial_call=True
)
def handle_sys_sim_actions(init_clicks, append_clicks, *values):
    import dash
    ctx = dash.callback_context
    global rf_models, q_table
    param_names = [
        'engagement_rate', 'time_on_task_s', 'hint_ratio', 'interaction_count',
        'task_completed', 'quiz_score', 'difficulty', 'error_rate',
        'task_timed_out', 'time_before_hint_used', 'prev_type'
    ]
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Model check for both buttons
    missing = []
    if rf_models is None:
        missing.append("Random Forest models")
    if q_table is None:
        missing.append("Q-Learning agent")
    if trigger == "sys-sim-init-btn":
        if missing:
            return html.Div(
                f"Please train the following before initializing the simulation: {', '.join(missing)}.",
                style={"color": "red", "fontWeight": "bold"}
            )
        return html.Div(
            "Simulation initialized! (You can now proceed with your simulation steps.)",
            style={"color": "green", "fontWeight": "bold"}
        )

    # Validation for append
    if trigger == "sys-sim-append-btn":
        # Unpack values
        (
            engagement_rate, time_on_task_s, hint_ratio, interaction_count,
            task_completed, quiz_score, difficulty, error_rate,
            task_timed_out, time_before_hint_used, prev_type
        ) = values

        errors = []
        if not (0.0 <= float(engagement_rate) <= 1.0):
            errors.append("Engagement Rate must be between 0.0 and 1.0.")
        if not (isinstance(time_on_task_s, (int, float)) and time_on_task_s > 0):
            errors.append("Time on Task must be a positive number.")
        if not (0.0 <= float(hint_ratio) <= 1.0):
            errors.append("Hint Ratio must be between 0.0 and 1.0.")
        if not (isinstance(interaction_count, (int, float)) and interaction_count >= 0):
            errors.append("Interaction Count must be a non-negative number.")
        if task_completed not in [0, 1]:
            errors.append("Task Completed must be 0 or 1.")
        if not (0 <= int(quiz_score) <= 100):
            errors.append("Quiz Score must be between 0 and 100.")
        if not (isinstance(difficulty, (int, float)) and difficulty > 0):
            errors.append("Difficulty must be a positive number.")
        if not (0.0 <= float(error_rate) <= 1.0):
            errors.append("Error Rate must be between 0.0 and 1.0.")
        if task_timed_out not in [0, 1]:
            errors.append("Task Timed Out must be 0 or 1.")
        if not (isinstance(time_before_hint_used, (int, float)) and time_before_hint_used >= 0):
            errors.append("Time Before Hint Used must be a non-negative number.")
        if str(prev_type).upper() not in ["A", "B", "C", "D"]:
            errors.append("Previous Task Type must be one of: A, B, C, D.")

        if errors:
            return html.Div([
                html.P("Input validation failed:", style={"color": "red", "fontWeight": "bold"}),
                html.Ul([html.Li(e) for e in errors])
            ])
        # If all checks pass, you can append/store/process the data as needed
        return html.Div(
            "Input parameters appended successfully!",
            style={"color": "green", "fontWeight": "bold"}
        )

