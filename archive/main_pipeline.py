import os
import random
import numpy as np

from system.data_handling import load_csv, preprocess_data, save_csv
from system.random_forest import train_models, predict, save_models
from system.kalman_filter import add_kalman_column
from system.simpsons_rule import simpsons_rule, discretize_simpsons_result
from system.ql_setup import define_state_space, define_action_space, initialize_q_table
from system.ql_training import train_q_learning_agent
from system.shewhart_control import initialize_control_chart, add_engagement_data, check_for_engagement_anomaly
from system.ql_analysis import plot_q_table_heatmap, print_policy_summary, plot_learning_curve
from system.plot_utils import plot_line_chart

# --- 0. Set random seeds for reproducibility ---
from system.random_forest import train_models, predict, save_models
random.seed(42)

# --- 1. Ensure output folders exist ---
os.makedirs("image", exist_ok=True)
os.makedirs("snapshot", exist_ok=True)

# --- 2. Load and preprocess data ---
feature_cols = [
    'engagement_rate', 'time_on_task_s', 'hint_ratio', 'interaction_count',
    'task_completed', 'quiz_score', 'difficulty', 'error_rate',
    'task_timed_out', 'time_before_hint_used'
]
target_cols = ['cognitive_load', 'engagement_level']

df = load_csv("data/sample_training_data.csv")
if df is None:
    raise FileNotFoundError("Failed to load data/sample_training_data.csv. Please check that the file exists and is readable.")
if not all(col in df.columns for col in feature_cols + target_cols):
    raise ValueError("Missing required columns in input data.")

df = preprocess_data(df, feature_cols, is_training_data=True)

# --- 3. Train Random Forest models ---
reg, clf, features, metrics = train_models(df, feature_cols, target_cols)

# --- 4. Predict on new data, smooth with Kalman, integrate with Simpson ---
reg_pred, clf_pred = predict((reg, clf), features, df)
if df is None:
    raise ValueError("DataFrame 'df' is None after prediction step.")
df['predicted_cognitive_load'] = reg_pred
df['predicted_engagement_level'] = clf_pred

if df is not None and 'predicted_cognitive_load' in df.columns and 'predicted_engagement_level' in df.columns:
    plot_line_chart(
        x=df.index,
        y=[df['predicted_cognitive_load'], df['predicted_engagement_level']],
        xlabel="Index",
        ylabel="Predicted Values",
        title="Random Forest Predictions",
        legend_labels=["Cognitive Load", "Engagement Level"],
        save_path="image/rf_predictions.png",
        show=False
    )
else:
    raise ValueError("DataFrame is None or missing required prediction columns for plotting.")

df = add_kalman_column(df, col='predicted_cognitive_load', new_col='smoothed_cognitive_load')

plot_line_chart(
    x=df.index,
    y=[df['predicted_cognitive_load'], df['smoothed_cognitive_load']],
    xlabel="Index",
    ylabel="Cognitive Load",
    title="Kalman Filter Smoothing",
    legend_labels=["Predicted", "Smoothed"],
    save_path="image/kalman_smoothing.png",
    show=False
)

h = 3  # Step size (adjust as appropriate for your data)
integral = simpsons_rule(df['smoothed_cognitive_load'], h)
bucket = discretize_simpsons_result(integral)

plot_line_chart(
    x=[0],
    y=[integral],
    xlabel="Integration",
    ylabel="Integral Value",
    title="Simpson's Rule Integral",
    legend_labels=["Integral"],
    save_path="image/simpsons_integral.png",
    show=False
)

# --- 5. RL: Setup and train Q-learning ---
states, state_to_index, index_to_state, num_states = define_state_space()
actions, action_to_index, index_to_action, num_actions = define_action_space()
q_table = initialize_q_table(num_states, num_actions)

num_episodes = 1000
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay_rate = 0.001
min_epsilon = 0.01

q_table, rewards, max_q_values, policy_evolution, _ = train_q_learning_agent(
    num_episodes=num_episodes,
    learning_rate=alpha,
    discount_factor=gamma,
    epsilon=epsilon,
    epsilon_decay_rate=epsilon_decay_rate,
    min_epsilon=min_epsilon,
    progress_interval=50
)

# --- 6. Visualization and diagnostics ---
plot_q_table_heatmap(q_table, filename="image/qtable_heatmap.png", show=False)
plot_learning_curve(rewards, window=10, filename="image/learning_curve.png", show=False)

# --- 7. Monitoring and feedback ---
chart_state = initialize_control_chart()
engagement_rate = df['predicted_engagement_level'].mean() / df['predicted_engagement_level'].max()
add_engagement_data(chart_state, engagement_rate)
if check_for_engagement_anomaly(chart_state):
    print("Engagement anomaly detected! Consider updating Q-table or adjusting difficulty.")

# --- 8. Print policy summary and diagnostics ---
print_policy_summary(q_table, state_to_index, index_to_action)

# --- 9. Save Q-table snapshot for reproducibility ---
np.save("snapshot/q_table_final.npy", q_table)