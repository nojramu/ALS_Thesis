# Adaptive Learning System Pipeline & App Plan

---

## 1. System Pipeline

### Preprocessing
- **Purpose:** Load, clean, and validate synthetic or user-uploaded CSV data.
- **Key Functions:**
    - `load_csv(csv_path)`: Load a CSV file into a DataFrame.
    - `preprocess_data(df, required_features, is_training_data=False)`: Check columns, convert types, fill missing values.
    - `save_csv(df, csv_path)`: Save a DataFrame to a CSV file.

### Prediction (Random Forest)
- **Purpose:** Train two separate models (one for cognitive load, one for engagement) at app start; predict cognitive load and engagement levels.
- **Key Functions:**
    - `train_models(df, feature_cols, target_cols, ...)`: Train Random Forest regressor and classifier, return models and metrics.
    - `predict(models, feature_names, new_data_df)`: Predict cognitive load and engagement using trained models.
    - `save_models(models, out_dir, prefix)`: Save trained models to disk.

### Feature Engineering
- **Purpose:** Enhance features for RL and analysis.
- **Key Functions:**
    - **Kalman Filter:** Smooth predicted cognitive load signals.
        - `add_kalman_column(df, col, new_col, **kf_kwargs)`: Apply Kalman filter to a column and add the result.
    - **Simpson’s Rule:** Integrate (area under curve) smoothed cognitive load.
        - `simpsons_rule(y, h)`: Numerically integrate a sequence using Simpson’s Rule.
    - **Discretization:** Convert integrated/smoothed cognitive load into discrete states for RL.
        - `discretize_simpsons_result(simpsons_integral_value, num_buckets, historical_integral_values)`: Discretize the integral result into a bucket.

### Q-Learning
- **Purpose:** Reinforcement learning for adaptive decision-making using discretized states.
- **Key Functions:**
    - `define_state_space()`: Define all possible RL states.
    - `define_action_space()`: Define all possible RL actions.
    - `initialize_q_table(num_states, num_actions)`: Create a zero-initialized Q-table.
    - `train_q_learning_agent(...)`: Run the Q-learning loop, update Q-table, track visit counts, rewards, and policy evolution.
    - `epsilon_greedy_action_selection(...)`: Select an action using epsilon-greedy policy, enforcing the task type sequence.
    - `update_q_table(...)`: Perform the Q-learning update for a state-action pair.

### Shewhart Control
- **Purpose:** Statistical process control/monitoring for anomaly detection and feedback. If a spike or dip in engagement is detected, the recommended difficulty is adjusted and the Q-table is updated.
- **Key Functions:**
    - `initialize_control_chart(window_size)`: Initialize the control chart state.
    - `add_engagement_data(chart_state, engagement_rate)`: Add a new engagement rate and update control limits.
    - `check_for_engagement_anomaly(chart_state)`: Check if the latest engagement rate is an anomaly.
    - `plot_shewhart_chart(chart_state, ...)`: Plot the control chart with engagement rates, control limits, and anomalies.

### Visualization
- **Purpose:** Visualize data, predictions, smoothed signals, RL learning curves, Q-tables, and control charts throughout the pipeline, with live updates as new data arrives or simulation progresses.
- **Key Functions:**
    - `plot_line_chart(...)`: Plot and save line charts (e.g., predictions, learning curves).
    - `plot_q_table_heatmap(q_table, filename, show)`: Plot and save a heatmap of the Q-table.
    - `plot_visit_counts_heatmap(visit_counts, save_path)`: Plot and save a heatmap of state-action visit counts.
    - `plot_learning_curve(rewards, window, filename, show)`: Plot and save the RL learning curve and moving average.

---

## 2. Current Functionality

- **Modular Pipeline:** Each stage (preprocessing, prediction, feature engineering, RL, control, visualization) is implemented as a separate module/class with clear interfaces.
- **Random Forest Models:** Trained at app start for cognitive load and engagement; used for prediction only (no real-time retraining).
- **Kalman + Simpson’s Rule:** Used to smooth and integrate cognitive load signals before discretization for RL.
- **Q-Learning:** Online, task-level updates; immediate reward feedback; policy and visit count tracking.
- **Shewhart Control:** Detects engagement anomalies and signals Q-learning for adaptive actions.
- **Visualization:** Matplotlib-based plots for all major pipeline outputs.
- **App Structure:** Sidebar navigation, main panel for visualizations and controls, data upload, simulation controls, parameter editing, and save/load state options.
- **No Authentication:** Intended for local, researcher use only.

---

## 3. Target Plans

### Dash App Implementation
- **Scaffold and implement an interactive Dash app** with:
    - Sidebar navigation for all pipeline stages (Preprocessing, Random Forest, Kalman Filter, Simpson’s Rule, Q-Learning, Shewhart Control, Visualization).
    - Main panel for each stage with:
        - Data upload (CSV)
        - Parameter controls (sliders, text boxes)
        - Simulation controls (step/run)
        - Live visualizations (plots, tables)
        - Save/load state options
    - Modular separation of frontend (layout/UI) and backend (callbacks/logic).

### Full Pipeline Integration
- Integrate all modules for seamless data flow from upload to adaptive feedback.

### User Interactivity
- Enable real-time parameter editing, simulation stepping, and live updates in the UI.

### Save/Load State in UI
- Allow export/import of simulation results through the app interface.

### Advanced Visualization
- Add more interactive and advanced visualizations (e.g., feature importance, real-time control charts).

### Documentation & Examples
- Provide comprehensive user documentation and example walkthroughs.

### CSV Data Formats
- Finalize detailed CSV data format specifications and module interface contracts.

---

### Model Adaptation & Personalization
- **RF Retraining Trigger:** Use engagement anomalies to flag when RF models need retraining; retrain periodically or on demand.
- **Shewhart Feedback Expansion:** Use anomalies to also flag retraining of RF or recalibration of feature engineering.
- **Batch/Session Updates:** Consider batch or session-level Q-learning updates or meta-learning strategies.
- **Per-User/Meta-Learning:** Implement per-user models or meta-learning for personalization.
- **No Multi-User Generalization:** Address by supporting per-user/session models and meta-learning.

### Feedback & Reward Improvements
- **Granular Feedback:** Incorporate more detailed feedback (e.g., partial completion, time-to-completion).
- **Reward Structure Review:** Regularly review and update reward logic to prevent gaming.
- **Reward Hacking:** Prevent agent from exploiting reward structure (e.g., always easy tasks).
- **State History:** Incorporate recent states/actions into RL state representation for temporal awareness.
- **Q-Learning Only Task-Level:** Enhance with batch/session-level or meta-learning adaptation.
- **Feedback Granularity:** Move beyond coarse feedback to nuanced engagement signals.

### Data Quality & Robustness
- **Robust Statistics:** Use median/IQR for control limits; filter outliers in Shewhart and reward calculation.
- **Data Quality/Outliers:** Address destabilization by filtering and robust statistics.
- **Feedback Handling:** Add logic for missing/delayed feedback (e.g., skip updates, impute).
- **Delayed/Missing Feedback:** Explicitly handle to prevent learning issues.
- **Kalman/Simpson’s Rule Tuning:** Validate and tune parameters; consider using only one smoothing/integration method if lag is an issue.
- **Kalman + Simpson’s Rule:** Avoid double smoothing and information loss by careful tuning or simplification.

### Control Feedback Robustness & Explainability
- **Control Feedback Robustness & Explainability:**
    - Use robust statistics for anomaly detection.
    - Monitor multiple metrics (engagement, errors, completion).
    - Require multiple anomalies before adaptation.
    - Visualize control limits and anomalies.
    - Provide natural language explanations for anomalies.
    - Make thresholds transparent and configurable.
- **Shewhart Only Feeds Q-Learning:** Expand to influence other modules as needed.

---