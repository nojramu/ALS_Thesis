# ALS_Thesis Project Documentation

Paper Title: Real-Time Adaptive Task Pathing Using Random Forest Prediction, Q-Learning, and Statistical Methods (Kalman, Simpson, Shewhart)

Made by: Engr. Marjon D. Umbay  
Created for paper article in compliance of the subject of:  
* Numerical Methods and Techniques  
* Technopreneurship and Innovation

This document provides an overview of the project structure and functionalities of key folders and modules.

# Folder Overview

- **system/**: Contains the main application code and modules for data processing, machine learning models, control charts, and simulations.
- **archive/**: Contains deprecated or legacy code that is no longer actively used but retained for reference.
- **data/**: Stores sample datasets used for training, testing, and simulation.
- **image/**: Contains generated images and plots from the application.
- **models/**: Stores trained machine learning model files.
- **snapshot/**: Contains saved snapshots of Q-learning tables and other state data.

# system/ Folder

This document provides detailed descriptions of the functionalities implemented in each file within the `system/` directory of the project.

---

# app.py

- Initializes the Dash web application with Bootstrap styling.
- Defines the main layout including a sidebar navigation and main content panel.
- Implements navigation callbacks to render different panels based on the URL path.
- Defines UI panels for:
  - **Preprocessing:** Upload and load CSV data for training and testing.
  - **Random Forest:** Configure training parameters, train random forest regression and classification models, and test models with custom input.
  - **Kalman Filter:** Upload or load prediction data and apply Kalman filter smoothing to cognitive load data.
  - **Simpson's Rule:** Configure the number of buckets and perform numerical integration of cognitive load using Simpson's Rule.
  - **Q-Learning:** Configure training parameters, train Q-learning agent, display Q-table heatmap, and test policy recommendations.
  - **Shewhart Control:** Display Shewhart control chart for engagement rate monitoring, simulate engagement rates, detect anomalies, and provide feedback input for difficulty adjustment.
  - **System Simulation:** Input simulation parameters, initialize and append simulation data, and display simulation graphs.
- Registers callbacks for navigation and panel interactions.

---

# callback.py

- Contains all Dash callbacks handling user interactions and data processing.
- Handles data upload, preprocessing, and display in the preprocessing panel.
- Manages training and testing of Random Forest models.
- Processes Kalman filter application on prediction data.
- Handles Simpson's Rule integration and bucket discretization.
- Manages Q-learning training, policy visualization, and testing.
- Controls Shewhart control chart updates, anomaly detection, and feedback UI visibility.
- Processes user feedback submission to adjust difficulty and update Q-table.
- Manages system simulation initialization and data appending.
- Utilizes global state variables for data, models, Q-table, and chart state.

---

# data_handling.py

- Provides functions to load CSV data and preprocess it for model training and testing.
- Handles missing value detection and data cleaning.
- Includes functions to convert columns to numeric and integer types with appropriate missing value handling.
- Supports displaying CSV data head and checking required columns.

---

# kalman_filter.py

- Implements Kalman filter smoothing on cognitive load data.
- Provides functions to apply the filter and add smoothed data as new columns.
- Includes example usage with real data and plotting.
- Supports saving smoothed data and visual comparison plots.
- Computes mean squared error (MSE) between raw and smoothed data for evaluation.

---

# plot_utils.py

- Contains utility functions for plotting charts and saving figures.
- Supports line charts, bar charts, heatmaps, and other visualizations used across modules.
- Provides both Matplotlib and Plotly plotting utilities.
- Includes functions to convert Matplotlib figures to images and save them with timestamped filenames.
- Supports interactive Plotly heatmaps and Q-table visualizations.
- Includes utilities for plotting state-action visit counts heatmaps.

---

# ql_analysis.py

- Provides functions to analyze Q-learning policies, including retrieving optimal and top actions for states.
- Includes functions to print and visualize policy summaries and learning curves.
- Supports saving and loading Q-table snapshots.
- Provides utilities for policy evolution tracking and learning curve statistics.
- Contains example usage demonstrating policy printing and plotting.

---

# ql_core.py

- Defines core Q-learning functions such as:
  - Task type sequencing enforcing A->B->C->D->A rotation.
  - Epsilon-greedy action selection with task type constraints.
  - Reward calculation based on cognitive load, engagement, and task completion.
  - Q-table update logic implementing the Q-learning algorithm.

---

# ql_setup.py

- Defines the state and action spaces for the Q-learning agent.
- Provides functions to initialize the Q-table and validate states and actions.
- States include discretized cognitive load, engagement level, task completion, and previous task type.
- Actions include task type and difficulty level combinations.

---

# ql_simulator.py

- Simulates environment responses to actions, including changes in cognitive load, engagement level, and task completion.
- Calculates rewards based on state transitions.
- Models probabilistic task completion based on engagement and difficulty.

---

# ql_training.py

- Implements the Q-learning training loop with logging and policy tracking.
- Supports epsilon decay and progress logging.
- Provides example test function to run training and print results.
- Visualizes visit counts and learning curves.

---

# random_forest.py

- Implements training and prediction functions for Random Forest regression and classification models.
- Supports model saving and loading.
- Includes example usage with data loading, training, prediction, and saving.

---

# shewhart_control.py

- Implements Shewhart control chart utilities for engagement rate monitoring.
- Calculates control limits (CL, UCL, LCL) using median and IQR.
- Detects anomalies and maintains an anomaly buffer.
- Provides functions to update the Q-table based on anomaly detection and user feedback.
- Supports plotting and saving control charts.
- Includes interactive CLI feedback interface for difficulty adjustment.

---

# simpsons_rule.py

- Implements Simpson's Rule numerical integration for cognitive load data.
- Provides discretization of integral results into buckets.
- Supports dynamic and historical range-based discretization.
- Includes quantitative analysis comparing Simpson's Rule with trapezoidal integration.
- Contains example usage demonstrating integration and discretization.

---

This detailed documentation summarizes the key functionalities of each module to aid understanding, maintenance, and further development of the system.
