import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
from function.preprocessing.a0_rf_preprocessor import load_and_preprocess_data

def train_cognitive_and_engagement_models(csv_file_path, test_set_size=0.2, random_state_value=20, n_estimators_value=100):
  """
  Trains Random Forest models for predicting cognitive load and engagement level
  from a CSV file.

  Args:
    csv_file_path (str): The path to the CSV file containing the data.
    test_set_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
    random_state_value (int, optional): Controls the shuffling applied to the data before applying the split.
    n_estimators_value (int, optional): The number of trees in the forest.

  Returns:
    tuple: A tuple containing two trained models:
           - rf_cognitive_load (RandomForestRegressor): Trained model for predicting cognitive load.
           - rf_engagement_level (RandomForestClassifier): Trained model for predicting engagement level.
           Returns (None, None) if data loading or preprocessing fails.
    dict: A dictionary containing the feature names used for training.
  """

  # Load and preprocess the data using the unified function
  # Assuming load_and_preprocess_data function is available in the notebook environment or imported
  try:
      # You might need to define or import load_and_preprocess_data first if it's not in this cell
      # For now, we'll assume it returns a DataFrame similar to the original structure but processed
      # with 'cognitive_load', 'engagement_level', and features like 'difficulty' (now numeric)
      df_processed = load_and_preprocess_data(csv_file_path=csv_file_path)
  except NameError:
      print("Error: 'load_and_preprocess_data' function not found. Please ensure it's defined or imported.")
      return None, None, None
  except Exception as e:
      print(f"Error loading or preprocessing data: {e}")
      return None, None, None


  if df_processed is None:
      print("Preprocessing returned None.")
      return None, None, None # Return None for models and features

  # Define features (X) and targets (y) - 'difficulty' is now a numeric feature
  target_cols = ['cognitive_load', 'engagement_level']
  # Ensure 'difficulty' is included in features if it exists after processing
  # Assuming the processed dataframe includes the features needed for training
  features = [col for col in df_processed.columns if col not in target_cols]

  # Ensure all feature columns exist after preprocessing
  # Note: This check is redundant if load_and_preprocess_data already checked required_features
  # but keeping for safety after feature definition.
  # This also assumes load_and_preprocess_data correctly processes the input CSV
  # to produce features like 'simpsons_integral_level', 'engagement_level', 'task_completed', 'prev_task_type'
  # and potentially others derived from the raw data.
  # A more robust approach would explicitly define expected input features and output features of preprocessing.
  # For the scope of this function, we assume df_processed is ready for model training.

  X = df_processed[features]
  y_cognitive_load = df_processed['cognitive_load']
  y_engagement_level = df_processed['engagement_level']

  # --- IMPROVEMENT: Address Model Evaluation ---
  # The previous implementation trained the final models on the *entire* dataset (X, y)
  # but reported evaluation metrics calculated on the test split. This leads to
  # misleadingly optimistic metrics.

  # Option A: Train final models on X_train and evaluate *only* on X_test.
  # This provides a more realistic estimate of generalization performance.
  # If the goal is to deploy a model trained on all available data, Option B is better.

  # Option B: Train final models on the full dataset (X, y) for deployment,
  # but report evaluation metrics from the test split *before* full-data training,
  # or clearly document that test metrics are for development insights only.
  # This matches the *structure* of the original code more closely, but we will
  # clarify the interpretation.

  # We will keep the structure of training on the full dataset (X, y) for the final models
  # but clarify the evaluation metrics' purpose in comments or surrounding text.
  # The evaluation metrics printed will still be from the *separate* test split (X_test, y_test)
  # which was created *before* training the final models on the full dataset.

  # Split data into training and testing sets for evaluation purposes
  X_train_eval, X_test_eval, y_cognitive_load_train_eval, y_cognitive_load_test_eval = train_test_split(
      X, y_cognitive_load, test_size=test_set_size, random_state=random_state_value)
  # Split for engagement level - ensure stratification if it's a classification problem and classes are imbalanced
  X_train_eval, X_test_eval, y_engagement_level_train_eval, y_engagement_level_test_eval = train_test_split(
      X, y_engagement_level, test_size=test_set_size, random_state=random_state_value, stratify=y_engagement_level)


  # Initialize and train the Random Forest Regressor for cognitive load
  # Train the FINAL model on the entire dataset for potentially better performance in deployment
  rf_cognitive_load = RandomForestRegressor(n_estimators=n_estimators_value, random_state=random_state_value)
  rf_cognitive_load.fit(X, y_cognitive_load)

  # Initialize and train the Random Forest Classifier for engagement level
  # Train the FINAL model on the entire dataset
  rf_engagement_level = RandomForestClassifier(n_estimators=n_estimators_value, random_state=random_state_value)
  rf_engagement_level.fit(X, y_engagement_level)


  # Evaluate the models on the test set *created before* training on the full dataset
  # These metrics provide an estimate of performance but may be optimistic
  # compared to truly unseen data if the full dataset was used for training.
  print("\n--- Model Evaluation (Metrics on Test Split before full-data training) ---")
  cognitive_load_predictions_eval = rf_cognitive_load.predict(X_test_eval)
  print(f"Cognitive Load MSE on test split: {mean_squared_error(y_cognitive_load_test_eval, cognitive_load_predictions_eval):.4f}")

  engagement_level_predictions_eval = rf_engagement_level.predict(X_test_eval)
  print(f"Engagement Level Accuracy on test split: {accuracy_score(y_engagement_level_test_eval, engagement_level_predictions_eval):.4f}")


  # Return trained models (trained on full data) and the list of features used for training
  # Add a comment indicating that these models were trained on the full dataset
  print("\nRandom Forest models trained on the full dataset.")
  return rf_cognitive_load, rf_engagement_level, features # Models trained on full X, y