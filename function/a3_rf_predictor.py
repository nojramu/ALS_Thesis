import pandas as pd
import numpy as np

def predict_cognitive_load_and_engagement(models, feature_names, new_data_path=None, new_data_df=None, new_data_list=None):
  """
  Makes predictions for cognitive load and engagement level using trained models.

  Args:
    models (tuple): A tuple containing the trained cognitive load regressor
                    and engagement level classifier (returned by train_cognitive_and_engagement_models).
    feature_names (list): A list of the feature names the models were trained on.
                          This is returned by train_cognitive_and_engagement_models.
    new_data_path (str, optional): The path to a CSV file containing the new data for prediction.
                                   Either new_data_path, new_data_df, or new_data_list must be provided.
    new_data_df (pd.DataFrame, optional): A DataFrame containing the new data for prediction.
                                        Either new_data_path, new_data_df, or new_data_list must be provided.
    new_data_list (list, optional): A list of values representing a single data point for prediction.
                                    The order of values should match the training features.
                                    Either new_data_path, new_data_df, or new_data_list must be provided.

  Returns:
    tuple: A tuple containing:
           - predicted_cognitive_load (np.array): Predicted cognitive load values.
           - predicted_engagement_level (np.array): Predicted engagement level values.
           Returns (None, None) if models are not provided or data is invalid.
  """
  if models is None or len(models) != 2 or feature_names is None:
    print("Error: Invalid models or feature names provided.")
    return None, None

  if new_data_path is None and new_data_df is None and new_data_list is None:
      print("Error: Either new_data_path, new_data_df, or new_data_list must be provided.")
      return None, None

  rf_cognitive_load, rf_engagement_level = models
  train_features = feature_names # Use feature_names passed from training

  if new_data_list is not None:
      if len(new_data_list) != len(train_features):
          print(f"Error: The number of values in new_data_list ({len(new_data_list)}) does not match the number of training features ({len(train_features)}).")
          return None, None
      # Create a DataFrame from the list, using the training feature names as columns
      # Ensure correct data types here if possible, though load_and_preprocess_data handles conversion
      new_data_for_pred = pd.DataFrame([new_data_list], columns=train_features)

  elif new_data_path is not None:
      # Use the unified load_and_preprocess_data function
      new_data_for_pred = load_and_preprocess_data(csv_file_path=new_data_path)
      if new_data_for_pred is None:
           print(f"Error loading or preprocessing data from {new_data_path}")
           return None, None

  elif new_data_df is not None:
      new_data_for_pred = new_data_df.copy() # Work on a copy

  else:
      return None, None # Should not happen based on checks above


  # Ensure new_data_for_pred has the same columns as the training data
  # Add missing columns from training data with a value of 0
  for col in train_features:
      if col not in new_data_for_pred.columns:
          new_data_for_pred[col] = 0

  # Ensure columns are in the same order as the training data
  # This is crucial for consistent predictions
  try:
      new_data_for_pred = new_data_for_pred[train_features]
  except KeyError as e:
      print(f"Error: Feature '{e}' from training data not found in preprocessed new data.")
      # This might happen if a required original column was missing in the new_data input
      return None, None


  try:
      predicted_cognitive_load = rf_cognitive_load.predict(new_data_for_pred)
      predicted_engagement_level = rf_engagement_level.predict(new_data_for_pred)
      return predicted_cognitive_load, predicted_engagement_level
  except Exception as e:
      print(f"Error during prediction: {e}")
      return None, None