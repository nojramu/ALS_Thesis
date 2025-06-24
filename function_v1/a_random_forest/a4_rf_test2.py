import pandas as pd
from a1_rf_training import train_cognitive_and_engagement_models
from a3_rf_predictor import predict_cognitive_load_and_engagement
# Define the path to your CSV file (Assuming you have mounted Google Drive and know the path)
# Replace with the actual path to your CSV file if it's different
csv_file_path = 'data/sample_training_data.csv'

# Train the models and capture the returned models and feature names
# Updated to unpack 3 values as returned by the function
trained_cognitive_model, trained_engagement_model, trained_feature_names = train_cognitive_and_engagement_models(csv_file_path)

# Combine the two trained models into a tuple to pass to the predict function
trained_models = (trained_cognitive_model, trained_engagement_model)


# Example new data for prediction as a DataFrame with original columns, including 'difficulty' as a number (0-10)
# The load_and_preprocess_data function will handle the conversion to numeric and filling missing values.
new_data_point_original_df = pd.DataFrame({
    'engagement_rate': [0.85], # 0-1
    'time_on_task_s': [501], # seconds
    'hint_ratio': [0.67], # 0-1
    'interaction_count': [14],
    'task_completed': [0],  # Can be int (0/1) or boolean (False/True)
    'quiz_score': [89.11], # 0-100
    'difficulty': [2], # Provide the difficulty as a number between 0-10
    'error_rate': [0.59], # 0-1
    'task_timed_out': [0], # Can be int (0/1) or boolean (False/True)
    'time_before_hint_used': [199] # The longer the better
})

# Make predictions using the original new data DataFrame and the captured models and feature names
if trained_models is not None and trained_feature_names is not None:
    predicted_cognitive_load, predicted_engagement_level = predict_cognitive_load_and_engagement(
        models=trained_models,
        feature_names=trained_feature_names,
        new_data_df=new_data_point_original_df # Pass the original new data DataFrame
    )

    if predicted_cognitive_load is not None and predicted_engagement_level is not None:
        print("Predicted Cognitive Load:", predicted_cognitive_load)
        print("Predicted Engagement Level:", predicted_engagement_level)
    else:
        print("Prediction failed.")
else:
    print("Models are not trained. Please check the training process.")