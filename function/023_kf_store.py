cognitive_load_measurements = df_predictions['cognitive_load'].values
smoothed_cognitive_load = apply_kalman_filter(cognitive_load_measurements)
df_predictions['smoothed_cognitive_load'] = smoothed_cognitive_load
display(df_predictions.head())