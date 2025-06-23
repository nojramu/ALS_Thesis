from b4_kf_plot2 import df_predictions
from b2_kf_filter import apply_kalman_filter

cognitive_load_measurements = df_predictions['cognitive_load'].values
smoothed_cognitive_load = apply_kalman_filter(cognitive_load_measurements)
df_predictions['smoothed_cognitive_load'] = smoothed_cognitive_load
print(df_predictions.head())