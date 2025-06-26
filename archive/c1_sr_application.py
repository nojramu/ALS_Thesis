from b0_data_utils import load_predictions
from b2_kf_filter import apply_kalman_filter  # Import Kalman filter
from c0_sr_integrate import simpsons_rule

df_predictions = load_predictions()
if df_predictions is None:
    exit(1)

# Use smoothed_cognitive_load if available, otherwise compute it
if 'smoothed_cognitive_load' in df_predictions.columns:
    cognitive_load_values = df_predictions['smoothed_cognitive_load'].values
elif 'cognitive_load' in df_predictions.columns:
    # Apply Kalman filter if not already present
    cognitive_load_values = apply_kalman_filter(df_predictions['cognitive_load'].values)
else:
    print("Error: No suitable cognitive load column found.")
    exit(1)

h = 3

# Apply Simpson's Rule
simpsons_integral = simpsons_rule(cognitive_load_values, h)

if simpsons_integral is not None:
    print(f"Approximate integral of smoothed cognitive load using Simpson's Rule: {simpsons_integral}")