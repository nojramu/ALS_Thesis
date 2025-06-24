import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from b_kalman_filter.b3_kf_store import load_predictions
from c0_sr_integrate import simpsons_rule

df_predictions = load_predictions()
if df_predictions is None:
    exit(1)

# Assuming the data points are equally spaced by 1 (since they are row numbers)
# If the data represents a time series with a different time step, 'h' should be that time step.
h = 3
cognitive_load_values = df_predictions['smoothed_cognitive_load'].values


# Apply Simpson's Rule
simpsons_integral = simpsons_rule(cognitive_load_values, h)

if simpsons_integral is not None:
    print(f"Approximate integral of smoothed cognitive load using Simpson's Rule: {simpsons_integral}")