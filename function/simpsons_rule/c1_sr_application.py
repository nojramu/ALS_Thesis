from b4_kf_plot2 import df_predictions
from c0_sr_integrate import simpsons_rule

# Assuming the data points are equally spaced by 1 (since they are row numbers)
# If the data represents a time series with a different time step, 'h' should be that time step.
h = 3
cognitive_load_values = df_predictions['smoothed_cognitive_load'].values


# Apply Simpson's Rule
simpsons_integral = simpsons_rule(cognitive_load_values, h)

if simpsons_integral is not None:
    print(f"Approximate integral of smoothed cognitive load using Simpson's Rule: {simpsons_integral}")