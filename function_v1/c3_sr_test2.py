from c0_sr_integrate import simpsons_rule
from c2_sr_discretize import discretize_simpsons_result
from c1_sr_application import simpsons_integral

if simpsons_integral is not None:
    # Discretize the integral into 5 buckets
    num_buckets = 5 # Adjustable number of buckets
    bucket_number = discretize_simpsons_result(simpsons_integral, num_buckets=num_buckets)

    if bucket_number is not None:
        print(f"Simpson's Integral: {simpsons_integral}")
        print(f"Discretized Bucket Number: {bucket_number}")

    # Example with a different number of buckets
    num_buckets_alt = 7
    bucket_number_alt = discretize_simpsons_result(simpsons_integral, num_buckets=num_buckets_alt)

    if bucket_number_alt is not None:
        print(f"\nDiscretized into {num_buckets_alt} buckets:")
        print(f"Discretized Bucket Number: {bucket_number_alt}")