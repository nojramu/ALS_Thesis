import subprocess

# List of scripts to run (relative to the workspace root)
scripts = [
    "function_v1/a_random_forest/a4_rf_test2.py",
    "function_v1/b_kalman_filter/run_kalman_workflow.py",
    "function_v1/c_simpsons_rule/c3_sr_test2.py",
    "function_v1/d_q_learning/d6_ql_test.py",
    "function_v1/e_shewart_control/e2_sc_test.py"
]

for script in scripts:
    print(f"\n--- Running {script} ---")
    result = subprocess.run(["python3", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)