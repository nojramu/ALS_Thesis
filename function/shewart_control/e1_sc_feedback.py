import matplotlib.pyplot as plt # Used for visualization
import sys # To handle input from user
from e0_sc_monitor import visualize_control_chart # Assuming this function is defined in e0_sc_monitor.py

# Assuming control chart functions (initialize_control_chart, visualize_control_chart, etc.)
# are defined in a previous cell.

def get_adjusted_decision(chart_state: dict, current_state: tuple, recommended_action: tuple) -> tuple:
    """
    Handles the interface for displaying control chart, system recommendations,
    and allowing manual adjustment of the recommended task difficulty by the user.

    Args:
        chart_state (dict): The dictionary representing the control chart state.
        current_state (tuple): The current state of the learner (for context).
        recommended_action (tuple): The action (task_type, difficulty) recommended
                                    by the Q-learning agent.

    Returns:
        tuple: A tuple containing:
            - adjusted_action (tuple): The adjusted action (task_type, adjusted_difficulty).
            - chart_fig (matplotlib.figure.Figure or None): The generated Matplotlib figure, or None if not enough data.
    """
    # 1. Display the current control chart
    print("\n--- Engagement Control Chart ---")
    # Use the visualization function
    chart_fig = visualize_control_chart(chart_state)
    if chart_fig:
        print("Matplotlib chart data generated.")
        # The calling code will display the figure using chart_fig.show()
    else:
        print("Not enough data to display the control chart.")

    # 2. Present system's recommended action
    recommended_task_type, recommended_difficulty = recommended_action
    print(f"\nSystem Recommendation:")
    print(f"  Task Type: {recommended_task_type}")
    print(f"  Difficulty: {recommended_difficulty}")
    print(f"  Current State: {current_state}")

    # 3. Provide mechanism for user adjustment
    while True:
        try:
            user_input = input(f"Enter desired difficulty (0-10) or press Enter to accept recommendation [{recommended_difficulty}]: ")
            if user_input == "":
                adjusted_difficulty = recommended_difficulty
                print("Accepted recommended difficulty.")
                break
            adjusted_difficulty = int(user_input)
            # 4. Validate user's input
            if 0 <= adjusted_difficulty <= 10:
                print(f"User adjusted difficulty to {adjusted_difficulty}.")
                break
            else:
                print("Invalid input. Difficulty must be an integer between 0 and 10.")
        except ValueError:
            print("Invalid input. Please enter an integer or press Enter.")
        except EOFError: # Handle potential issues in non-interactive environments
             print("\nEOF encountered. Using recommended difficulty.")
             adjusted_difficulty = recommended_difficulty
             break
        except Exception as e:
             print(f"An unexpected error occurred during input: {e}. Using recommended difficulty.")
             adjusted_difficulty = recommended_difficulty
             break

    # 5. Return the adjusted action and the chart figure
    adjusted_action = (recommended_task_type, adjusted_difficulty)
    print(f"Returning adjusted decision: {adjusted_action}")
    # Return both the adjusted action and the chart figure
    return adjusted_action, chart_fig