# --- Validation Functions ---

def is_valid_state(state_tuple: tuple, state_to_index: dict) -> bool:
    """
    Checks if a given state tuple is a valid state within the defined state space.

    Validity is determined by whether the state tuple exists as a key in the
    provided state-to-index mapping.

    Args:
        state_tuple (tuple): The state tuple to validate.
        state_to_index (dict): Dictionary mapping state tuples to their indices,
                               representing the set of valid states.

    Returns:
        bool: True if the state is valid, False otherwise. Prints an error message if invalid.
    """
    if state_tuple in state_to_index:
        return True
    else:
        print(f"Validation Error: Invalid state tuple {state_tuple} not found in state space mapping.")
        return False


def is_valid_action(action_tuple: tuple, action_to_index: dict) -> bool:
    """
    Checks if a given action tuple is a valid action within the defined action space.

    Validity is determined by whether the action tuple exists as a key in the
    provided action-to-index mapping.

    Args:
        action_tuple (tuple): The action tuple to validate.
        action_to_index (dict): Dictionary mapping action tuples to their indices,
                                representing the set of valid actions.

    Returns:
        bool: True if the action is valid, False otherwise. Prints an error message if invalid.
    """
    if action_tuple in action_to_index:
        return True
    else:
        print(f"Validation Error: Invalid action tuple {action_tuple} not found in action space mapping.")
        return False