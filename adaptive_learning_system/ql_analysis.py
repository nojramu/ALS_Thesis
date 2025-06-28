import os
import numpy as np
from ql_setup import is_valid_state
import matplotlib.pyplot as plt
import seaborn as sns
from plot_utils import plotly_qtable_heatmap
from ql_core import get_next_task_type_in_sequence
import plotly.graph_objs as go
import plotly.express as px

def get_optimal_action_for_state(current_state, q_table, state_to_index, index_to_action, action_to_index):
    """
    Returns the optimal action for a given state based on the Q-table,
    enforcing the task type sequence.
    """
    if not is_valid_state(current_state, state_to_index):
        return None
    state_index = state_to_index[current_state]
    prev_task_type = current_state[3]
    required_task_type = get_next_task_type_in_sequence(prev_task_type)
    # Filter valid actions
    valid_action_indices = [
        idx for idx, action in index_to_action.items()
        if action[0] == required_task_type
    ]
    if not valid_action_indices:
        return None
    valid_q_values = q_table[state_index, valid_action_indices]
    optimal_idx_in_valid = np.argmax(valid_q_values)
    optimal_action_index = valid_action_indices[optimal_idx_in_valid]
    return index_to_action[optimal_action_index]

def get_top_n_actions_for_state(current_state, q_table, state_to_index, index_to_action, action_to_index, n=5):
    """
    Returns the top-n actions and their Q-values for a given state,
    enforcing the task type sequence.
    """
    if not is_valid_state(current_state, state_to_index):
        return None
    state_index = state_to_index[current_state]
    prev_task_type = current_state[3]
    required_task_type = get_next_task_type_in_sequence(prev_task_type)
    valid_action_indices = [
        idx for idx, action in index_to_action.items()
        if action[0] == required_task_type
    ]
    if not valid_action_indices:
        return None
    valid_q_values = q_table[state_index, valid_action_indices]
    sorted_indices = np.argsort(valid_q_values)[::-1][:n]
    top_n_action_indices = [valid_action_indices[i] for i in sorted_indices]
    return [(index_to_action[i], q_table[state_index, i]) for i in top_n_action_indices]

def print_policy_for_state(current_state, q_table, state_to_index, index_to_action, n=5):
    """
    Prints the top-n actions and their Q-values for a given state.
    """
    top_actions = get_top_n_actions_for_state(current_state, q_table, state_to_index, index_to_action, n)
    if top_actions is None:
        print("Invalid state.")
        return
    print(f"Top {n} actions for state {current_state}:")
    for action, q_value in top_actions:
        print(f"  Action: {action}, Q-value: {q_value:.3f}")

def plot_q_table_heatmap(q_table, filename="image/qtable_heatmap.png", show=False):
    """
    Plots a heatmap of the Q-table using seaborn and saves it as an image.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(q_table, cmap="viridis")
    plt.title("Q-table Heatmap")
    plt.xlabel("Action Index")
    plt.ylabel("State Index")
    plt.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()
    plt.close()

def plot_q_table_heatmap_plotly(q_table, save_path=None, show=False):
    """
    Interactive Q-table heatmap using Plotly.
    """
    plotly_qtable_heatmap(q_table, save_path=save_path, show=show)

def extract_policy(q_table, state_to_index, index_to_action):
    """
    Returns a dict mapping state to its optimal action.
    """
    policy = {}
    for state, idx in state_to_index.items():
        optimal_action_idx = np.argmax(q_table[idx, :])
        policy[state] = index_to_action[optimal_action_idx]
    return policy

def track_policy_evolution(policy_evolution_list, state, index_to_action):
    """
    Given a list of optimal action indices for a tracked state over time,
    returns the corresponding action tuples.
    """
    return [index_to_action[a] for a in policy_evolution_list]

def save_q_table_snapshot(q_table, filename):
    """
    Save a Q-table snapshot to disk (numpy .npy format) in the snapshot folder.
    """
    snapshot_dir = os.path.join(os.path.dirname(__file__), '..', 'snapshot')
    os.makedirs(snapshot_dir, exist_ok=True)
    full_path = os.path.join(snapshot_dir, filename)
    np.save(full_path, q_table)
    print(f"Q-table snapshot saved to {full_path}")

def load_q_table_snapshot(filename):
    """
    Load a Q-table snapshot from disk.
    """
    return np.load(filename)

def plot_learning_curve(rewards, window=10, filename="image/learning_curve.png", show=False):
    """
    Plot the learning curve and moving average.
    """
    rewards = np.array(rewards)
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward per Episode")
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f"{window}-episode Moving Avg")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()
    plt.close()

def learning_curve_stats(rewards):
    """
    Print statistics for the learning curve.
    """
    rewards = np.array(rewards)
    print(f"Learning Curve Stats:")
    print(f"  Mean reward: {np.mean(rewards):.2f}")
    print(f"  Max reward: {np.max(rewards):.2f}")
    print(f"  Min reward: {np.min(rewards):.2f}")
    print(f"  Std reward: {np.std(rewards):.2f}")

def print_policy_summary(q_table, state_to_index, index_to_action, sample_states=None, n=5):
    """
    Print optimal and top-N actions for a list of sample states.
    """
    if sample_states is None:
        sample_states = list(state_to_index.keys())[:5]
    for state in sample_states:
        print_policy_for_state(state, q_table, state_to_index, index_to_action, n=n)

def print_policy_examples(q_table, state_to_index, index_to_action, action_to_index, example_states, n=5):
    """
    Print optimal and top-N actions for a list of example states.
    """
    for state in example_states:
        print(f"\n--- Policy for State: {state} ---")
        optimal_action = get_optimal_action_for_state(state, q_table, state_to_index, index_to_action, action_to_index)
        if optimal_action is None:
            prev_task_type = state[3]
            required_task_type = get_next_task_type_in_sequence(prev_task_type)
            print(f"No valid actions for required next task type '{required_task_type}' in this state.")
        else:
            print(f"Optimal action: {optimal_action}")
        top_actions = get_top_n_actions_for_state(state, q_table, state_to_index, index_to_action, action_to_index, n=n)
        print("Top actions:")
        if top_actions is not None and len(top_actions) > 0:
            for action, q_value in top_actions:
                print(f"  {action}: {q_value:.4f}")
        else:
            print("  No actions available for this state.")

if __name__ == "__main__":
    from ql_setup import define_state_space, define_action_space
    from ql_training import train_q_learning_agent
    from plot_utils import plot_line_chart

    # --- Training ---
    q_table, rewards, max_q_values, policy_evolution = train_q_learning_agent(
        num_episodes=200,
        max_steps_per_episode=30,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay_rate=0.005,
        min_epsilon=0.05,
        reward_mode="state",
        progress_interval=20
    )

    # --- Analysis and Visualization ---
    print("\n--- Learned Policy Examples ---")
    states, state_to_index, index_to_state, num_states = define_state_space()
    actions, action_to_index, index_to_action, num_actions = define_action_space()

    example_states = [
        (3, 5, 1, 'D'),  # Mid-cognitive load, High engagement, Task Completed, Previous task D -> Next should be A
        (1, 1, 0, 'A'),  # Low cognitive load, Low engagement, Task Not Completed, Previous task A -> Next should be B
        (5, 3, 0, 'B'),  # High cognitive load, Mid engagement, Task Not Completed, Previous task B -> Next should be C
        (2, 4, 1, 'C'),  # Low-ish cognitive load, High engagement, Task Completed, Previous task C -> Next should be D
        (3, 3, 0, 'A')   # The starting state used in training, Previous task A -> Next should be B
    ]

    print_policy_examples(q_table, state_to_index, index_to_action, action_to_index, example_states, n=5)

    # Plot learning curve
    plot_line_chart(
        x=list(range(1, len(rewards) + 1)),
        y=[rewards],
        xlabel='Episode',
        ylabel='Total Reward',
        title='Total Reward per Episode',
        legend_labels=['Total Reward'],
        show=True
    )

    # Plot Q-table heatmap
    plot_q_table_heatmap(q_table)