import numpy as np
from ql_setup import is_valid_state
import matplotlib.pyplot as plt
import seaborn as sns
from plot_utils import plotly_qtable_heatmap

def get_optimal_action_for_state(current_state, q_table, state_to_index, index_to_action):
    if not is_valid_state(current_state, state_to_index):
        return None
    state_index = state_to_index[current_state]
    optimal_action_index = np.argmax(q_table[state_index, :])
    return index_to_action[optimal_action_index]

def get_top_n_actions_for_state(current_state, q_table, state_to_index, index_to_action, n=5):
    if not is_valid_state(current_state, state_to_index):
        return None
    state_index = state_to_index[current_state]
    q_values = q_table[state_index, :]
    sorted_action_indices = np.argsort(q_values)[::-1]
    top_n_action_indices = sorted_action_indices[:n]
    return [(index_to_action[i], q_values[i]) for i in top_n_action_indices]

def print_policy_for_state(current_state, q_table, state_to_index, index_to_action, n=5):
    top_actions = get_top_n_actions_for_state(current_state, q_table, state_to_index, index_to_action, n)
    if top_actions is None:
        print("Invalid state.")
        return
    print(f"Top {n} actions for state {current_state}:")
    for action, q_value in top_actions:
        print(f"  Action: {action}, Q-value: {q_value:.3f}")

def plot_q_table_heatmap(q_table, filename="image/qtable_heatmap.png", show=False):
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
    Returns a dict mapping state index to optimal action index.
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
    Save a Q-table snapshot to disk (numpy .npy format).
    """
    np.save(filename, q_table)
    print(f"Q-table snapshot saved to {filename}")

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