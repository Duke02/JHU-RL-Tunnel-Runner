import matplotlib.pyplot as plt
import numpy as np

# Load the data from the text files
sarsa_moving_averages = np.loadtxt('avg_sarsa_moving_averages.txt')
q_moving_averages = np.loadtxt('avg_q_moving_averages.txt')


def plot_moving_averages(sarsa_moving_averages, q_moving_averages):
    plt.figure(figsize=(10, 6))

    # Plot SARSA moving averages
    plt.plot(sarsa_moving_averages, label='SARSA Moving Average')

    # Plot Q-Learning moving averages
    plt.plot(q_moving_averages, label='Q-learning Moving Average')

    plt.title('Moving Average Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Plot the moving averages for both SARSA and Q-learning agents
plot_moving_averages(sarsa_moving_averages, q_moving_averages)
