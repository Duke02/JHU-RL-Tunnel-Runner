import RL_Final_Project_agentS as ag1
import RL_Final_Project_agentQ as ag2
import RL_final_env as env1
import numpy as np
from collections import deque

excluded_states = {12, 61, 110, 159, 208, 257, 306,
                   13, 62, 111, 160, 209, 258, 307,
                   14, 63, 112, 161, 210, 259, 308,
                   15, 64, 113, 162, 211, 260, 309,
                   16, 65, 114, 163, 212, 261, 310,
                   17, 66, 115, 164, 213, 262, 311,
                   24, 73, 122, 171, 220, 269, 318,
                   25, 74, 123, 172, 221, 270, 319,
                   26, 75, 124, 173, 222, 271, 320}


def calculate_visited_percentage(state_action_visit_count, excluded_states):
    # Update to consider all states in the environment
    total_pairs = 0
    visited_pairs = 0
    for state in range(343):  # Adjust to cover all states
        if state not in excluded_states:
            for action in range(4):  # Assuming 4 actions
                if state_action_visit_count[state, action] > 0:
                    visited_pairs += 1
                total_pairs += 1

    visited_percentage = visited_pairs / total_pairs if total_pairs else 0
    return visited_percentage


def is_win(final_state):
    win_states = {43, 92, 141, 190, 239, 288, 337}  # Define winning states
    return final_state in win_states


def train_single_agent(agent, environment, episodes=1, window_size=30):
    moving_averages = np.zeros(episodes - window_size + 1)
    moving_window_rewards = deque(maxlen=window_size)
    win_counts = np.zeros(episodes)  # Change to episodes if tracking win per episode
    state_action_visit_count = np.zeros((agent.get_number_of_states(), agent.get_number_of_actions()))
    total_episode_reward_list = []
    final_states = np.zeros(episodes, dtype=int)  # To store final state of each episode
    total_episode_rewards = np.zeros(episodes)
    state_action_visit_count = np.zeros((agent.get_number_of_states(), agent.get_number_of_actions()))
    for episode in range(episodes):
        agent.start_episode()
        current_state = environment.reset()
        game_end = False
        episode_reward = 0

        while not game_end:
            action = agent.select_action(current_state)
            new_state, reward, game_end = environment.execute_action(action)
            next_action = None if game_end else agent.select_action(new_state)

            if isinstance(agent, ag1.SARSA_0):
                agent.update_Q_SARSA(current_state, action, reward, new_state, next_action)
            else:  # Assuming it's the Q-learning agent
                agent.update_q_value(current_state, action, reward, new_state)

            state_action_visit_count[current_state][action] += 1
            episode_reward += reward
            current_state = new_state

        total_episode_rewards[episode] = episode_reward
        final_states[episode] = current_state  # Update final state for the episode

    # Calculate win counts based on final states
    for episode in range(episodes):
        win_counts[episode] = is_win(final_states[episode])

    # Calculate win rate for the most recent 30 episodes
    win_rate_last_30 = np.mean(win_counts[-30:])

    visited_percentage = calculate_visited_percentage(state_action_visit_count, excluded_states)
    moving_avg_rewards = np.convolve(total_episode_rewards, np.ones(window_size) / window_size, mode='valid')

    return moving_avg_rewards, win_rate_last_30, visited_percentage


def main():
    environment = env1.TunnelRunner()
    episodes = 1000
    window_size = 30
    num_agents = 10

    # Initialize metrics storage
    sarsa_averages_list = []
    q_averages_list = []
    sarsa_win_rate_list = []
    q_win_rate_list = []
    sarsa_visited_list = []
    q_visited_list = []
    sarsa_visited2_list = []
    q_visited2_list = []
    sarsa_moving_avg_rewards_list = []
    q_moving_avg_rewards_list = []

    # Train SARSA agents
    print("Training SARSA agents...")
    for _ in range(num_agents):
        agent_sarsa = ag1.SARSA_0()  # Initialize inside loop to ensure separate instances
        sarsa_averages, sarsa_win_rate, sarsa_visited = train_single_agent(agent_sarsa, environment,
                                                                           episodes, window_size)
        sarsa_moving_avg_rewards_list.append(sarsa_averages)
        sarsa_win_rate_list.append(sarsa_win_rate)
        sarsa_visited_list.append(sarsa_visited)

    # Train Q-learning agents
    print("Training Q-learning agents...")
    for _ in range(num_agents):
        agent_q = ag2.QAgent()  # Initialize inside loop to ensure separate instances
        q_averages, q_win_rate, q_visited = train_single_agent(agent_q, environment, episodes, window_size)
        q_moving_avg_rewards_list.append(q_averages)
        q_win_rate_list.append(q_win_rate)
        q_visited_list.append(q_visited)

    # Calculate average metrics
    avg_sarsa_moving_avg_rewards = np.mean(sarsa_moving_avg_rewards_list, axis=0)
    avg_q_moving_avg_rewards = np.mean(q_moving_avg_rewards_list, axis=0)
    avg_sarsa_win_rate = np.mean(sarsa_win_rate_list)
    avg_q_win_rate = np.mean(q_win_rate_list)
    avg_sarsa_visited = np.mean(sarsa_visited_list)
    avg_q_visited = np.mean(q_visited_list)

    # Save metrics for both agents
    np.savetxt('avg_sarsa_moving_averages.txt', avg_sarsa_moving_avg_rewards)
    np.savetxt('avg_q_moving_averages.txt', avg_q_moving_avg_rewards)

    # Print averaged metrics
    print(f"Avg SARSA win rate (last 30 episodes): {avg_sarsa_win_rate * 100:.2f}%")
    print(f"Avg Q-learning win rate (last 30 episodes): {avg_q_win_rate * 100:.2f}%")
    print("Training of all agents completed successfully.")


if __name__ == "__main__":
    main()
