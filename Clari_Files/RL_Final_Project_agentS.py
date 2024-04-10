import numpy as np


def get_state_index(x, y, z):
    x_idx = x
    y_idx = 6 * y  # Adjust to match environment's calculation
    z_idx = 49 * z  # Adjust to match environment's calculation
    return x_idx + y_idx + z_idx  # Reflects the environment's state calculation


class SARSA_0:
    """ On policy SARSA RL agent, updated for the TunnelRunner environment."""

    def __init__(self):
        self.initial_state = get_state_index(0, 0, 0)
        self.expl_x = 0  # Explorer's x position from 0 to 7
        self.expl_y = 0  # Explorer's y position from 0 to 7
        self.expl_z = 0  # Explorer's z position now ranges from 0 to 6 for the 7 levels
        self.state = self.get_state()
        self.num_states = 343  # Updated to reflect the total number of states in the environment
        self.num_actions = 4  # Assuming the number of actions remains the same
        self.episode = []  # To store state, action, reward for the episode
        self.gamma = 0.99  # Discount rate
        self.epsilon = 0.1  # Epsilon for exploration-exploitation trade-off
        self.alpha = 0.1  # Learning rate
        self.action = 0
        self.epsilon_max = 1.0  # Starting value of epsilon
        self.epsilon_min = 0.01  # Minimum value of epsilon
        self.N = 1000  # Total number of episodes to reduce epsilon
        self.n_step = 0  # Counter for the number of steps (for epsilon decay)
        self.q = np.zeros((self.num_states, self.num_actions), dtype="float64")  # State-action values array, resized

    # Get the key environment parameters
    def get_number_of_states(self):
        return self.num_states

    def update_epsilon(self):
        # Calculate decayed epsilon based on the number of episodes completed
        r = max((self.N - self.n_step) / self.N, 0)
        self.epsilon = (self.epsilon_max - self.epsilon_min) * r + self.epsilon_min
        self.n_step += 1  # Increment the episode counter

    def get_number_of_actions(self):
        return self.num_actions

    def get_state(self):
        return get_state_index(self.expl_x, self.expl_y, self.expl_z)

    def e_greedy(self, state):
        self.state = state
        rng = np.random.default_rng()
        a_star = np.argmax(self.q[state])
        if rng.random() > self.epsilon:
            return a_star
        else:
            return np.random.choice(self.num_actions)

    def select_action(self, state):
        if state >= self.num_states:
            # Adjust state or handle it appropriately to avoid out-of-bounds access
            state = self.num_states - 1  # Example adjustment
        self.update_epsilon()  # Call update_epsilon whenever you're selecting a new action
        action = self.e_greedy(state)
        return action

    def start_episode(self):
        self.episode = []

    def store_step(self, state, action, reward):
        self.episode.append((state, action, reward))

    def update_Q_SARSA(self, s, a, r, next_s, next_a):
        # Apply the SARSA update rule directly on self.q
        Q_next = self.q[next_s][next_a] if next_a is not None else 0  # If next state is goal, Q_next is 0.
        self.q[s][a] = self.q[s][a] + self.alpha * (r + self.gamma * Q_next - self.q[s][a])
