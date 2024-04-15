import numpy as np


def get_state_index(x, y, z):
    x_idx = x
    y_idx = 6 * y  # Adjust to match environment's calculation
    z_idx = 49 * z  # Adjust to match environment's calculation
    return x_idx + y_idx + z_idx  # Reflects the environment's state calculation


class SARSA_0:
    """ On policy SARSA RL agent, updated for the TunnelRunner environment."""

    def __init__(self, num_episodes_to_decay_epsilon: int = 5_000, seed: int = 13):
        self.initial_state = get_state_index(0, 0, 0)
        self.expl_x = 0  # Explorer's x position from 0 to 7
        self.expl_y = 0  # Explorer's y position from 0 to 7
        self.expl_z = 0  # Explorer's z position now ranges from 0 to 6 for the 7 levels
        self.state = self.get_state()
        self.num_states = 343  # Updated to reflect the total number of states in the environment
        self.num_actions = 9  # Assuming the number of actions remains the same
        self.episode = []  # To store state, action, reward for the episode
        self.gamma = 0.9  # Discount rate

        self.alpha = 0.1  # Learning rate
        self.action = 0
        self.epsilon_initial = 0.9  # Starting value of epsilon
        self.epsilon_final = 0.1  # Minimum value of epsilon
        self.epsilon = self.epsilon_initial  # Epsilon for exploration-exploitation trade-off
        self.N = num_episodes_to_decay_epsilon  # Total number of episodes to reduce epsilon
        self.n_step = 0  # Counter for the number of steps (for epsilon decay)
        self.q = np.zeros((self.num_states, self.num_actions), dtype="float64")  # State-action values array, resized

        self.seed: int = seed
        self.rng = np.random.default_rng(seed=self.seed)

        # Data for converge rate
        self.last_qtable: np.ndarray | None = None
        self.convergence_rate: int = -1
        self.episodes_trained: int = 0
        self.convergence_tolerance: float = 1e-3

        # Data to save for sample efficiency
        self.cumulative_rewards: float = 0

        # Data to save for asymptotic performance
        self.rewards_since_convergence: float = 0
        self.rewards_for_curr_episode: float = 0

    @property
    def asymptotic_performance(self) -> float:
        return 0 if not self.is_converged else self.rewards_since_convergence / (self.episodes_trained - self.convergence_rate)

    @property
    def is_converged(self) -> bool:
        return self.convergence_rate > 0

    @property
    def sample_efficiency(self) -> float:
        return self.cumulative_rewards

    # Get the key environment parameters
    def get_number_of_states(self):
        return self.num_states

    def update_epsilon(self):
        # Calculate decayed epsilon based on the number of episodes completed
        decay_rate: float = (self.epsilon_final - self.epsilon_initial) / self.N
        self.epsilon = max(self.epsilon_final, self.epsilon + decay_rate)
        self.n_step += 1  # Increment the episode counter

    def get_number_of_actions(self):
        return self.num_actions

    def get_state(self):
        return get_state_index(self.expl_x, self.expl_y, self.expl_z)

    def e_greedy(self, state):
        self.state = state
        if self.rng.random() >= self.epsilon:
            return np.argmax(self.q[state])
        else:
            return self.rng.choice(self.num_actions)

    def select_action(self, state):
        if state >= self.num_states:
            # Adjust state or handle it appropriately to avoid out-of-bounds access
            state = self.num_states - 1  # Example adjustment
        self.update_epsilon()  # Call update_epsilon whenever you're selecting a new action
        action = self.e_greedy(state)
        return action

    def start_episode(self):
        self.episode = []


        self.cumulative_rewards += self.rewards_for_curr_episode
        if self.is_converged:
            self.rewards_since_convergence += self.rewards_for_curr_episode
        else:
            self.rewards_since_convergence = 0
        self.rewards_for_curr_episode = 0

        if self.episodes_trained > 0:
            has_small_difference: bool = np.isclose(self.last_qtable - self.q, 0, atol=self.convergence_tolerance).all()
            if not self.is_converged and has_small_difference:
                self.convergence_rate = self.episodes_trained
            elif self.is_converged and not has_small_difference:
                self.convergence_rate = -1
        self.last_qtable: np.ndarray = self.q.copy()
        self.episodes_trained += 1

    def store_step(self, state, action, reward):
        self.episode.append((state, action, reward))

    def update_Q_SARSA(self, s, a, r, next_s, next_a):
        self.rewards_for_curr_episode += r
        # Apply the SARSA update rule directly on self.q
        Q_next = self.q[next_s][next_a] if next_a is not None else 0  # If next state is goal, Q_next is 0.
        self.q[s][a] = self.q[s][a] + self.alpha * (r + self.gamma * Q_next - self.q[s][a])