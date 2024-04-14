"""
Q-Learning:
Init Q table with 0 for everything
get state
choose action based on epsilon thing
take action, observe reward
Q(s, a) += lr * (reward + discount_factor * max(Q-value for any action in next state) - Q(s,a))
s = next state
"""
import numpy as np
import typing as tp
from collections import deque


class QAgent:
    def __init__(self, num_states: int, num_actions: int, initial_epsilon: float, discount_factor: float, learning_rate: float,
            terminal_states: tp.Set[int], win_states: set[int], final_epsilon: float, epsilon_step: float, initial_q_value: float, seed: int = 13):
        self.initial_q_value: float = initial_q_value
        self.qtable: np.ndarray = self.initial_q_value * np.ones((num_states + 1, num_actions))
        self.final_epsilon: float = final_epsilon
        self.epsilon_step: float = epsilon_step
        self.epsilon: float = initial_epsilon
        self.discount_factor: float = discount_factor
        self.learning_rate: float = learning_rate
        self.seed: int = seed
        self.random_num_gen = np.random.default_rng(seed=self.seed)
        self.last_action: int | None = None
        self.last_state_seen: int | None = None

        self.terminal_states: tp.Set[int] = terminal_states
        self.win_states: set[int] = win_states
        self.qtable[list(self.terminal_states), :] = 0
        self.curr_rewards: float = 0
        self.last_30_rewards: deque[float] = deque(maxlen=30)
        self.did_win_last_30: deque[bool] = deque(maxlen=30)

        # Data to save for convergence rate
        self.last_qtable: np.ndarray | None = None
        self.convergence_rate: int | None = None
        self.episodes_trained: int = 0
        self.convergence_tolerance: float = 1e-3

        # Data to save for sample efficiency
        self.cumulative_rewards: float = 0

        # Data to save for asymptotic performance
        self.rewards_since_convergence: float = 0

    @property
    def total_gain_over_last_30_episodes(self) -> float:
        return sum(self.last_30_rewards)

    @property
    def discounted_gain(self):
        return sum([(self.discount_factor ** i) * reward for i, reward in enumerate(self.last_30_rewards)])

    @property
    def asymptotic_performance(self) -> float:
        if not self.is_converged:
            return 0
        return self.rewards_since_convergence / (self.episodes_trained - self.convergence_rate)

    @property
    def sample_efficiency(self) -> float:
        return self.cumulative_rewards

    @property
    def is_converged(self) -> bool:
        return self.convergence_rate is not None and self.convergence_rate > 0

    @property
    def perc_states_visited(self) -> float:
        return np.sum(np.any(np.abs(self.qtable - self.initial_q_value) > 1e-6, axis=1)) / self.qtable.shape[0]

    def is_win(self, state: int) -> bool:
        return state in self.win_states

    def reset(self):
        if self.last_state_seen is not None:
            self.did_win_last_30.append(self.is_win(self.last_state_seen))
            self.last_30_rewards.append(self.curr_rewards)

        self.last_action = None
        self.last_state_seen = None

        if self.is_converged:
            self.rewards_since_convergence += self.curr_rewards
        else:
            self.rewards_since_convergence = 0
        self.curr_rewards = 0

        # If we're to increase from init to final
        if self.epsilon_step > 0 and self.epsilon < self.final_epsilon:
            self.epsilon = min(self.final_epsilon, self.epsilon_step + self.epsilon)
        elif self.epsilon_step < 0 and self.epsilon > self.final_epsilon:
            self.epsilon = max(self.final_epsilon, self.epsilon + self.epsilon_step)

        # If we've already trained
        if self.episodes_trained > 0:
            has_small_difference: bool = np.isclose(self.last_qtable - self.qtable, 0, atol=self.convergence_tolerance).all()
            # And we haven't seen a significant amount of update
            if not self.is_converged and has_small_difference:
                # Then we've converged at this episode
                self.convergence_rate = self.episodes_trained
            # But if we're still changing even after we've "converged"
            elif self.is_converged and not has_small_difference:
                # Then we haven't actually converged.
                self.convergence_rate = None
        self.last_qtable = self.qtable.copy()
        self.episodes_trained += 1

    def get_actions_for_state(self, state: int) -> np.ndarray:
        return self.qtable[state, :]

    def get_best_action(self, state: int) -> int:
        return np.argmax(self.get_actions_for_state(state))

    def choose_action(self, state: int) -> int:
        if self.random_num_gen.uniform(low=0, high=1) > self.epsilon:
            return self.get_best_action(state)
        else:
            return self.random_num_gen.integers(low=0, high=self.qtable.shape[1] - 1, size=1).item()

    def react_to_state(self, new_state: int, reward: float) -> int:
        self.curr_rewards += reward
        self.cumulative_rewards += reward
        if self.last_state_seen is not None:
            self.update_qtable(new_state, reward)
        self.last_state_seen = new_state
        self.last_action = self.choose_action(new_state)
        return self.last_action

    def update_qtable(self, new_state: int, reward: float):
        self.qtable[self.last_state_seen, self.last_action] += self.learning_rate * (
                reward
                + self.discount_factor * np.max(self.get_actions_for_state(new_state))
                - self.qtable[self.last_state_seen, self.last_action])
