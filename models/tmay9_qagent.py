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


def is_win(state: int) -> bool:
    return state == 126


class QAgent:
    def __init__(self, num_states: int, num_actions: int, initial_epsilon: float, discount_factor: float, learning_rate: float,
            terminal_states: tp.Set[int], final_epsilon: float, epsilon_step: float, initial_q_value: float):
        self.qtable: np.ndarray = initial_q_value * np.ones((num_states, num_actions))
        self.final_epsilon: float = final_epsilon
        self.epsilon_step: float = epsilon_step
        self.epsilon: float = initial_epsilon
        self.discount_factor: float = discount_factor
        self.learning_rate: float = learning_rate
        self.seed: int = 13
        self.last_action: int | None = None
        self.last_state_seen: int | None = None

        self.terminal_states: tp.Set[int] = terminal_states
        self.qtable[list(self.terminal_states), :] = 0
        self.curr_rewards: float = 0

    def reset(self):
        np.random.seed(self.seed)
        self.last_action = None
        self.last_state_seen = None

        # If we're to increase from init to final
        if self.epsilon_step > 0 and self.epsilon < self.final_epsilon:
            self.epsilon = min(self.final_epsilon, self.epsilon_step + self.epsilon)
        elif self.epsilon_step < 0 and self.epsilon > self.final_epsilon:
            self.epsilon = max(self.final_epsilon, self.epsilon + self.epsilon_step)

    def get_actions_for_state(self, state: int) -> np.ndarray:
        return self.qtable[state, :]

    def get_best_action(self, state: int) -> int:
        return np.argmax(self.get_actions_for_state(state))

    def choose_action(self, state: int) -> int:
        if np.random.uniform(low=0, high=1) > self.epsilon:
            return self.get_best_action(state)
        else:
            return np.random.randint(low=0, high=self.qtable.shape[1] - 1)

    def react_to_state(self, new_state: int, reward: float) -> int:
        self.curr_rewards += reward
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
