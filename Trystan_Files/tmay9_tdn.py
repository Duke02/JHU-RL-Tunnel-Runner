import typing as tp

import numpy as np


class TrajectoryTuple(tp.TypedDict):
    curr_state: int
    curr_action: int
    next_state: int
    next_action: int
    curr_reward: float


class TDNAgent:
    def __init__(self, n: int, num_actions: int, num_states: int, epsilon: float, discount_factor: float, learning_rate: float,
                 terminal_states: set[int], win_states: set[int]):
        self.num_actions: int = num_actions
        self.num_states: int = num_states
        self.epsilon: float = epsilon
        self.discount_factor: float = discount_factor
        self.learning_rate: float = learning_rate
        self.q_table: np.ndarray = np.zeros((num_states, num_actions))
        self.terminal_states: set[int] = terminal_states

        self.last_state_seen: int | None = None
        self.last_action_performed: int | None = None

        self.n: int = n
        self.trajectory: list[TrajectoryTuple] = []
        # tau in the textbook, page 147
        self.internal_timer: int = 0
        self.total_time: int = 0

    def reset(self):
        self.trajectory.clear()
        self.internal_timer = 0
        self.total_time = 10 ** 25
        self.last_state_seen = None
        self.last_action_performed = None

    def get_all_actions_for_state(self, state: int) -> np.ndarray:
        return self.q_table[state, :]

    def select_an_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.get_all_actions_for_state(state))

    def react_to_state(self, new_state: int, reward: float, external_timer: int) -> int:
        if new_state in self.terminal_states:
            self.total_time = external_timer + 1

        next_action: int = self.select_an_action(new_state)
        if self.last_state_seen is not None:
            self.trajectory.append(
                TrajectoryTuple(curr_state=self.last_state_seen,
                                curr_action=self.last_action_performed,
                                curr_reward=reward,
                                next_state=new_state,
                                next_action=next_action))

        self.internal_timer = external_timer - self.n + 1

        if self.internal_timer >= 0:
            self.update_qtable()

        self.last_state_seen = new_state
        self.last_action_performed = next_action
        return self.last_action_performed

    def update_qtable(self):
        gain: float = np.sum([self.discount_factor ** (i - self.internal_timer - 1) * self.trajectory[i]['curr_reward'] for i in
                              range(self.internal_timer + 1, min(self.internal_timer + self.n, self.total_time))])
        if self.internal_timer + self.n < self.total_time:
            chosen_traj_tuple: TrajectoryTuple = self.trajectory[self.n + self.internal_timer]
            gain += self.discount_factor ** self.n * self.q_table[chosen_traj_tuple['curr_state'], chosen_traj_tuple['curr_action']]
        curr_traj_tuple: TrajectoryTuple = self.trajectory[self.internal_timer]
        self.q_table[curr_traj_tuple['curr_state'], curr_traj_tuple['curr_action']] += self.learning_rate * (
                    gain - self.q_table[curr_traj_tuple['curr_state'], curr_traj_tuple['curr_action']])
