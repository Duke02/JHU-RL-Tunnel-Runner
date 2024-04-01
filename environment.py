import typing as tp
import numpy as np
from utils import DATA_DIR


SIDE_LENGTH: int = 7


def get_state(x: int, y: int, dimension: int) -> int:
    return x + SIDE_LENGTH * y + (SIDE_LENGTH ** 2) * dimension


class Environment:
    def __init__(self):
        # self.map: np.ndarray = np.loadtxt(DATA_DIR / 'map.txt').astype(int)
        # assert self.map.shape[0] == self.map.shape[1] == SIDE_LENGTH, f'Hi welcome to the Krusty Krab, we only accept maps with a side length of {SIDE_LENGTH}'
        # self.agent_location: np.ndarray = np.argwhere(self.map == 1)[0]

        self.agent_location: np.ndarray
        self.current_cannon_timestep: int = 0

        start_xy: tuple[int, int] = (self.side_length - 1, 0)
        wall_xy: list[tuple[int, int]] = ([(self.side_length - x, 1) for x in range(1, 4)] +
                                          [(x, 2) for x in range(0, 3)] +
                                          [(x + 2, 3) for x in range(0, 3)])
        goal_xy: tuple[int, int] = (0, self.side_length - 1)
        cannon_ball_xyd: list[tuple[int, int, int]] = ([(xd, self.side_length - 2, xd) for xd in range(self.side_length)] +
                                                       [(self.side_length - xd - 1, self.side_length - 3, xd) for xd in range(self.side_length)])
        self.lethal_states: set[int] = {get_state(x, y, d) for x, y, d in cannon_ball_xyd}
        self.terminal_states: set[int] = {get_state(goal_xy[0], goal_xy[1], d) for d in range(self.side_length)}.union(self.lethal_states)
        self.start_states: set[int] = {get_state(start_xy[0], start_xy[1], d) for d in range(self.side_length)}
        self.inaccessible_states: set[int] = {get_state(x, y, d) for x, y in wall_xy for d in range(self.side_length)}
        self.goal_states: set[int] = {get_state(goal_xy[0], goal_xy[1], d) for d in range(self.side_length)}
        # States that are not normal states.
        cool_states: set[int] = (self.lethal_states.union(self.terminal_states)
                                 .union(self.start_states)
                                 .union(self.inaccessible_states)
                                 .union(self.goal_states))
        # Just exhaustive search it baby~, what can go wrong?
        self.normal_states: set[int] = {s for x in range(self.side_length) for y in range(self.side_length) for d in range(self.side_length)
                                        if (s := get_state(x, y, d)) not in cool_states}
        self.all_states_holy_heck: set[int] = self.normal_states.union(cool_states)
        self.initial_agent_location: np.ndarray = np.array(list(start_xy))
        self.agent_location = self.initial_agent_location

        self.directions: list[tuple[int, int]] = [(x, y) for x in range(-1, 2) for y in range(-1, 2)]

    @property
    def number_of_states(self) -> int:
        return max(self.all_states_holy_heck.difference(self.inaccessible_states))

    def get_number_of_states(self) -> int:
        return self.number_of_states

    @property
    def number_of_possible_actions(self) -> int:
        return 9

    def get_number_of_actions(self) -> int:
        return self.number_of_possible_actions

    def get_terminal_states(self) -> set[int]:
        return self.terminal_states

    def get_state(self) -> int:
        return self.get_cell_state(self.agent_location[0].item(), self.agent_location[1].item())

    @property
    def side_length(self) -> int:
        return SIDE_LENGTH

    def get_dimension(self) -> int:
        return self.current_cannon_timestep

    def get_cell_state(self, x: int, y: int) -> int:
        return get_state(x, y, self.get_dimension())

    def is_accessible(self, x: int, y: int) -> bool:
        return self.get_cell_state(x, y) not in self.inaccessible_states

    def is_terminal(self, x: int, y: int) -> bool:
        return self.get_cell_state(x, y) in self.terminal_states

    def reset(self, seed: int = 13) -> int:
        np.random.seed(seed)
        self.agent_location = self.initial_agent_location
        self.current_cannon_timestep = 0
        out_initial_state: int = self.get_state()
        return out_initial_state

    def update_states(self):
        self.current_cannon_timestep = (self.current_cannon_timestep + 1) % self.side_length

    def apply_action_to_player_loc(self, action: int):
        new_location: np.ndarray = self.agent_location + np.array(list(self.directions[action]))
        new_x, new_y = new_location[0].item(), new_location[1].item()

        while new_x < 0:
            new_x += self.side_length
        while new_y < 0:
            new_y += self.side_length

        new_x %= self.side_length
        new_y %= self.side_length

        if self.is_accessible(new_x, new_y):
            self.agent_location = np.array([new_x, new_y])

    def execute_action(self, action: int):
        self.update_states()
        self.apply_action_to_player_loc(action)
        is_player_dead: bool = self.get_state() in self.lethal_states
        did_player_win: bool = self.get_state() in self.goal_states

        if is_player_dead:
            reward: float = -5
        elif did_player_win:
            reward: float = 5
        else:
            reward: float = -1

        return self.get_state(), reward, self.get_state() in self.terminal_states
