import typing as tp
import numpy as np
from utils import DATA_DIR


def get_state(x: int, y: int, dimension: int) -> int:
    return x + 7 * y + 49 * dimension


class Environment:
    def __init__(self):
        self.map: np.ndarray = np.loadtxt(DATA_DIR / 'map.txt').astype(int)
        self.agent_location: np.ndarray = np.argwhere(self.map == 1)

    @property
    def number_of_states(self):
        return self.map.shape[0] ** 2

    @property
    def number_of_possible_actions(self):
        return 9

    @property
    def side_length(self) -> int:
        return self.map.shape[0]

    def get_dimension(self) -> int:
        return int(np.argwhere(self.map[-2, :] == 4).item())

    def get_agent_state(self) -> int:
        x, y = self.agent_location
        coeff: int = self.side_length
        return x + coeff * y + (coeff ** 2) * self.get_dimension()

    def get_cell(self, x: int, y: int) -> int:
        return int(self.map[y, x])

    def is_accessible(self, x: int, y: int) -> bool:
        return self.get_cell(x, y) in {0, 1, 3, 4, 5}

    def is_terminal(self, x: int, y: int) -> bool:
        return self.get_cell(x, y) in {3, 4, 5}

    def reset(self, seed: int = 13) -> int:
        np.random.seed(seed)
        self.agent_location = np.argwhere(self.map == 1)
        cannon1_location: np.ndarray = np.argwhere(self.map == 3)
        cannon2_location: np.ndarray = np.argwhere(self.map == 4)
        self.map[cannon2_location] = 0
        self.map[cannon1_location] = 0
        self.map[cannon1_location[0], -1] = 3
        self.map[cannon2_location[0], 0] = 4
        return self.get_agent_state()

    def get_new_cannon_location(self, cannon_type: int):
        assert cannon_type in {3, 4}, 'Must have the correct cannon type.'
        curr_location: np.ndarray = np.argwhere(self.map == cannon_type)
        direction: int = -1 if cannon_type == 3 else 1
        curr_location[1] += direction
        if cannon_type == 3 and curr_location[1] < 0:
            curr_location[1] = self.side_length - 1
        elif cannon_type == 4 and curr_location[1] >= self.side_length:
            curr_location[1] = 0
        return curr_location

    def update_states(self):
        new_cannon1_location: np.ndarray = self.get_new_cannon_location(3)
        self.map[new_cannon1_location] = 3
        new_cannon2_location: np.ndarray = self.get_new_cannon_location(4)
        self.map[new_cannon2_location] = 4

    def execute_action(self, action: int):
        self.update_states()
        # TODO: Update agent location based on action
        #   and then check if agent is in the same location as the cannon balls.
        #   then return the current state for the agent's location.
