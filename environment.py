import typing as tp
import numpy as np


class Environment:
    def __init__(self):
        pass

    @property
    def number_of_states(self) -> int:
        return 1    # TODO

    @property
    def number_of_possible_actions(self):
        return 9

