import typing as tp
import numpy as np

State = int
Action = int
Reward = int


class DecisionStateObj(tp.TypedDict):
    player_point_total: int
    dealer_card_value: int
    usable_ace_present: bool


class TerminalStateObj(tp.TypedDict):
    win: bool
    tie: bool
    loss: bool


# Mostly here for debug purposes honestly.
# Helps me keep track of stuff.
class StateObj(tp.TypedDict):
    data: DecisionStateObj | TerminalStateObj
    is_terminal: bool


def is_decision_state(state: State) -> bool:
    return state < 200


def parse_state(state: State) -> StateObj:
    if is_decision_state(state):
        usable_ace_present: bool = (state // 100) > 0
        dealer_card_value: int = (state // 10) % 10 + 1
        player_point_total: int = (state % 10) + 12
        decision_state: DecisionStateObj = DecisionStateObj(player_point_total=player_point_total, dealer_card_value=dealer_card_value,
                                                            usable_ace_present=usable_ace_present)
        return StateObj(data=decision_state, is_terminal=False)
    else:
        wlt_digit: int = state % 10
        inner_state: TerminalStateObj = TerminalStateObj(win=wlt_digit == 3, tie=wlt_digit == 2, loss=wlt_digit == 1)
        return StateObj(data=inner_state, is_terminal=True)


def to_state_num(state: StateObj) -> State:
    if not state['is_terminal']:
        inner_state: DecisionStateObj = state['data']
        return (inner_state['player_point_total'] - 12) + (inner_state['dealer_card_value'] - 1) * 10 + int(inner_state['usable_ace_present']) * 100
    else:
        inner_state: TerminalStateObj = state['data']
        return 200 + int(inner_state['win']) * 3 + int(inner_state['tie']) * 2 + int(inner_state['loss'])


class RLAgent:
    def __init__(self, num_actions: int, num_states: int, epsilon: float, seed: int = 42, discount_factor: float = 0.9):
        self.qtable: np.ndarray = np.zeros((num_states + 1, num_actions))
        self.counts: np.ndarray = np.zeros((num_states, num_actions))
        self.epsilon: float = epsilon
        self.discount_factor: float = discount_factor
        self.trajectory: tp.List[tp.Tuple[StateObj, Action, Reward]] = []
        self.last_action_taken: Action | None = None
        self.last_state_seen: StateObj | None = None
        self.seed: int = seed
        self.num_wins: int = 0
        self.cumulative_gain: float = 0

        self.random_num_gen = np.random.default_rng(seed=self.seed)

        # Data to save for convergence rate
        self.last_qtable: np.ndarray | None = None
        self.convergence_rate: int = -1
        self.episodes_trained: int = 0
        self.convergence_tolerance: float = 1e-3

        # Data to save for sample efficiency
        self.cumulative_rewards: float = 0
        self.rewards_for_curr_episode: float = 0

        # Data to save for asymptotic performance
        self.rewards_since_convergence: float = 0

    @property
    def asymptotic_performance(self) -> float:
        return 0 if self.is_converged else self.rewards_since_convergence / (self.episodes_trained - self.convergence_rate)

    @property
    def is_converged(self) -> bool:
        return self.convergence_rate > 0

    @property
    def sample_efficiency(self):
        return self.cumulative_rewards

    @property
    def perc_visited_states(self) -> float:
        return np.any(self.counts > 0, axis=1).sum() / self.counts.shape[0]

    def reset(self):
        # self.qtable = np.zeros(self.qtable.shape)
        # self.counts = np.zeros(self.counts.shape)
        self.trajectory = []
        self.last_action_taken = None
        self.last_state_seen = None
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
        self.last_qtable: np.ndarray = self.qtable.copy()
        self.episodes_trained += 1
        # self.cumulative_gain: float = 0

    def get_actions_for_state(self, state: StateObj) -> np.ndarray:
        return self.qtable[to_state_num(state)]

    def get_best_action(self, state: StateObj) -> Action:
        return np.argmax(self.get_actions_for_state(state))

    def choose_action(self, state: StateObj) -> Action:
        self.random_num_gen = np.random.default_rng()
        if self.random_num_gen.random() > self.epsilon:
            return self.get_best_action(state)
        else:
            return np.random.randint(low=0, high=self.qtable.shape[1] - 1)

    def react_to_state(self, state: State, reward: float | int) -> Action:
        self.rewards_for_curr_episode += reward
        if self.last_action_taken is not None:
            # Add to the trajectory if we've managed to get a reward.
            sar_triple: tp.Tuple[StateObj, Action, Reward] = (self.last_state_seen, self.last_action_taken, reward)
            if sar_triple not in self.trajectory:
                # If we haven't seen this SAR triple yet, then we should add so that we can use it when we update our Q-Table.
                # We exclude ones we have seen in this episode because we're a first policy MC algorithm.
                self.trajectory.append(sar_triple)
        self.last_state_seen = parse_state(state)
        if self.last_state_seen['is_terminal']:
            self.num_wins += int(self.last_state_seen['data']['win'])
        self.last_action_taken = self.choose_action(self.last_state_seen)
        return self.last_action_taken

    def update_qtable(self):
        curr_gain: float = 0
        for state, action, reward in self.trajectory[::-1]:
            curr_gain = self.discount_factor * curr_gain + reward
            state_index: int = to_state_num(state)
            self.counts[state_index, action] += 1
            self.qtable[state_index, action] += (1 / self.counts[state_index, action]) * (curr_gain - self.qtable[state_index, action])
        self.cumulative_gain = curr_gain
