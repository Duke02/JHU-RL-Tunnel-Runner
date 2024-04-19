import numpy as np


class RlAgent:
    """RL Q-Learning agent for the Tunnel Runner puzzle"""

    def __init__(self):
        self.q = np.zeros((343, 9), dtype="float64")
        self.state_actn_pairs = {}
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        self.turn = 0
        self.epsilon = 0.2
        self.alpha = 0.1
        self.gamma = 0.9
        self.number_of_states = 343
        self.number_of_actions = 9

    def get_number_of_states(self):
        return self.number_of_states

    def get_number_of_actions(self):
        return self.number_of_actions

    def e_greedy(self, actions):
        a=list(np.where(np.array(actions)==max(actions))[0])
        b=len(a)
        if b < 2:
            a_star_idx = np.argmax(actions)
            return a_star_idx
        else:
            rng = np.random.default_rng()
            idx = rng.integers(low=0, high=b)
            return a[idx]
    
    def get_state_actn_visits(self):
        sum1 = sum(list(self.state_actn_pairs.values()))
        return sum1
        
    def select_action(self, state):
        self.turn += 1
        self.state = state
        actions = self.q[state, ]
        action = self.e_greedy(actions)
        self.action = action
        if not((state,action) in self.state_actn_pairs):
            self.state_actn_pairs[(state,action)] = 1
            
        return action

    def update_q(self, new_state, reward):
        self.next_state = new_state
        q =  self.q[self.state, self.action]
        max_q = max(self.q[new_state, ])
        self.q[self.state,self.action]=q+self.alpha*(reward+self.gamma*max_q-q)
        f"Turn = {self.turn} \nQ = {self.q}"

    def reset(self):
        self.q = np.zeros((343, 9), dtype="float64")
        self.state_actn_pairs1 = {}
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        self.turn = 0
