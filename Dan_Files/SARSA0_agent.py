import numpy as np


class RlAgent:
    """RL SARSA-0 agent for the Tunnel Runner puzzle"""

    def __init__(self):
        self.q = np.zeros((343, 9), dtype="float64")
        self.state_actn_pairs = {}
        self.state = 0
        self.next_state = 0
        self.next_action = -1
        self.next_q = 0
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
        rng = np.random.default_rng()
        
        if self.epsilon <= rng.random():
            if b < 2:
                a_star_idx = np.argmax(actions)
                return a_star_idx
            else:
                idx = rng.integers(low=0, high=b)
                return a[idx]
        else:
            b = actions.size
            idx = rng.integers(low=0, high=b)
            return idx
    
    def get_state_actn_visits(self):
        sum1 = sum(list(self.state_actn_pairs.values()))
        return sum1
        
    def select_action(self, state):
        self.state = state
        if self.next_action < 0:
            actions = self.q[state, ]
            action = self.e_greedy(actions)
        else:
            action = self.next_action
        self.action = action
        if not((state,action) in self.state_actn_pairs):
            self.state_actn_pairs[(state,action)] = 1
        return action

    def set_future_action(self, new_state):
        actions = self.q[new_state, ]
        action = self.e_greedy(actions)
        self.next_q = self.q[new_state, action]
        self.next_action = action
        
        
    def update_q(self, new_state, reward):
        self.set_future_action(new_state)
        self.next_state = new_state
        q =  self.q[self.state, self.action]
        self.q[self.state,self.action]=q+self.alpha*(reward+self.gamma*self.next_q-q)
        f"Turn = {self.turn} \nQ = {self.q}"
        
    def reset(self):
        self.q = np.zeros((343, 9), dtype="float64")
        self.state_actn_pairs = {}
        self.state = 0
        self.next_state = 0
        self.next_action = -1
        self.next_q = 0
        self.reward = 0
        self.action = 0
        self.turn = 0

