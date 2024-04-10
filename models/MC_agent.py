#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class RlAgent:
    """RL agent for the blackjack game"""

    def __init__(self):
        self.q = np.zeros((343, 9), dtype="float64") 
        self.n = np.zeros((343, 9), dtype="float64") 
        self.visited_states = []
        self.episode = {}
        self.stte_actn_pair = {}
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        self.turn = 0
        self.epsilon = 0.2
        # self.alpha = 0.1
        self.gamma = 1
        self.number_of_states = 343
        self.number_of_actions = 9

    def get_number_of_states(self):
        return self.number_of_states

    def get_number_of_actions(self):
        return self.number_of_actions

    def e_greedy(self, actions):
        a_star_idx = np.argmax(actions)
        rng = np.random.default_rng()
        if self.epsilon <= rng.random():
            return a_star_idx
        else:
            b = actions.size
            idx = rng.integers(low=0, high=b)
            return idx

    def select_action(self, state):
        self.turn += 1
        # print("Turn = ", self.turn)
        self.state = state
        # print("State = ", self.state)
        actions = self.q[state, ]
        action = self.e_greedy(actions)
        self.action = action
        return action

    def update_q(self):
        #get final state info
        f_stte = self.episode[self.turn-1][0]
        f_actn = self.episode[self.turn-1][1]
        f_rwd = self.episode[self.turn-1][2]
        G = f_rwd
        self.n[f_stte][f_actn] += 1
        n = self.n[f_stte][f_actn]
        self.q[f_stte][f_actn] += (1/n)*(G- self.q[f_stte][f_actn])
        
        #update visited states
        if (f_stte in self.visited_states) == False:
            self.visited_states.append(f_stte)
        
        #update G, backtracking from second to last state
        for i in range(self.turn-2,-1,-1):
            stte = self.episode[i][0]
            actn = self.episode[i][1]
            rwd = self.episode[i][2]
            G = self.gamma*G + rwd
            visits = self.stte_actn_pair[(stte, actn)]
            if visits > 1:
                self.stte_actn_pair[(stte, actn)] -= 1
                continue
            else:
                self.n[stte][actn] += 1
                n = self.n[stte][actn]
                self.q[stte][actn] += (1/n)*(G- self.q[stte][actn])
            
            #update visited states
            if (stte in self.visited_states) == False:
                self.visited_states.append(stte)
        
        # get # of visited states for current episode
        v_t = len(set(self.visited_states))
        return G, v_t
        
    def update_episode(self, current_state, action, reward):
        self.episode[self.turn-1] = [current_state, action, reward]
        if (current_state, action) in self.stte_actn_pair:
            self.stte_actn_pair[(current_state, action)] = self.stte_actn_pair[(current_state, action)] +1
        else:
            self.stte_actn_pair[(current_state, action)] = 1
            
    def new_episode(self):
        self.episode = {}
        self.stte_actn_pair = {}
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        self.turn = 0
        
    def reset(self):
        self.visited_states = []
        self.new_episode()
        self.q = np.zeros((343, 9), dtype="float64") 
        self.n = np.zeros((343, 9), dtype="float64") 
        
        
