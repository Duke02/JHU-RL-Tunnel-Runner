#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:56:05 2024

@author: boss
"""

#optima path
#x,y= 6,0 -> 5,0 -> 4,0 -> 3,1 -> 4,2 -> 5,3 -> 4,4 -> 3,5 -> 2,6 -> 1,6 -> 0,6
#    START   W(4)   W(4)   SW(8  SE(6)  SE(6)  SW(8)   SW(8)  SW(8)  W(4)   W(4)
# number of moves = 11
import QLearning_agent as QL
import FP_Env as env
import numpy as np


def main():
    environment = env.TunnelRunner()
    agent = QL.RlAgent()
    # Check that the environment parameters match
    if (environment.get_number_of_states() == agent.get_number_of_states()) and \
            (environment.get_number_of_actions() == agent.get_number_of_actions()):
        # train 10 agents for 2000 episodes
        agent_knoweledge = {}
        agents = 1
        episodes = 10000
        for a in range(1,agents+1):
            print("Agent ",a)
            agent.reset()
            stats = np.zeros((episodes, 3)) # lost, tied, won games per episode
            Gs = np.zeros((episodes,1)) #cumulative returns
            V_t = np.zeros((episodes,1)) # percent states visited per episode
            for i in range(episodes):
                # reset the game and observe the current state
                current_state = environment.reset(False)
                game_end = False
                # Do until the game ends:
                while not game_end:
                    action = agent.select_action(current_state)
                    new_state, reward, game_end = environment.execute_action(action)
                    #agent.update_episode(current_state, action, reward)
                    current_state = new_state
                    agent.update_q(new_state, reward)
                if (current_state in environment.loss):
                    j = 0
                elif (current_state in environment.win):
                    j = 2
                stats[i][j] = 1
                #Gs[i,0], v_t = agent.update_q(new_state, reward)
                
                v_t = agent.get_state_actn_visits()
                V_t[i,0] = (v_t/agent.get_number_of_states())*100
                #agent.new_episode()
            
            #save new agent data
            agent_knoweledge[a] = [Gs, stats, V_t]
        
        #save all agents data
        print("\nProgram completed successfully.")
        np.save('1_agent_10k_epsds_QL.npy', agent_knoweledge)
        np.save('1_agent_10k_epsds_QL_q.npy', agent.q)
    else:
        print("Environment and Agent parameters do not match. Terminating program.")


if __name__ == "__main__":
    main()
   
