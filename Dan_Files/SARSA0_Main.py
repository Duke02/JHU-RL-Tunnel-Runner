#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:17:18 2024

@author: boss
"""

import SARSA_TD0_agent as S
import FP_Env as env
import numpy as np


def main():
    environment = env.TunnelRunner()
    agent = S.RlAgent()
    # Check that the environment parameters match
    if (environment.get_number_of_states() == agent.get_number_of_states()) and \
            (environment.get_number_of_actions() == agent.get_number_of_actions()):
        # train 10 agents for 2000 episodes
        agent_knoweledge = {}
        agents = 1
        episodes = 10000
        for a in range(1,agents+1):
            print("AGENT",a)
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
                # agent.update_q(new_state, reward)
                v_t = agent.get_state_actn_visits()
                V_t[i,0] = (v_t/agent.get_number_of_states())*100
                #agent.new_episode()
            
            #save new agent data
            agent_knoweledge[a] = [Gs, stats, V_t]
        #save all agents data
        print("\nProgram completed successfully.")
        np.save('1_agent_10k_epsds_SARSA.npy', agent_knoweledge)
        np.save('1_agent_10k_epsds_SARSA_q.npy', agent.q)
    else:
        print("Environment and Agent parameters do not match. Terminating program.")


if __name__ == "__main__":
    main()
   
