#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np



#open saved data
data = np.load('10_agent_data.npy',allow_pickle='TRUE').item()


###############################################################################
######################## EXTRACT ALL DATA #####################################


g_t = data[1][0] #sum returns over 10 agents per episode
wins = data[1][1][:,2] #sum of wins over 10 agents per episode
losses = data[1][1][:,0] #sum of losses over 10 aagents per episode
v_t = data[1][2] # sum of mean percentage of visited states

#sum all reutrns/wins/losses for each of the 10 agents
for i in range(2,len(data)+1):
    wins += data[i][1][:,2]
    losses += data[i][1][:,0]
    g_t += data[i][0]
    v_t += data[i][2]
    
    
    
###############################################################################
#CALCULATE CUMULATIVE RETURN (G_T)
#CALCULATE MEAN WINNING PERCENTAGE (W_T) 
#CALCULATE MEAN LOSSING PERCENTAGE (L_T)

# take sum of returns per state and find the average of cumulative returns over
# the 10 agents
G_t = []
# take sum of wins/losses and find the percentage of wins as the ratio
# of total wins/losses up to current episode over # of episodes played so far
# average the found percentage over the 10 agents
W_t = []
L_t = []
for i in range(2000):
    G_t.append(sum(g_t[:i+1])[0]/10)
    W_t.append(sum(wins[:i+1])/10/(i+1))
    L_t.append(sum(losses[:i+1])/10/(i+1))
    
# average the mean percentagge of visited stated by 10 agents
V_t = v_t/10
    

###############################################################################
######################## PLOT  ALL OF THE DATA ################################
x = np.arange(1, 2001,1)
plt.figure()
plt.plot(x,G_t)
plt.grid()
plt.title("Mean Cumulative Return Per Episode (G_t)")
plt.ylabel("Cumulative Return")
plt.xlabel("Episode #")

plt.figure()
plt.plot(x,W_t)
plt.grid()
plt.title("Mean Winning Percentage Per Episode (W_t)")
plt.ylabel("Percentage (%)")
plt.xlabel("Episode #")

plt.figure()
plt.plot(x,L_t)
plt.title("Mean Losing Percentage Per Episode (L_t)")
plt.ylabel("Percentage (%)")
plt.xlabel("Episode #")
plt.grid()


plt.figure()
plt.plot(x,V_t)
plt.grid()
plt.title("Mean Pecentage Visited States (V_t)")
plt.ylabel("% States Visited")
plt.xlabel("Episode #")
