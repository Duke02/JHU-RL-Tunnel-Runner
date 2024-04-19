#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:50:41 2024

@author: boss
"""

#optima path if agent does not use the 'stay put' move
#x,y= 6,0 -> 5,0 -> 4,0 -> 3,1 -> 4,2 -> 5,3 -> 4,4 -> 3,5 -> 2,6 -> 1,6 -> 0,6
#    START   W(4)   W(4)   SW(8  SE(6)  SE(6)  SW(8)   SW(8)  SW(8)  W(4)   W(4)
# number of moves = 10

import FP_Env as env
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

#open saved data
q = np.load('1_agent_10k_epsds_MC_q.npy',allow_pickle='TRUE')

def plotResult(z, y, x):
    level = z+1
    y=y+1
    x=x+1
    fig = plt.figure()
    ax = fig.add_subplot(111,xlim=(0,7),ylim=(-7,0))
    
    #Walls
    ax.add_patch(Polygon([(4,-1), (7,-1), (7,-2), (4,-2)],color='k'))
    ax.add_patch(Polygon([(0,-2), (3,-2), (3,-3), (0,-3)],color='k'))
    ax.add_patch(Polygon([(2,-3), (5,-3), (5,-4), (2,-4)],color='k'))
    
    #PathBlocker/CannonBall/MovingObstacle
    ax.add_patch(Polygon([(level-1,-5), (level,-5), (level,-6), (level-1,-6)],color='r'))
    ax.add_patch(Polygon([(7-level,-4), (8-level,-4), (8-level,-5), (7-level,-5)],color='r'))
    
    #agent
    ax.add_patch(Polygon([(x,1-y), (x-1,1-y), (x-1,-y), (x,-y)],color='c'))
    
    ax.set_aspect('equal')
    ax.grid()
    plt.show()
    

#instantiatet env
environment = env.TunnelRunner()

#print agent optimal path
state =  environment.reset(False)
[x,y,z] = environment.get_xyz()
plotResult(z, y, x)
print("Start",state)
while not(state in environment.win):
    action = q[state,]
    a_idx = np.where(action == max(action))[0][0]
    print("Move: ", a_idx)
    state, rwd, end = environment.execute_action(a_idx)
    [x,y,z] = environment.get_xyz()
    print("State: ",state)
    plotResult(z, y, x)