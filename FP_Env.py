

import random


def get_state_index(x, y, z):
    x_idx = x
    y_idx = 7 * y + y
    z_idx = 49 * z
    return x_idx + y_idx + z_idx   # ranges from 0 to 342


class TunnelRunner:
    """The Tunnel Runner puzzle"""

    def __init__(self):

        self.num_states = 343
        self.num_actions = 9
        self.runr_x = 6 # explorer's x position from 0 to 6
        self.runr_y = 0  # explorer's y position from 0 to 6
        self.runr_z = 0  # explorer's z position from 0 to 6
        self.win = {42,91,140,189,238,287,336}
        self.loss = { 34, 35,
                      82, 85, 
                     130,135,
                     178,185, 
                     226,235,
                     274,285,
                     322,335}
        self.wall = { 11, 12, 13,  14, 15, 16,  23, 24, 25,
                       60, 61, 62,  63, 64, 65,  72, 73, 74,
                      109,110,111, 112,113,114, 121,122,123,
                      158,159,160, 161,162,163, 170,171,172,
                      207,208,209, 210,211,212, 219,220,221,
                      256,257,258, 259,260,261, 268,269,270,
                      305,306,307, 308,309,310, 317,318,319}

    # Get the key environment parameters
    def get_number_of_states(self):
        return self.num_states

    def get_number_of_actions(self):
        return self.num_actions

    # Get the state IDs that should not be set optimistically
    def get_terminal_states(self):
        term = self.win.union(self.loss, self.wall)
        return term

    def get_state(self):
        return get_state_index(self.runr_x, self.runr_y, self.runr_z)

    # Set the current state to the initial state
    def reset(self, runr_starts):
        x = 6
        y = 0
        z = 0
        if runr_starts:
            done = False
            while not done:
                x = random.randint(0, 6)
                y = random.randint(0, 6)
                z = random.randint(0, 6)
                st = get_state_index(x, y, z)
                if (st in self.win) or (st in self.loss) or (st in self.wall):
                    done = False
                else:
                    done = True
        self.runr_x = x
        self.runr_y = y
        self.runr_z = z
        st = get_state_index(self.runr_x, self.runr_y, self.runr_z)
        return st

    def execute_action(self, action):
        # Use the agent's action to determine the next state and reward #
        # Note: 'X' = 0, 'NW' = 1, 'N' = 2, 'NE' = 3 , 
        #       'W' = 4, 'E' = 5, 'SE' = 6, 'S' = 7, 'SW' = 8

        current_state = get_state_index(self.runr_x, self.runr_y, self.runr_z)
        new_state = current_state
        reward = 0
        game_end = False

        # if in terminal states, stay in terminal states
        if (current_state in self.win) or (current_state in self.loss):
            new_state = current_state
            reward = 0
            game_end = True

        elif (current_state in self.wall):
            new_state = current_state
            reward = -1
            game_end = True

        else:
            temp_x = self.runr_x
            temp_y = self.runr_y
            temp_z = self.runr_z

            # make sure potential move is within bounds
            # if action is 0 = 'X' (stays in same place), skip if-elif logic
            if action == 1: # action is 'NW'  
                temp_x = 0 if temp_x == 0 else (temp_x-1)
                temp_y = 0 if temp_y == 0 else (temp_y-1)
                    
            elif action == 2:  # action is 'N'
                temp_y = 0 if temp_y == 0 else (temp_y-1)
                    
            elif action == 3: # action is 'NE'  
                temp_x = 6 if temp_x == 6 else (temp_x+1)
                temp_y = 0 if temp_y == 0 else (temp_y-1)
                    
            elif action == 4:  # action is 'W'
                temp_x = 0 if temp_x == 0 else (temp_x-1)

            elif action == 5:  # action is 'E'
                temp_x = 6 if temp_x == 6 else (temp_x+1)
                
            elif action == 6: # action is 'SE'  
                temp_x = 6 if temp_x == 6 else (temp_x+1)
                temp_y = 6 if temp_y == 6 else (temp_y+1)
            
            elif action == 7: # action is 'S'  
                temp_y = 6 if temp_y == 6 else (temp_y+1)
            
            elif action == 6: # action is 'SW'  
                temp_x = 0 if temp_x == 0 else (temp_x-1)
                temp_y = 6 if temp_y == 6 else (temp_y+1)
            
            # recalculate the new state based on change of level
            new_state = get_state_index(temp_x, temp_y, temp_z)
            temp_z = 0 if new_state > 293 else temp_z+1
            new_state = get_state_index(temp_x, temp_y, temp_z)
           
            # check if new state is viable
            if new_state in self.wall: #if new state = wall, stay in prev state
                temp_x = self.runr_x
                temp_y = self.runr_y
                temp_z = self.runr_z
                new_state = get_state_index(temp_x, temp_y, temp_z)
                reward = -1
                game_end = False

            elif new_state in self.loss:      # you lose
                reward = -50
                game_end = True

            elif new_state in self.win:     # you won
                reward = 50
                game_end = True

            else:
                reward = -1
                game_end = False

            self.runr_x = temp_x
            self.runr_y = temp_y
            self.runr_z = temp_z

        return new_state, reward, game_end



