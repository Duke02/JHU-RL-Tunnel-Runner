import random


def get_state_index(x, y, z):
    x_idx = x
    y_idx = 7 * y
    z_idx = 49 * z
    return x_idx + y_idx + z_idx  # Adjusted to match the provided formula


class TunnelRunner:
    """The Tunnel Runner game that cycle through levels."""

    def __init__(self):
        self.num_states = 343  # 49*7 [levels]
        self.num_actions = 4
        self.expl_x = 0  # Runner's x position from 0 to 7
        self.expl_y = 0  # Runner's y position from 0 to 7
        self.expl_z = 0  # Runner's z position from 0 to 6 (There are 7 levels, indexed from 0)
        self.win = {43, 92, 141, 190, 239, 288, 337}
        self.loss = {35, 83, 131, 179, 227, 275, 323, 36, 86, 136, 186, 236, 286, 336}
        self.walls = {12, 61, 110, 159, 208, 257, 306,
                      13, 62, 111, 160, 209, 258, 307,
                      14, 63, 112, 161, 210, 259, 308,
                      15, 64, 113, 162, 211, 260, 309,
                      16, 65, 114, 163, 212, 261, 310,
                      17, 66, 115, 164, 213, 262, 311,
                      24, 73, 122, 171, 220, 269, 318,
                      25, 74, 123, 172, 221, 270, 319,
                      26, 75, 124, 173, 222, 271, 320}

    def reset(self):
        x = 0
        y = 0
        z = 0
        self.expl_x = x
        self.expl_y = y
        self.expl_z = z
        st = get_state_index(self.expl_x, self.expl_y, self.expl_z)
        return st

    def execute_action(self, action):
        # Use the agent's action to determine the next state and reward #
        # Note: 'N' = 0, 'E' = 1, 'S' = 2, 'W' = 3 #

        current_state = get_state_index(self.expl_x, self.expl_y, self.expl_z)
        new_state = current_state
        reward = 0
        game_end = False

        # if in terminal states, stay in terminal states
        if (current_state in self.win) or (current_state in self.loss):
            new_state = current_state
            reward = 0
            game_end = True

        elif current_state in self.walls:
            new_state = current_state
            reward = -1000
            game_end = True

        else:
            temp_x = self.expl_x
            temp_y = self.expl_y
            temp_z = self.expl_z

            # determine a potential next state
            if action == 0:  # action is 'N'
                if temp_y == 0:
                    temp_y = 0
                else:
                    temp_y = temp_y - 1

            elif action == 1:  # action is 'E'
                if temp_x == 7:
                    temp_x = 7
                else:
                    temp_x = temp_x + 1

            elif action == 2:  # action is 'S'
                if temp_y == 7:
                    temp_y = 7
                else:
                    temp_y = temp_y + 1

            else:  # action is 'W'
                if temp_x == 0:
                    temp_x = 0
                else:
                    temp_x = temp_x - 1

            # recalculate the new state
            new_state = get_state_index(temp_x, temp_y, temp_z)



            if new_state in self.walls:
                temp_x = self.expl_x
                temp_y = self.expl_y
                temp_z = self.expl_z
                new_state = get_state_index(temp_x, temp_y, temp_z)
                reward = -1
                game_end = False

            elif new_state in self.loss:  # you lose
                reward = -30
                game_end = True

            elif new_state in self.win:  # you won
                reward = 30
                game_end = True

            else:
                reward = -1
                game_end = False

            self.expl_x = temp_x
            self.expl_y = temp_y
            self.expl_z = temp_z

        return new_state, reward, game_end




