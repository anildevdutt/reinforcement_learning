import numpy as np
from random import random, choice
import matplotlib.pyplot as plt

class RL:
    def __init__(self):
        self.gamma = 0.8    # discount factor
        self.epsilon = 0.99 # randomness
        self.lr = 0.05       # learning rate
        self.decay = 0.00166    # decay factor of randomness
        self.episodes = 600 # number of episodes
        self.episode = 0
        self.return_vals = []
        self.trajectory = []

        self.max_steps = 100
        self.state_action_values = []

        self.environment = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 2],
        ]
        self.state = [4,0]
        self.max_steps = 20
        self.ini_state_action_values()


    # initilizing the Q table with 0s
    def ini_state_action_values(self):
        for i in range(len(self.environment)):
            row = []
            for j in range(len(self.environment[0])):
                row.append([0, 0, 0, 0])
            self.state_action_values.append(row)

    # trajectory => [state, action. reward, next_state, return]
    # this function updates the return value inplace
    # using formula Gt = r_t + gamma*(r_t+1) + gamma^2*(r_t+2) + gamme^3*(r_t+3) ... 
    def gt(self, trajectory):

        # calculating the return value of last item
        trajectory[len(trajectory)-1]["return"] = trajectory[len(trajectory)-1]["reward"]

        # iterating backwards, eg if len is 10 then i = 9, 8, 7 .. 0
        for i in range(len(trajectory)-2, -1, -1):
            trajectory[i]["return"] = trajectory[i]["reward"] + self.gamma * trajectory[i+1]["return"]



    # updating the state action values (Q Table) based on the return valuse from the trajectory
    # 0+(-1-0)*0.1(lr)
    def update_state_action_value(self, trajectory):
        for t in trajectory:
            self.state_action_values[t["state"][0]][t["state"][1]][t["action"]] += self.lr*(t["return"]-self.state_action_values[t["state"][0]][t["state"][1]][t["action"]])


    # returns next_states, reward, done
    def act(self, action):
        y = self.state[0]
        x = self.state[1]

        # up
        if action == 0:
            if y > 0 and self.environment[y-1][x] != 1:
                if self.environment[y-1][x] == 2:
                    return [y-1, x], 0, True
                else:
                    return [y-1, x], -1, False
            else:
                return [y, x], -1, False
        
        # right
        elif action == 1:
            if x < len(self.environment[0])-1 and self.environment[y][x+1] != 1:
                if self.environment[y][x+1] == 2:
                    return [y, x+1], 0, True
                else:
                    return [y, x+1], -1, False
            else:
                return [y, x], -1, False

        # down
        elif action == 2:
            if y < len(self.environment)-1 and self.environment[y+1][x] != 1:
                if self.environment[y+1][x] == 2:
                    return [y+1, x], 0, True
                else:
                    return [y+1, x], -1, False
            else:
                return [y, x], -1, False

        # left
        elif action == 3:
            if x > 0 and self.environment[y][x-1] != 1:
                if self.environment[y][x-1] == 2:
                    return [y, x-1], 0, True
                else:
                    return [y, x-1], -1, False
            else:
                return [y, x], -1, False


    def run_episode(self):  
        if self.episode >= self.episodes:      
            return False
        step = 0
        done = False
        self.state = [4,0]
        trajectory = []
        print("Episode:", self.episode, "epsilon:", self.epsilon)
        
        while step < self.max_steps  and not done:
            action = np.argmax(self.state_action_values[self.state[0]][self.state[1]]).item()
            if random() < self.epsilon:                    
                action = choice([0, 1, 2, 3])
            new_state, reward, done = self.act(action)
            trajectory.append({"state": self.state, "action": action, "next_state":new_state, "reward": reward, "return": 0})
            self.state = new_state
            step += 1
        self.epsilon -= self.decay
        self.gt(trajectory)
        self.update_state_action_value(trajectory)
        self.return_vals.append(trajectory[0]["return"])
        self.episode += 1
        self.trajectory = trajectory
        return True

if __name__ == "__main__":
    rl = RL()
    ok = rl.run_episode()    
    while ok:
        ok  = rl.run_episode()

    plt.plot(rl.return_vals)
    plt.show()
