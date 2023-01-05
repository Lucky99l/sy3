import gym
import pandas as pd
import numpy as np
from numpy import random


class Env(gym.Env):
    def __init__(self):
        self.viewer = None
        # self.files = [(1,1),(2,6),(5,18),(6,22),(7,28)]
        # self.files = [(2,3),(3,4),(4,5),(5,6),(4,3),(7,12),(3,3),(2,2)]
        path = './data/large.csv'
        self.data = pd.read_csv(path)
        self.min_weight = 2
        self.limit_weight = 0.1 * sum(self.data['weight'])
        self.state_dim = len(self.data)
        self.action_space = np.arange(len(self.data))
        self.n_actions = self.state_dim
        # self.repet = 0

    def reset(self):
        self.state = np.zeros(self.state_dim)
        # self.state = np.array([0,0,0,0,0,0,0,0])
        return self.state

    def step(self, action):
        if self.state[action].all() == 1:
            weight_sum = self.state[action] * self.data['weight'][action]
            weight_sum += (self.data['weight'] * self.state).sum()
            # for i in range(self.state_dim):
            #     weight_sum += self.state[i] * self.data['weight'][i]
            if weight_sum > self.limit_weight:
                # done = True
                r = -100
                # self.repet = 1
            else:
                done = False
                r = -100
            next_state = self.state
        else:
            state = self.state
            state[action] = 1
            weight_sum = (self.data['weight'] * self.state).sum()
            # for i in range(self.state_dim):
            #     weight_sum += self.state[i] * self.data['weight'][i]
            if weight_sum > self.limit_weight:
                # done = True
                r = -100
            else:
                done = False
                r = 0
                for i in range(self.state_dim):
                    r += self.state[i] * self.data['value'][i]
            next_state = state
            # self.state = next_state
        if weight_sum + self.min_weight >= self.limit_weight:
            done = True

        return next_state, r, done, weight_sum, {}


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None