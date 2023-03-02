import gym
import pandas as pd
import numpy as np
from numpy import random

class Env(gym.Env):
    def __init__(self):
        self.viewer = None
        path = './data/large.csv'
        self.data = pd.read_csv(path)
        self.min_weight = min(self.data['weight'])
        self.limit_weight = 0.5 * sum(self.data['weight'])
        self.state_dim = len(self.data)
        self.action_space = np.arange(len(self.data))
        self.n_actions = self.state_dim
        self.total_value = sum(self.data['value'])

    def reset(self):
        self.state = np.zeros(self.state_dim)
        return self.state

    def step(self, action):
        if self.state[action] == 1:
            weight_sum = sum(self.state)
            reward = -sum(self.data['value'])
            if weight_sum > self.limit_weight:
                done = True
            else:
                done = False
            next_state = self.state

        else:
            self.state[action] = self.data['weight'][action]
            weight_sum = sum(self.state)
            
            reward = 0
            if weight_sum > self.limit_weight:
                done = True
                reward += -sum(self.data['value'])
            else:
                done = False
                for i in range(self.state_dim):
                    if self.state[i] != 0:
                        reward += self.data['value'][i]
            
            next_state = self.state

        if weight_sum + self.min_weight > self.limit_weight:
            done = True

        return next_state, reward, done, weight_sum, {}


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# class Env(gym.Env):
#     def __init__(self):
#         self.viewer = None
#         path = './data/small.csv'
#         self.data = pd.read_csv(path)
#         self.min_weight = 2
#         self.limit_weight = 0.5 * sum(self.data['weight'])
#         self.state_dim = len(self.data)
#         self.action_space = np.arange(len(self.data))
#         self.n_actions = self.state_dim
#         self.total_value = sum(self.data['value'])

#     def reset(self):
#         self.state = np.zeros(self.state_dim)
#         return self.state

#     def step(self, action):
#         if self.state[action] == 1:
#             weight_sum = self.state[action] * self.data['weight'][action]
#             for i in range(self.state_dim):
#                 weight_sum += self.state[i] * self.data['weight'][i]
#             # weight_sum += (self.data['weight'] * self.state).sum()
#             if weight_sum > self.limit_weight:
#                 done = True
#                 r = -self.total_value
#             else:
#                 done = False
#                 r = -self.total_value
#             next_state = self.state
#         else:
#             state = self.state
#             state[action] = 1
#             weight_sum = 0
#             for i in range(self.state_dim):
#                 weight_sum += self.state[i] * self.data['weight'][i]
#             # weight_sum = (self.data['weight'] * self.state).sum()
#             if weight_sum > self.limit_weight:
#                 done = True
#                 r = -self.total_value
#             else:
#                 done = False
#                 r = 0
#                 for i in range(self.state_dim):
#                     r += self.state[i] * self.data['value'][i]
#                 # r = self.data['value'][action]/(self.data['weight'][action] * self.limit_weight)
#             next_state = state
#             self.state = next_state
#         if weight_sum + self.min_weight >= self.limit_weight:
#             done = True

#         return next_state, r, done, weight_sum, {}


#     def close(self):
#         if self.viewer:
#             self.viewer.close()
#             self.viewer = None