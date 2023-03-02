import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from torch import optim

from model import Model
from environment import Env
from utils import save_data, set_seeds

seeds = 999
set_seeds(seeds)

name = 'knapsack_5'
path = './dqn_result/result/'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Agent:
    def __init__(self, env, state_n_dim, n_actions, gamma=0.5, epsilon=0.5):
        self.env = env
        self.n_actions = n_actions
        self.state_n_dim = state_n_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = 0.05
        self.eps_decay = 0.9
        self.iter_count = 0

        self.policy_model = Model(self.state_n_dim, self.n_actions).to(device)
        self.target_model = Model(self.state_n_dim, self.n_actions).to(device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=1e-3)

        self.replay_memory = deque(maxlen=10000)
        self.min_replay_memory_size = 100
        self.batch_size = 64
        self.update_target = 10
        self.scores = []
        self.steps = []
        self.weights = []


    def train(self):
        if len(self.replay_memory) < self.batch_size:
            return
        self.optimizer.zero_grad()
        minibatch = random.sample(self.replay_memory, self.batch_size)

        batch_state = torch.from_numpy(np.asarray([elem[0] for elem in minibatch])).to(torch.float32).to(device)
        batch_action = torch.from_numpy(np.asarray([elem[1] for elem in minibatch])).view(-1, 1).to(device)
        batch_reward = torch.from_numpy(np.asarray([elem[2] for elem in minibatch])).view(-1, 1).to(device)
        batch_next_state = torch.from_numpy(np.asarray([elem[3] for elem in minibatch])).to(torch.float32).to(device)
        batch_done = torch.from_numpy(np.asarray([elem[4] for elem in minibatch])).to(torch.float32).view(-1, 1).to(device)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # q_eval = self.policy_model(batch_state)[batch_index, batch_action]
        q_eval = self.policy_model(batch_state).gather(1, batch_action)
        target_value, _ = torch.max(self.target_model(batch_next_state), 1)
        target_value = target_value.view(-1, 1)
        # print(target_value)
        next_qval = batch_reward + self.gamma * target_value * (1 - batch_done)

        # target_value = self.target_model(batch_next_state)
        # # target_value[batch_done] = 0.0
        # next_qval = batch_reward + self.gamma * torch.max(target_value, dim=1)[0]

        q_eval = q_eval.to(torch.float32)
        next_qval = next_qval.to(torch.float32)

        loss = self.criterion(q_eval, next_qval)
        loss.backward()
        self.optimizer.step()
        self.iter_count += 1
        if not self.iter_count % self.update_target:
            self.target_model.load_state_dict(self.policy_model.state_dict())

    def step(self):
        done = False
        state = self.env.reset()
        episode_reward = 0
        reward = 0
        step = 0
        count_stop = 0
        num_step = 1000
        # for i in range(num_step):
        while not done:
            step += 1
            action_selected = set(np.where(state == 1)[0])
            if np.random.random() > self.epsilon and reward >= 0:
                action = torch.argmax(self.policy_model(torch.from_numpy(np.asarray(state)).to(torch.float32).to(device)).cpu(), dim=0).item()
            else:
                action_list = list(set(self.env.action_space).difference(action_selected))
                action = np.random.choice(action_list)

            next_state, reward, done, w, _ = self.env.step(action)

            if reward > 0:
                episode_reward = reward
            else:
                count_stop += 1
            
            self.replay_memory.append((state, action, reward, next_state, done))
            self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.eps_min else self.eps_min
            state = next_state

            # if done or count_stop > 2:
            if done:
                break

            # if count_stop > 30:
            #     break

        self.scores.append(episode_reward)
        self.steps.append(step)
        self.weights.append(w)
        # print(state)

    def test(self):
        done = False
        state = self.env.reset()
        reward = 0
        step = 0
        action = -1
        num_step = 100
        for i in range(num_step):
            step += 1
            old_action = action
            action_selected = set(np.where(state == 1)[0])
            action = torch.argmax(self.policy_model(torch.tensor([state], dtype=torch.float32).to(device)).cpu()).item()
            if action == old_action:
                action_list = list(set(self.env.action_space).difference(action_selected))
                action = np.random.choice(action_list)

            next_state, reward, done, w, _ = self.env.step(action)
            state = next_state
            if done:
                break

        weight = (self.env.data['weight'] * state).sum()
        value = (self.env.data['value'] * state).sum()

        print('test:')
        print(self.env.state)
        print("step: {} weight: {} value: {}".format(step, weight, value))


    def save_(self):
        if not os.path.exists(path + name):
            os.mkdir(path + name)

        torch.save(self.policy_model, './dqn_result/save_model/' + name + '_model.pkl')
        save_data(self.scores, path + name + '/episode.pickle')
        save_data(self.steps, path + name + '/steps.pickle')


if __name__ == '__main__':
    env = Env()
    n_actions = env.n_actions
    state_n_dims = env.state_dim
    agent = Agent(env, n_actions, state_n_dims)
    episodes = int(2e3)
    for episode in range(episodes):
        agent.step()
        agent.train()
        print('Episode: {} reward: {} weights: {} step: {}'.format(episode, agent.scores[episode], agent.weights[episode], agent.steps[episode]))

    # agent.test()
    agent.save_()
    agent.env.close()

