import os
import sys
import numpy as np
import random

from collections import Counter, deque
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

sys.path.append("/home/xliu/scheduler/sy3")

from utils import set_seeds, sample, plot_trajs, save_data
from model import Actor, Critic
from environment import SchedulerEnv


seeds = 999
set_seeds(seeds)

# Hyperparameters
num_rounds = int(3e4)

# save path
name1 = 'sy3_6'
name2 = 'sy3_6_test'
path = './a2c_result/result/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = SchedulerEnv()

class A2C:
    def __init__(self):
        # create the current network and value network
        self.policy_model = Actor(env.action_space.n).to(device)
        # policy_model.apply(weight_init)
        self.value_model = Critic().to(device)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=1e-4)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, 3, 0.1)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=1e-4)

    def choose_action(self, state):
        prob = Categorical(self.policy_model(torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device)))
        action = prob.sample()
        return prob, action

    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns
    
    def learn(self, next_value, log_probs, rewards, values, masks):
        Gt = self.compute_returns(next_value, rewards, masks)

        Gt = torch.cat(Gt).detach()
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)

        At = Gt - values
        actor_loss  = -(log_probs * At.detach()).mean()
        actor_loss.requires_grad_(True)
        critic_loss = At.pow(2).mean()
        critic_loss.requires_grad_(True)

        self.policy_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step()


    def save(self):
        torch.save(self.policy_model, './a2c_result/save_model/' + name1 + '_model.pkl')
        save_data(score_total, path + name1 + '/test_score.pickle')
        save_data(episode_reward_total, path + name1 + '/test_episode_reward.pickle')
        save_data(step_total, path + name1 + '/test_step.pickle')


step_total = []
score_total = []
episode_reward_total = []

test_score_total = []
test_episode_reward_total = []
test_step_total = []

a2c = A2C()
for i in range(num_rounds):
    # change this for while not true once it works
    state = env.reset()
    episode_reward = 0
    action = -5
    score_list = []
    count_stop = 0
    rewards = []
    log_probs = []
    values = []
    masks = []

    count_stop = 0

    for j in range(1000):
        old_action = action
        probs, action = a2c.choose_action(state)
        log_prob = probs.log_prob(action)
        value = a2c.value_model(torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device))

        next_state, reward, done, index, agent_pos, _ = env.step(action, old_action)

        score_list.append(env.scores)
        episode_reward += reward

        # if reward < 0:
        #     count_stop += 1

        log_probs.append(log_prob)
        rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
        values.append(value)
        masks.append(torch.FloatTensor([1-done]).unsqueeze(1).to(device))

        # Move to the next state
        state = next_state

        # if done or (count_stop >= 10):
        if done:
            break

    next_v = a2c.value_model(torch.FloatTensor(torch.from_numpy(next_state).to(torch.float32).unsqueeze(0)).to(device))
    a2c.learn(next_v, log_probs, rewards, values, masks)

    # if count_stop < 10:
    episode_reward_total.append(episode_reward)
    step_total.append(j)
    score_list_ = []
    for t in range(len(score_list)):
        if score_list[t] >= 0:
            score_list_.append(score_list[t])

    score_total.append(sum(score_list_))

    # test
    if i % 100 == 0:
        test_score_list = []
        test_log_probs = []
        test_rewards = []
        test_values = []

        test_episode_reward = 0
        test_action = -5

        test_state = env.reset()
        for k in range(1000):
            test_old_action = test_action
            test_probs, test_action = a2c.choose_action(test_state)
            test_log_prob = test_probs.log_prob(test_action)
            test_value = a2c.value_model(torch.from_numpy(test_state).to(torch.float32).unsqueeze(0).to(device)).cpu().item()

            test_next_state, test_reward, test_done, test_index, test_agent_pos, _ = env.step(test_action, test_old_action)

            test_score_list.append(env.scores)
            test_episode_reward += test_reward
            test_reward = torch.tensor(test_reward)

            test_log_probs.append(test_log_prob)
            test_rewards.append(test_reward)
            test_values.append(test_value)

            # Move to the next state
            test_state = test_next_state

            if test_done:
                break

        test_score_list_ = []
        for t in range(len(test_score_list)):
            if test_score_list[t] >= 0:
                test_score_list_.append(test_score_list[t])

        test_score_total.append(sum(test_score_list_))
        test_episode_reward_total.append(test_episode_reward)
        test_step_total.append(k)

        print('episode: {}'.format(i))
        print('stopped_step: {}'.format(k))
        print('episode_reward: {}'.format(test_episode_reward))
        print('score: {}'.format(sum(test_score_list_)))
        print('----------------------------\n')

# save data
if not os.path.exists(path + name1):
    os.mkdir(path + name1)

save_data(score_total, path + name1 + '/score.pickle')
save_data(episode_reward_total, path + name1 + '/episode_reward.pickle')
save_data(step_total, path + name1 + '/step.pickle')

a2c.save()
