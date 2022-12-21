import os
import sys
import math
import numpy as np
from collections import Counter, deque

sys.path.append("/home/xliu/scheduler/sy3")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical

from utils import set_seeds, sample, save_data
from model import A3CNet
from environment import SchedulerEnv


seeds = 999
set_seeds(seeds)

# Hyperparameters
num_rounds = int(2e4)
gamma = 0.99

# save path
name1 = 'sy3_1'
name2 = 'sy3_1_test'
path = './a3c_result/data/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_process = 5

env =SchedulerEnv()
n_s = env.observation_space.shape[0]
n_a = env.action_space.n

# share optimizer
class SharedOptim(optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedOptim, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
    
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    # def step(self, closure=None):
    #     loss = None
    #     if closure is not None:
    #         loss = closure()

    #     for group in self.param_groups:
    #         for p in group['params']:
    #             if p.grad is None:
    #                 continue
    #             grad = p.grad.data
    #             state = self.state[p]

    #             exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
    #             beta1, beta2 = group['betas']

    #             state['step'] += 1

    #             if group['weight_decay'] != 0:
    #                 grad = grad.add(group['weight_decay'], p.data)

    #             # Decay the first and second moment running average coefficient
    #             exp_avg.mul_(beta1).add_(1 - beta1, grad)
    #             exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

    #             denom = exp_avg_sq.sqrt().add_(group['eps'])

    #             bias_correction1 = 1 - beta1**state['step'][0]
    #             bias_correction2 = 1 - beta2**state['step'][0]
    #             step_size = group['lr'] * math.sqrt(
    #                 bias_correction2) / bias_correction1

    #             p.data.addcdiv_(-step_size, exp_avg, denom)

    #     return loss

class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_ep, reward_record, step_record, score_record, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' %name
        self.global_ep = global_ep
        self.global_net = global_net
        self.local_net = A3CNet(n_s, n_a).to(device)

        if optimizer is None:
            self.optimizer = optim.Adam(global_net.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer

        self.score_record, self.reward_record, self.step_record = score_record, reward_record, step_record
        # self.lock, self.counter = lock, counter
        self.env = SchedulerEnv()

    def run(self):
        for episode in range(num_rounds):
            self.local_net.load_state_dict(self.global_net.state_dict())
            values, log_probs, rewards, masks = [], [], [], []
            state = self.env.reset()
            action = -5
            episode_reward = 0
            score_list = []

            for step in range(1000):
                old_action = action
                logit, value = self.local_net(torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device))
                
                prob = Categorical(F.softmax(logit, dim=-1))
                action = prob.sample()
                log_prob = prob.log_prob(action)

                next_state, reward, done, index, agent_pos, _ = self.env.step(action, old_action)

                episode_reward += reward
                score_list.append(self.env.scores)
                
                log_probs.append(log_prob)
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
                values.append(value)
                masks.append(torch.FloatTensor([1-done]).unsqueeze(1).to(device))

                # with self.lock:
                #     self.counter.value += 1

                state = next_state
                
                if done:
                    break

            self.reward_record.put(episode_reward)
            self.step_record.put(step)
            score_list_ = []
            for t in range(len(score_list)):
                if score_list[t] >= 0:
                    score_list_.append(score_list[t])

            self.score_record.put(sum(score_list_))

            _, next_v = self.local_net(torch.FloatTensor(torch.from_numpy(next_state).to(torch.float32).unsqueeze(0)).to(device))

            R = next_v
            returns = []
            for k in reversed(range(len(rewards))):
                R = rewards[k] + gamma * R * masks[k]
                returns.insert(0, R)

            Gt = torch.cat(returns).detach()
            values = torch.cat(values)
            log_probs = torch.cat(log_probs)

            At = Gt - values
            actor_loss  = -(log_probs * At.detach()).mean()
            actor_loss.requires_grad_(True)
            critic_loss = At.pow(2).mean()
            critic_loss.requires_grad_(True)

            self.loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            self.loss.backward()
            
            for l_p, g_p in zip(self.local_net.parameters(), self.global_net.parameters()):
                g_p._grad = l_p._grad
            
            self.optimizer.step()

            if int(self.global_ep.value) % 100 == 0:
                print(
                    self.name,
                    "episode: ", int(self.global_ep.value),
                    "episode reward: ", episode_reward,
                    "step: ", step,
                    "score: ", sum(score_list_)
                )
                print('--------------------------------------------------')

            with self.global_ep.get_lock():
                self.global_ep.value += 1

        self.score_record.put(None)
        self.reward_record.put(None)
        self.step_record.put(None)
        
# class Worker(mp.Process):
#     def __init__(self, global_net, opt, global_ep, global_ep_r, reward_queue, name):
#         super(Worker, self).__init__()
#         self.name = 'w%02i' %name
#         self.g_ep, self.g_ep_r, self.r_queue = global_ep, global_ep_r, reward_queue
#         self.global_net, self.opt = global_net, opt
#         self.local_net = A3CNet(n_s, n_a)                # local network
#         self.env = SchedulerEnv().unwrapped
#         self.step_total = []

#     def update(self, done, next_state, buffer_s, buffer_a, buffer_r, gamma=0.99):
#         if done:
#             next_v = 0.
#         else:
#             _, next_v = self.local_net.forward(torch.from_numpy(next_state).to(torch.float32).unsqueeze(0))
        
#         R = next_v
#         buffer_value = []
#         for step in reversed(range(len(buffer_r))):
#             R = buffer_r[step] + gamma * R
#             buffer_value.insert(0, R)
        
#         loss = self.local_net.loss_function(np.array(buffer_s), torch.from_numpy(np.array(buffer_a)), torch.cat(buffer_value).detach())

#         # calculate and push local parameters to global parameters
#         self.opt.zero_grad()
#         loss.backward()
#         for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
#             gp._grad = lp._grad
#         self.opt.step()

#         # pull global parameters
#         self.local_net.load_state_dict(self.global_net.state_dict())


#     def record(self, episode_reward, stop_step):
#         with self.g_ep.get_lock():
#             self.g_ep.value += 1
#         with self.g_ep_r.get_lock():
#             if self.g_ep_r.value == 0.:
#                 self.g_ep_r.value = episode_reward
#             else:
#                 self.g_ep_r.value = self.g_ep_r.value * 0.99 + episode_reward * 0.01

#         self.r_queue.put(self.g_ep_r.value)
#         # self.step_total.append(stop_step)

#         print(
#             self.name,
#             "Ep:", self.g_ep.value,
#             "| Ep_r: %.0f" % self.g_ep_r.value,
#         )
#         print('----------------------------')

#     def run(self):
#         while self.g_ep.value < num_rounds:
#             state = self.env.reset()
#             buffer_s, buffer_a, buffer_r = [], [], []
#             episode_reward = 0.
#             action = -5

#             for j in range(1000):
#                 old_action = action
#                 action = self.local_net.choose_action(state)
#                 next_state, reward, done, index, agent_pos, _ = self.env.step(action, old_action)

#                 episode_reward += reward
#                 buffer_s.append(state)
#                 buffer_a .append(action)
#                 buffer_r.append(torch.FloatTensor([reward]).unsqueeze(1))

#                 if j % 10 == 0 or done:
#                     self.update(done, next_state, buffer_s, buffer_a, buffer_r)
#                     buffer_s, buffer_a, buffer_r = [], [], []

#                     if done or (j == 1000):
#                         self.record(episode_reward, j)
#                         break

#                 state = next_state
        
#         self.r_queue.put(None)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    global_net = A3CNet(n_s, n_a).to(device)                        # global network
    global_net.share_memory()

    opt = SharedOptim(global_net.parameters(), lr=1e-4)
    opt.share_memory()

    episode_rewards, steps, scores = mp.Queue(), mp.Queue(), mp.Queue()

    # global_ep, global_ep_r, reward_queue, step_queue, score_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue(), mp.Queue(), mp.Queue()

    # parallel training
    global_ep = mp.Value('d', 0)
    # counter = mp.Value('i', 0)
    # lock = mp.RLock()

    workers = [Worker(global_net, opt, global_ep, episode_rewards, steps, scores, j) for j in range(num_process)]
    [w.start() for w in workers]

    episode_reward_total, step_total, score_total = [], [], []

    while True:
        # if episode_rewards.empty() == False:
        r, st, s = episode_rewards.get(), steps.get(), scores.get()
        if r is not None:
            episode_reward_total.append(r)
            step_total.append(st)
            score_total.append(s)
        else:
            break

    [w.join() for w in workers]

    if not os.path.exists(path + name1):
        os.mkdir(path + name1)

    torch.save(global_net, './a3c_result/save_model/' + name1 + '_model.pkl')
    save_data(score_total, path + name1 + '/score.pickle')
    save_data(episode_reward_total, path + name1 + '/episode_reward.pickle')
    save_data(step_total, path + name1 + '/step.pickle')
