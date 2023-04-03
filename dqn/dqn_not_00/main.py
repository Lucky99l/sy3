import os
from unicodedata import name
import numpy as np
import random
import pickle

from collections import Counter, deque
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import set_seeds, sample, plot_trajs, plot_traj, save_data, gradient_cascade
from model import Model, NoisyDQN
from environment import SchedulerEnv


seeds = 999
set_seeds(seeds)

# Hyperparameters
batch_size = 32 # 16, 64
gamma = 0.1
K = 2

epsilon = 0.3
eps_decay = 0.995
eps_min = 0.001      # Minimal exploration rate (epsilon-greedy)

num_rounds = int(1e4)
# num_episodes = 500
learning_limit = 100
replay_limit = int(5e3)  # Number of steps until starting replay
# weight_update = 1000 # Number of steps until updating the target weights

# save path
name1 = 'sy2_60'
name2 = 'sy2_60_test'
path_dir = './main_c/'
path = path_dir + 'data/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = SchedulerEnv()

# start writing to tensorboard
# writer = SummaryWriter(comment="Scheduler DQN")

# create the current network and target network
# policy_model = Model(env.observation_space.shape, env.action_space.n).to(device)
policy_model = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
# policy_model.apply(weight_init)

# target_model = Model(env.observation_space.shape, env.action_space.n).to(device)
target_model = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
target_model.load_state_dict(policy_model.state_dict())

criterion = nn.MSELoss()
optimizer = optim.Adam(policy_model.parameters(), lr=1e-4)
# optim.lr_scheduler.StepLR(optimizer, 3, 0.6)

# Exploration rate
replay_buffer = deque(maxlen=5000)

loss_total = []
stopped_index_record = []
gamma_record = []

score_total = []
episode_reward_total = []
step_total = []

test_score_total = []
test_episode_reward_total = []
test_step_total = []

agent_pos_total = []
bad_choice_total = []
good_position_total = []

fre_count = 0

for i in range(num_rounds):
    # change this for while not true once it works
    state = env.reset()
    episode_reward = 0
    done = False
    step_idx = 0
    buffer = []

    action = -5
    count_stop = 0
    score_list = []

    agent_pos_record = []
    expect_pos_record = []
    bad_choice = []
    good_position = []

    # episode_state_topk = [state]
    # episode_reward_topk = []
    # episode_action_topk = []

    # dynamic gamma
    # increase initial gamma=0.1
    # if i <= 10:
    #     if gamma < 0.99:
    #         gamma = 1 - 0.98 * (1 - gamma)
    #     else:
    #         gamma = 0.99
    #     if i == 10:
    #         last_avg_step = sum(step_total[-10:]) / 10
    # else:
    #     avg_step = sum(step_total[-10:]) / 10
    #     if (avg_step >= 500) or ((avg_step - last_avg_step) > 50):
    #         if gamma < 0.99:
    #             gamma = 1 - 0.98 * (1 - gamma)
    #         else:
    #             gamma = 0.99
    #     else:
    #         if avg_step < last_avg_step:
    #             best_gamma = gamma
    #         gamma = best_gamma

    #     last_avg_step = avg_step

    if gamma < 0.99:
        gamma = 1 - 0.98 * (1 - gamma)
    else:
        gamma = 0.99

    gamma_record.append(gamma)

    # # decrease initial gamma=0.9
    # if gamma > 0.1:
    #     # gamma = 1 - 0.98 * (1 - gamma)
    #     gamma = 0.98 * gamma
    # else:
    #     gamma = 0.1

    # print('reset here')
    for j in range(1000):
        #    while not done:
        step_idx += 1
        # print(i,j,step_idx)

        # epsilon for epsilon greedy strategy
        if epsilon > eps_min:
            epsilon *= eps_decay

        old_action = action

        # Select and perform an action
        # # if step_idx > learning_limit:
        if ((np.random.rand()) < epsilon):
        # if np.random.rand() < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            # print(policy_model(torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device)).cpu())
            # action = torch.argmax(policy_model(torch.from_numpy(state).to(torch.float32).to(device)).cpu())
            action = torch.argmax(policy_model(torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device)).cpu()).item()
        
        # action = torch.argmax(policy_model(torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device)).cpu()).item()
        
        # print("episode:{} step:{} action:{}".format(i, j, action))
        next_state, reward, done, index, agent_pos, _ = env.step(action, old_action)

        agent_pos_record.append(agent_pos)
        # expect_pos_record.append(env.position)

        if reward < -0.2:
            count_stop += 1
            bad_choice.append(agent_pos)
        
        if reward > 0.:
            good_position.append(agent_pos)

        score_list.append(env.scores)
        episode_reward += reward
        reward = torch.tensor(reward)
        fre_count += 1
        replay_buffer.append((state, action, reward, next_state, done))

        # top K transition store
        # episode_state_topk.append(next_state)
        # episode_action_topk.append(action)
        # episode_reward_topk.append(reward)

        # print('here rewards', episode_reward, reward, step_idx)

        # Store other info in replay buffer
        
        # buffer.append((state, action, reward, next_state, done))
        # print('len buffer', len(replay_buffer))

        # once we're ready to learn then start learning with mini batches
        if (len(replay_buffer) == replay_limit) and (j % 5 == 0):
            # print('replay buffer')
            optimizer.zero_grad()

            # minibatch = random.sample(replay_buffer, batch_size)
            minibatch = sample(replay_buffer, batch_size)

            batch_state = torch.from_numpy(np.asarray([elem[0] for elem in minibatch])).to(torch.float32).to(device)
            batch_action = torch.from_numpy(np.asarray([elem[1] for elem in minibatch])).view(-1, 1).to(device)
            batch_reward = torch.from_numpy((np.asarray([elem[2] for elem in minibatch]))).view(-1, 1).to(device)
            batch_next_state = torch.from_numpy(np.asarray([elem[3] for elem in minibatch])).to(torch.float32).to(device)
            batch_done = torch.from_numpy(np.asarray([elem[4] for elem in minibatch])).to(torch.float32).view(-1, 1).to(device)

            q_eval = policy_model(batch_state).gather(1, batch_action)

            '''
            for state, action, reward, next_state, done in minibatch:
                # pass state to policy to get qval from policy
                pred_qval = max(policy_model(state))
            '''
            # pass next state to target policy to get next set of qvals (future gains)
            # if not done:
            target_value, _ = torch.max(target_model(batch_next_state), 1)
            target_value = target_value.view(-1, 1).detach()
            next_qval = batch_reward + gamma * target_value * (1 - batch_done)
            # next_qval = batch_reward + gamma * target_value
            # print("target: ", next_qval)
            # print("target_value: ", target_value)
            # print("reward: ", batch_reward)
            # print("q_eval: ", q_eval)
            q_eval = q_eval.to(torch.float32)
            # print('pred_qval', pred_qval, pred_qval.size())
            next_qval = next_qval.to(torch.float32)
            # print('next_qval', next_qval, next_qval.size())

            # # top K calculate process
            # obs = episode_state_topk[-1]
            # act = torch.argmax(policy_model(torch.from_numpy(obs).to(torch.float32).unsqueeze(0).to(device)).cpu()).item()
            # prob = target_model(torch.from_numpy(obs).to(torch.float32).unsqueeze(0).to(device)).cpu()
            # prob_act = prob[0][act]
            # # DQN does not need off-policy correction
            # # off_policy_correction = min(prob_act, 1)
            # tok_correction = gradient_cascade(prob_act, K)
            # # print("off_co:{}".format(off_policy_correction))
            # # print("tok_co:{}".format(tok_correction))
            # loss = criterion(q_eval, next_qval) * tok_correction

            loss = criterion(q_eval, next_qval)
            # loss = F.mse_loss(q_eval, next_qval)
            # print('loss: ', loss.item())
            # writer.add_scalar('loss', loss, i)

            loss.backward()

            optimizer.step()
            loss_total.append(loss.cpu().item())

            # Noisy DQN
            policy_model.reset_noise()
            target_model.reset_noise()

            # for name, parms in policy_model.named_parameters():
                # if parms.requires_grad:
                #     print(name)
                # print('-->grad_require: ', parms.requires_grad)
                # print('-->grad_value: ', parms.grad)

            # print("episode:{} loss:{:.4f}".format(i, loss.item()))
            # print('step', step_idx, 'i', i, 'j', j)

            # Update the target network, copying all weights and biases in DQN
            # Periodically update the target network by Q network to target Q network
        if fre_count >= 50:
            # print('update weights', step_idx)
            fre_count = 0
            # Update weights of target
            target_model.load_state_dict(policy_model.state_dict())

        # Move to the next state
        state = next_state

        if done:
            target_model.load_state_dict(policy_model.state_dict())
            break

        # if count_stop >= 10:
        #     break

    # if count_stop < 10:
    episode_reward_total.append(episode_reward)
    step_total.append(step_idx)
    stopped_index_record.append(index)
    score_list_ = []
    for t in range(len(score_list)):
        if score_list[t] >= 0:
            score_list_.append(score_list[t])

    score_total.append(sum(score_list_))

    agent_pos_total.append(agent_pos_record)
    bad_choice_total.append(bad_choice)
    good_position_total.append(good_position)

    # if episode_reward < 0:
    # if index < 50:
    if i % 100 == 0:
        # plot_trajs(env.num_gps, env.num_slots, agent_pos_record, expect_pos_record, bad_choice, good_position, index, sum(score_list_), i, path_dir + 'train_traj_record/' + name1)
        plot_traj(env.num_gps, env.num_slots, agent_pos_record, expect_pos_record, bad_choice, good_position, index, sum(score_list_), i, path_dir + 'train_traj_record/' + name1)

    if i % 100 == 0:
        print('episode: {}'.format(i))
        print('stopped_step: {}'.format(j))
        print('stopped_idx: {}'.format(index))
        print('episode_reward: {}'.format(episode_reward))
        print('score: {}'.format(sum(score_list_)))
        print('----------------------------\n')

    # if i % 100 == 0:
    #     policy_model.eval()
    #     state_ = env.reset()
    #     agent_pos_record = []
    #     bad_choice = []
    #     action_ = -5
    #     test_episode_reward = 0
    #     # test_count_stop = 0
    #     score = []
    #     book = env.to_book

    #     for k in range(1000):
    #         old_action_ = action_
    #         action_ = torch.argmax(policy_model(torch.from_numpy(state_).to(torch.float32).unsqueeze(0).to(device)).cpu()).item()
    #         next_state_, reward_, done_, index_, agent_pos_, _ = env.step(action_, old_action_)
    #         score.append(env.scores)
    #         state_ = next_state_

    #         # if reward_ < 0:
    #         #     test_count_stop += 1

    #         if ((old_action_ == action_) and ((agent_pos_[0] == 0) or (agent_pos_[1] == 0) or (agent_pos_[0] == 31) or (agent_pos_[1] == 99))) or (old_action_+action_)==5 or (old_action_+action_)==1:
    #             bad_choice.append(agent_pos_)

    #         test_episode_reward += reward_
    #         agent_pos_record.append(agent_pos_)

    #         if done_:
    #             break

    #     if i != 0:
    #         plot_trajs(test_episode_reward, env.num_gps, env.num_slots, agent_pos_record, bad_choice, i, './main_c/train_traj_record/' + name1)
    #     score_ = []
    #     for l in range(len(score)):
    #         if score[l] >= 0:
    #             score_.append(score[l])

    #     test_score_total.append(sum(score_))
    #     test_episode_reward_total.append(test_episode_reward)
    #     test_step_total.append(k)

    #     policy_model.train()
    #     print('episode: {}'.format(i))
    #     print('stopped_step: {}'.format(k))
    #     print('episode_reward: {}'.format(test_episode_reward))
    #     print('score: {}'.format(sum(score_)))
    #     print('----------------------------\n')


if not os.path.exists(path + name1):
    os.mkdir(path + name1)

# save data and model
save_data(score_total, path + name1 + '/score.pickle')
save_data(episode_reward_total, path + name1 + '/episode_reward.pickle')
save_data(step_total, path + name1 + '/step.pickle')
save_data(stopped_index_record, path + name1 + '/stopped_index.pickle')
save_data(gamma_record, path + name1 + '/gamma.pickle')
# save_data(test_score_total, path + name1 + '/test_score.pickle')
# save_data(test_episode_reward_total, path + name1 + '/test_episode_reward.pickle')
# save_data(test_step_total, path + name1 + '/test_step.pickle')
save_data(agent_pos_total, path + name1 + '/agent_position.pickle')
save_data(bad_choice_total, path + name1 + '/bad_choice.pickle')
save_data(good_position_total, path + name1 + '/good_choice.pickle')

torch.save(policy_model, path_dir + 'save_model/' + name1 + '_model.pkl')