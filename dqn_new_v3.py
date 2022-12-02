import os
import numpy as np
import random

from collections import Counter, deque
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from utils import set_seeds, sample, plot_trajs, save_data
from model import Model
from environment import SchedulerEnv


seeds = 999
set_seeds(seeds)


# Hyperparameters
batch_size = 32 # 16, 64
gamma = 0.5

epsilon = 0.3
eps_decay = 0.9
eps_min = 0.001      # Minimal exploration rate (epsilon-greedy)

num_rounds = int(2e4)
# num_episodes = 500
learning_limit = 100
replay_limit = 1000  # Number of steps until starting replay
# weight_update = 1000 # Number of steps until updating the target weights

# save path
name1 = 'sy3_14'
name2 = 'sy3_14_test'
path = './data/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = SchedulerEnv()

# start writing to tensorboard
# writer = SummaryWriter(comment="Scheduler DQN")

# create the current network and target network
policy_model = Model(env.observation_space.shape, env.action_space.n).to(device)
# policy_model.apply(weight_init)

target_model = Model(env.observation_space.shape, env.action_space.n).to(device)
target_model.load_state_dict(policy_model.state_dict())

criterion = nn.MSELoss()
optimizer = optim.Adam(policy_model.parameters(), lr=1e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, 3, 0.1)

# Exploration rate
replay_buffer = deque(maxlen=1000)

loss_total = []

score_total = []
episode_reward_total = []
step_total = []

test_score_total = []
test_episode_reward_total = []
test_step_total = []

fre_count = 0

for i in range(num_rounds):
    # change this for while not true once it works
    state = env.reset()
    episode_reward = 0
    done = False
    step_idx = 0
    buffer = []
    action = -5
    score_list = []
    count_stop = 0

    # epsilon for epsilon greedy strategy
    # if epsilon > eps_min:
    #     epsilon *= eps_decay

    # print('reset here')
    for j in range(1000):
        #    while not done:
        step_idx += 1
        # print(i,j,step_idx)
        old_action = action

        # Select and perform an action
        # if step_idx > learning_limit:
        if ((np.random.rand()) < epsilon):
        # if np.random.rand() < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            # print(policy_model(torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device)).cpu())
            # action = torch.argmax(policy_model(torch.from_numpy(state).to(torch.float32).to(device)).cpu())
            action = torch.argmax(policy_model(torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device)).cpu(), dim=1).item()
        
        # print("episode:{} step:{} action:{}".format(i, j, action))
        next_state, reward, done, index, agent_pos, _ = env.step(action, old_action)
        if reward < 0:
            count_stop += 1

        score_list.append(env.scores)
        episode_reward += reward
        reward = torch.tensor(reward)
        fre_count += 1
        # buffer.append((state, action, reward, next_state, done))
        replay_buffer.append((state, action, reward, next_state, done))

        # print('here rewards', episode_reward, reward, step_idx)

        # Store other info in replay buffer
        
        # buffer.append((state, action, reward, next_state, done))
        # print('len buffer', len(replay_buffer))

        # once we're ready to learn then start learning with mini batches
        if (len(replay_buffer) == replay_limit) and (j % 10 == 0):
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
            target_value = target_value.view(-1, 1)
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

            loss = criterion(q_eval, next_qval)
            # loss = F.mse_loss(q_eval, next_qval)
            # print('loss: ', loss.item()) 
            # writer.add_scalar('loss', loss, i)

            loss.backward()

            optimizer.step()
            # scheduler.step()
            loss_total.append(loss.cpu().item())
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
        
        if count_stop >= 10:
            # step_idx = 999
            break

    if count_stop < 10:
        episode_reward_total.append(episode_reward)
        step_total.append(step_idx)
        score_list_ = []
        for t in range(len(score_list)):
            if score_list[t] >= 0:
                score_list_.append(score_list[t])

        score_total.append(sum(score_list_))

    # test
    if i % 100 == 0:
        policy_model.eval()
        # score, test_episode_reward, stopped_step = test(policy_model, device)
        state_ = env.reset()
        agent_pos_record = []
        bad_choice = []
        action_ = -5
        test_episode_reward = 0
        # test_count_stop = 0
        score = []
        book = env.to_book

        for k in range(1000):
            old_action_ = action_
            action_ = torch.argmax(policy_model(torch.from_numpy(state_).to(torch.float32).unsqueeze(0).to(device)).cpu()).item()
            next_state_, reward_, done_, index_, agent_pos_, _ = env.step(action_, old_action_)
            score.append(env.scores)
            state_ = next_state_

            # if reward_ < 0:
            #     test_count_stop += 1

            if ((old_action_ == action_) and ((agent_pos_[0] == 0) or (agent_pos_[1] == 0) or (agent_pos_[0] == 31) or (agent_pos_[1] == 99))) or (old_action_+action_)==5 or (old_action_+action_)==1:
                bad_choice.append(agent_pos_)

            test_episode_reward += reward_
            agent_pos_record.append(agent_pos_)

            if done_:
                break

        if i != 0:
            plot_trajs(test_episode_reward, env.num_gps, env.num_slots, agent_pos_record, bad_choice, i, './train_traj_record/' + name1)
        score_ = []
        for l in range(len(score)):
            if score[l] >= 0:
                score_.append(score[l])

        test_score_total.append(sum(score_))
        test_episode_reward_total.append(test_episode_reward)
        test_step_total.append(k)

        policy_model.train()
        print('episode: {}'.format(i))
        print('stopped_step: {}'.format(k))
        print('episode_reward: {}'.format(test_episode_reward))
        print('score: {}'.format(sum(score_)))
        print('----------------------------\n')

    # writer.add_scalar('episode_reward', episode_reward, i)

# writer.close()
# plot graph
# plot_reward(episode_reward_total, 'reward_'+name1)
# plot_graph(loss_total, avg_episode_reward, avg_step, name1)
# plot_graph(None, test_episode_reward, test_step, name2)

if not os.path.exists(path + name1):
    os.mkdir(path + name1)

# save data and model
save_data(score_total, path + name1 + '/score.pickle')
save_data(episode_reward_total, path + name1 + '/episode_reward.pickle')
save_data(step_total, path + name1 + '/step.pickle')

save_data(test_score_total, path + name1 + '/test_score.pickle')
save_data(test_episode_reward_total, path + name1 + '/test_episode_reward.pickle')
save_data(test_step_total, path + name1 + '/test_step.pickle')

torch.save(policy_model, './save_model/' + name1 + '_model.pkl')
