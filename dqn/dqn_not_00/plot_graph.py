import os
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data, plot_graph, plot_train, plot_traj, plot_trajs, average_filter

# path
name1 = 'sy2_55'
name2 = 'sy2_55_test'
path1 = './main_c/data/'
path2 = './main_c/picture/'
if not os.path.exists(path2 + name1):
    os.mkdir(path2 + name1)

# train data
score = load_data(path1 + name1 + '/score.pickle')
episode_reward = load_data(path1 + name1 + '/episode_reward.pickle')
step = load_data(path1 + name1 + '/step.pickle')
# gamma = load_data(path1 + name1 + '/gamma.pickle')

# gamma = []
# init_gamma = 0.1
# for _ in range(len(step)):
#     if init_gamma < 0.99:
#         init_gamma = 1-0.98*(1-init_gamma)
#         gamma.append(init_gamma)
#     else:
#         gamma.append(0.99)

# episode_index = np.linspace(0, len(gamma), len(gamma))
# fig1, ax1 = plt.subplots(figsize=(5,5))
# ax1.plot(episode_index, gamma, 'r-')
# # ax2.fill_between(episode_index, data1, data2, color='r', alpha=0.2)
# plt.title('gamma')
# ax1.set(xlabel='episode')
# fig1.savefig(path2 + name1 + '/gamma.png', dpi=600, format='png')
# plt.close()

# test data
# test_score = load_data(path1 + name1 + '/test_score.pickle')
# test_episode_reward = load_data(path1 + name1 + '/test_episode_reward.pickle')
# test_step = load_data(path1 + name1 + '/test_step.pickle')

# data proprecessing
interval = 100
avg_episode_reward = []
max_episode_reward = []
min_episode_reward = []
for i in range(len(episode_reward)):
    # if episode_reward[i] > 0:
    #     episode_reward[i] += 400
    if (i % interval == 0) and (i != 0):
        temp1 = episode_reward[(i-interval):i]
        avg_episode_reward.append(sum(temp1)/len(temp1))
        max_episode_reward.append(max(temp1))
        min_episode_reward.append(min(temp1))

avg_score = []
max_score = []
min_score = []
# avg_rate = []
for j in range(len(score)):
    if (j % interval == 0) and (j != 0):
        temp2 = score[(j-interval):j]
        avg_score.append(sum(temp2)/len(temp2))
        max_score.append(max(temp2))
        min_score.append(min(temp2))
        # avg_rate.append(temp/196)

avg_step = []
max_step = []
min_step = []
for k in range(len(step)):
    # if step[k] > 800:
    #     step[k] -= 800
    if (k % interval == 0) and (k != 0):
        temp3 = step[(k-interval):k]
        avg_step.append(sum(temp3)/len(temp3))
        max_step.append(max(temp3))
        min_step.append(min(temp3))

# train total graph
plot_train(score, title='score', path=path2 + name1 + '/score.png')
plot_train(step, title='step', path=path2 + name1 + '/step.png')
plot_train(episode_reward, title='episode_reward', path=path2 + name1 + '/train_reward.png')

# train average graph
# plot_train(avg_score, title='avg_score', path=path2 + name1 + '/avg_score.png')
# # plot_train(avg_rate, 'avg_rate', path2 + name1 + '/avg_rate.png')
# plot_train(avg_step, title='avg_step', path=path2 + name1 + '/avg_step.png')
# plot_train(avg_episode_reward, title='avg_episode_reward', path=path2 + name1 + '/avg_train_reward.png')

plot_train(avg_score, title='avg_score', path=path2 + name1 + '/avg_score.png')
# plot_train(avg_rate, 'avg_rate', path2 + name1 + '/avg_rate.png')
plot_train(avg_step, title='avg_step', path=path2 + name1 + '/avg_step.png')
plot_train(avg_episode_reward, title='avg_episode_reward', path=path2 + name1 + '/avg_train_reward.png')

plot_train(avg_score, max_score, min_score, title='avg_score', path=path2 + name1 + '/avg_score_1.png')
# plot_train(avg_rate, 'avg_rate', path2 + name1 + '/avg_rate.png')
plot_train(avg_step, max_step, min_step, title='avg_step', path=path2 + name1 + '/avg_step_1.png')
plot_train(avg_episode_reward, max_episode_reward, min_episode_reward, title='avg_episode_reward', path=path2 + name1 + '/avg_train_reward_1.png')

# train average filter graph
# print(np.mean(avg_episode_reward))
# avg_reward_filter = average_filter(avg_episode_reward, 400)
# plot_train(avg_reward_filter, 'avg_episode_reward', path2 + name1 + '/train_filter_avg_reward_400.png')
# # print(np.mean(avg_score))
# avg_score_filter = average_filter(avg_score, 150)
# plot_train(avg_score_filter, 'avg_score', path2 + name1 + '/train_filter_avg_score_400.png')
# avg_rate_filter = average_filter(avg_rate, 0.75)
# plot_train(avg_rate_filter, 'avg_rate', path2 + name1 + '/train_filter_avg_rate_400.png')

# avg_step_filter = average_filter(avg_step, 150)
# plot_train(avg_step_filter, 'avg_step', path2 + name1 + '/train_filter_avg_step_400.png')

# test graph
# plot_train(test_score, title='test_score', path=path2 + name1 + '/test_score.png')
# plot_train(test_episode_reward, title='test_episode_reward', path=path2 + name1 + '/test_episode_reward.png')
# plot_train(test_step, title='test_stesp', path=path2 + name1 + '/test_step.png')

