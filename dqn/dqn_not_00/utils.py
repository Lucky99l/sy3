import os
import pickle
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_book(num_to_book):
    book_list =[]
    for _ in range(num_to_book):
        book_list.append(np.random.randint(3, 7))
    
    return book_list

def set_seeds(seeds):
    np.random.seed(seeds)
    random.seed(seeds)
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def sample(buffer, batch_size):
    index_list = np.random.choice(len(buffer), batch_size, replace=False)
    batch = []
    for index in index_list:
        batch.append(buffer[index])
    return batch


def plot_graph(loss_total, avg_episode_reward, avg_step, path):
    if loss_total != None:
        fig1,ax1 = plt.subplots(figsize=(5,5))
        x = np.arange(len(loss_total))
        ax1.plot(x, loss_total)
        fig1.savefig('./picture/' + path + '_loss' + '.png', dpi=600, format='png')
        # plt.show()
        plt.close()

    if (avg_episode_reward != None) and (avg_step != None):
        episode_index = np.linspace(0, len(avg_episode_reward), len(avg_episode_reward))
        fig2, ax2 = plt.subplots(1, 2, figsize=(5,5))
        ax2[0].plot(episode_index, avg_episode_reward, 'r-', label='avg_episode_reward')
        ax2[0].legend()
        ax2[0].title('max_r' + str(max(avg_episode_reward)))
        ax2[0].set(xlabel='episode/100', ylabel='avg_episode_reward')
        ax2[1].plot(episode_index, avg_step, 'b-', label='avg_step')
        ax2[1].legend()
        ax2[1].set(xlabel='episode/100', ylabel='avg_step')
        fig2.savefig('./picture/' + path + '.png', dpi=600, format='png')
        plt.close()
        # plt.show()

def plot_train(data, data1=None, data2=None, title='abc', path='./path.png'):
    episode_index = np.linspace(0, len(data), len(data))
    fig2, ax2 = plt.subplots(figsize=(5,5))
    ax2.plot(episode_index, data, 'r-')
    if (data1 != None) and (data2 != None):
        ax2.fill_between(episode_index, data1, data2, color='r', alpha=0.2)
    plt.title(title)
    ax2.set(xlabel='episode/100')
    fig2.savefig(path, dpi=600, format='png')
    plt.close()
    # plt.show()

def plot_trajs(length, height, pos1, pos2, pos3, pos4, index, score, episode, path):
    if not os.path.exists(path):
        os.mkdir(path)
    x1, y1, x2, y2 = [], [], [], []
    x3, y3, x4, y4 = [], [], [], []
    for i in range(len(pos1)):
        x1.append(pos1[i][1])
        y1.append(pos1[i][0])

    for j in range(len(pos2)):
        x2.append(pos2[j][1])
        y2.append(pos2[j][0])

    for k in range(len(pos3)):
        x3.append(pos3[k][1])
        y3.append(pos3[k][0])
    
    for l in range(len(pos4)):
        x4.append(pos4[l][1])
        y4.append(pos4[l][0])

    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)
    x3, y3 = np.array(x3), np.array(y3)
    x4, y4 = np.array(x4), np.array(y4)

    fig2, ax2 = plt.subplots(figsize=(5,5))
    if len(x1) != 0:
        ax2.plot(x1, y1, color='r', marker='-', label='real traj')
    if len(x2) != 0:
        ax2.plot(x2, y2, color='b', marker='-', label='expect traj')
    # print(x3)
    # print(len(y3))
    if len(x3) != 0:
        ax2.scatter(x3, y3, color='y', marker='*', label='bad choice')
    if len(x4) != 0:
        ax2.scatter(x4, y4, color='g', marker='*', label='good choice')

    plt.xlim(0, length)
    plt.ylim(0, height)
    ax2.legend()
    plt.title('index' + str(index) + '  ' + 'score' + str(score))
    fig2.savefig(path + '/' + str(episode) + '.png', dpi=600, format='png')
    plt.close()

def plot_traj(length, height, pos1, pos2, pos3, pos4, index, score, episode, path):
    if not os.path.exists(path):
        os.mkdir(path)
    x1, y1, x2, y2 = [], [], [], []
    x3, y3, x4, y4 = [], [], [], []
    for i in range(len(pos1)):
        x1.append(pos1[i][1])
        y1.append(pos1[i][0])

    for j in range(len(pos2)):
        x2.append(pos2[j][1])
        y2.append(pos2[j][0])

    for k in range(len(pos3)):
        x3.append(pos3[k][1])
        y3.append(pos3[k][0])
    
    for l in range(len(pos4)):
        x4.append(pos4[l][1])
        y4.append(pos4[l][0])

    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)
    x3, y3 = np.array(x3), np.array(y3)
    x4, y4 = np.array(x4), np.array(y4)

    fig2, ax2 = plt.subplots(figsize=(5,5))
    if len(x1) != 0:
        ax2.scatter(x1, y1, color='r', marker='o', label='real traj')
    if len(x2) != 0:
        ax2.scatter(x2, y2, color='b', marker='o', label='expect traj')
    # print(x3)
    # print(len(y3))
    if len(x3) != 0:
        ax2.scatter(x3, y3, color='y', marker='*', label='bad choice')
    if len(x4) != 0:
        ax2.scatter(x4, y4, color='g', marker='*', label='good choice')

    plt.xlim(0, length)
    plt.ylim(0, height)
    ax2.legend()
    plt.title('index' + str(index) + '  ' + 'score' + str(score))
    fig2.savefig(path + '/' + str(episode) + '.png', dpi=600, format='png')
    plt.close()

def save_data(data, path): 
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(path): 
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def average_filter(data, base, filter_size=9):
    pad = base * np.ones((filter_size,))
    data = np.array(data)
    data_pad = np.append(np.append(pad, data), pad)
    data_pad_cp = data_pad.copy()
    for i in range(filter_size, len(data_pad)-filter_size):
        temp = data_pad[(i-filter_size):(i+filter_size+1)]
        data_pad_cp[i] = sum(temp)/len(temp)
    
    data_ = data_pad_cp[filter_size:(len(data_pad)-filter_size)]
    return data_

def gradient_cascade(p, k):
    return k*(1-p)**(k-1)
