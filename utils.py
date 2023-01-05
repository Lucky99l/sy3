import os
import sys
import pickle
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("/home/xliu/scheduler/sy3")

def get_book(num_to_book):
    set_seeds(999)
    # book_list =[]
    # for _ in range(num_to_book):
    #     book_list.append(np.random.randint(1,4))
    path = r'/home/xliu/scheduler/sy3/data/JanData.xlsx'
    data = pd.read_excel(path)
    num_F2F = sum(data['F2F'])
    num_HV = sum(data['HV'])
    num_T = sum(data['T'])
    num_V = sum(data['V'])
    total = num_F2F + num_HV + num_T + num_V
    p_1 = (num_F2F + num_V) / total
    p_2 = num_T / total
    p_3 = num_HV / total

    book_list = random.choices([1, 2, 3], weights=[p_1, p_2, p_3], k=num_to_book)
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
 
    # if title == 'avg_score':
    #     plt.yticks(np.arange(100, 180, 10))
    #     plt.ylim((100, 180))
    # elif title == 'avg_step':
    #     plt.yticks(np.arange(50, 200, 20))
    #     plt.ylim((50, 200))

    fig2.savefig(path, dpi=600, format='png')
    plt.close()
    # plt.show()


def plot_traj(length, height, pos1, pos3, path, index):
    x1, y1, x2, y2 = [], [], [], []
    x3, y3 = [], []
    for i in range(len(pos1)):
        x1.append(pos1[i][1])
        y1.append(pos1[i][0])

    for k in range(len(pos3)):
        x3.append(pos3[k][1])
        y3.append(pos3[k][0])

    x1, y1 = np.array(x1), np.array(y1)
    x3, y3 = np.array(x3), np.array(y3)
    fig2, ax2 = plt.subplots(figsize=(5,5))
    ax2.plot(x1, y1, 'r-', label='real_traj')
    # print(x3)
    # print(len(y3))
    if len(x3) != 0:
        ax2.scatter(x3, y3, marker='*', color='y')

    plt.xlim(0, length)
    plt.ylim(0, height)
    ax2.legend()
    plt.title('index' + str(index))
    fig2.savefig(path, dpi=600, format='png')
    plt.close()

def plot_trajs(reward, length, height, pos1, pos2, episode, path):
    if not os.path.exists(path):
        os.mkdir(path)
    x1, y1, x2, y2 = [], [], [], []
    for i in range(len(pos1)):
        x1.append(pos1[i][1])
        y1.append(pos1[i][0])

    for j in range(len(pos2)):
        x2.append(pos2[j][1])
        y2.append(pos2[j][0])

    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)

    fig2, ax2 = plt.subplots(figsize=(5,5))
    ax2.plot(x1, y1, 'r-', label='real_traj')

    if len(x2) != 0:
        ax2.scatter(x2, y2, marker='*', color='y')

    plt.xlim(0, length)
    plt.ylim(0, height)
    ax2.legend()
    plt.title('traj')
    fig2.savefig(path + '/' + str(episode) +'_' + str(reward) + '.png', dpi=600, format='png')
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
