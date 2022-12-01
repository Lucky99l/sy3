import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("/home/ustc/baojie/scheduler/sy3")

from utils import load_data

data = load_data('./fitness4.pickle')
# data2 = load_data('./index2.pickle')

# data proprecessing
data_avg = []
data_max = []
data_min = []
num_step = 20
for i in range(int(len(data)/num_step)):
    temp = data[(num_step * i):(num_step * i + num_step)]
    data_avg.append(sum(temp)/num_step)
    data_max.append(max(temp))
    data_min.append(min(temp))
data_avg, data_max, data_min = np.array(data_avg), np.array(data_max), np.array(data_min)

x = np.linspace(0, len(data_avg), len(data_avg))
fig, ax = plt.subplots(figsize=(5,5))
# plt.xlim((0,len(data)))
# plt.ylim((80, 140))
# plt.xticks(np.arange(0,len(data),50))
# plt.yticks(np.arange(80,140,5))
# plt.grid(ls='--',c='#D3D3D3')
ax.plot(x, data_avg, 'b-', label='Fitness')
ax.fill_between(x, data_max, data_min, alpha=0.3)
plt.xlabel('generation')
plt.ylabel('fitness')
ax.legend()
# plt.show()
fig.savefig('./fitness4.png', dpi=600, format='png')
