import os
import torch
import random
import numpy as np

from utils import get_book

seed = 999
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# book_rate = 0.1
# num_to_book = int((32 * 100 - 750) * book_rate)
num_to_book = 50
to_book = get_book(num_to_book)

# starting parameters
num_gps = 100
num_slots = 32
num_pre_booked = 750
agent_pos = [0, 0]

state = np.zeros((num_slots, num_gps), dtype=float)

# randomly enters a 1 for each pre booked appointments
pre_booked_pos = []
while num_pre_booked > 0:
    num_pre_booked -= 1
    x, y = np.random.randint(num_slots), np.random.randint(num_gps)
    while state[x, y] == 1:
        x, y = np.random.randint(num_slots), np.random.randint(num_gps)
    
    state[x, y] = 1

    pre_booked_pos.append([x, y])

# print(state)

score = 0
for i in range(num_to_book):
    book = to_book[i]
    flag = 0
    for j in range(num_slots):
        for k in range(num_gps):
            for l in range(book):
                if ((j+l) < num_slots) and (state[j+l, k] == 0):
                    continue
                else:
                    l -= 1
                    break
            if l == (book-1):
                flag = 1
                break
        if flag == 1:
            break             

    for t in range(book):
        state[j + t, k] = 1

    up = max(0, j-1)
    down = min(num_slots-1, j+book)

    if (state[up, k] == 0):
        if (state[down, k] == 0):
            score += 0
        else:
            score += book
    else:
        if (state[down, k] == 0):
            score += book
        else:
            score += 2 * book 

print("score: ", score)
