import gym
import random
import torch
import numpy as np

from utils import get_book

seed = 999
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# starting parameters
num_gps = 100
num_slots = 32
# num_pre_booked = random.randint(20*num_gps, 30*num_gps)
# num_pre_booked = 0.5 * num_gps * num_slots
num_pre_booked = 750
agent_pos = [0, 0]

num_to_book = 50
to_book = get_book(num_to_book)

class SchedulerEnv(gym.Env):
    def __init__(self):
        # set parameters for the day
        self.num_gps = num_gps
        self.num_slots = num_slots

        self.num_pre_booked = num_pre_booked
        self.to_book = to_book
        self.num_to_book = num_to_book
        self.diary_slots = num_gps * num_slots
        self.agent_pos = agent_pos

        self.num_change = 10

        # set action space to move around the grid
        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right

        # set observation space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(9, 9), dtype=np.int32)
        self.state = np.zeros((9, 9), dtype=float)

        # creates zero filled dataframe with row per time slot and column per gp
        self.table = np.zeros((self.num_slots + 8, self.num_gps + 8), dtype=float)
        for i in range(self.num_slots + 4):
            for j in range(self.num_gps + 4):
                if (i < 4) or (j < 4) or (i > self.num_slots+3) or (j > self.num_gps+3):
                    self.table[i, j] = -1

        # randomly enters a 1 for each pre booked appointments
        self.pre_booked_pos = []
        pre_booked = self.num_pre_booked
        while pre_booked > 0:
            pre_booked -= 1
            x, y = np.random.randint(4, self.num_slots+4), np.random.randint(4, self.num_gps+4)
            while self.table[x, y] == 1:
                x, y = np.random.randint(4, self.num_slots+4), np.random.randint(4, self.num_gps+4)

            self.table[x, y] = 1
            self.pre_booked_pos.append([x, y])
            
        self.init_table = self.table.copy()
        

    # creates daily diary for each gp, randomly populates prebooked appointments and resets parameters
    def reset(self):
        # randomly sets the agent start space
        self.agent_pos = [16, 50]

        # self.agent_pos = [np.random.randint(4, self.num_slots+4), np.random.randint(4, self.num_gps+4)]
        # while self.state[0, self.agent_pos[0], self.agent_pos[1]] == 1:
        #     self.agent_pos = [np.random.randint(self.num_slots), np.random.randint(self.num_gps)]
        # self.table = self.init_table.copy()

        self.state = self.table[(self.agent_pos[0]-4):(self.agent_pos[0]+5), (self.agent_pos[1]-4):(self.agent_pos[1]+5)]

        self.scores = -1
        self.done = False
        self.reward = 0
        self.appt_idx = 0

        return self.state

    # calculates new position of the agent based on the action
    def move_agent(self, action):

        # set boundaries for the grid
        max_row = self.num_slots + 3
        max_col = self.num_gps + 3

        # setting new co-ordinates for the agent
        new_row = self.agent_pos[0]
        new_col = self.agent_pos[1]

        # calculate what the new position may be based on the action without going out the grid
        if action == 0:
            # print('up')
            new_row = max(self.agent_pos[0] - 1, 4)
        if action == 1:
            # print('down')
            new_row = min(self.agent_pos[0] + 1, max_row)
        if action == 2:
            # print('left')
            new_col = max(self.agent_pos[1] - 1, 4)
        if action == 3:
            # print('right')
            new_col = min(self.agent_pos[1] + 1, max_col)

        new_pos = [new_row, new_col]
        # print('new pos', new_pos)

        return new_pos

    # checks if we can look to book appointment starting here
    def check_bookable(self):
        sum_ = 0
        for idx in range(self.agent_pos[0], self.agent_pos[0]+self.to_book[self.appt_idx]):
            sum_ += self.table[idx, self.agent_pos[1]]
        if sum_ > 0:
            return 0
        else:
            return 1

        # up = self.agent_pos[0] - 1
        # down = self.agent_pos[0] + self.to_book[self.appt_idx]
        # if up < 4:
        #     for idx in range(4, down+1):
        #         sum_ += self.table[idx, self.agent_pos[1]]
        # else:
        #     if down > self.num_slots + 3:
        #         for idx in range(up, 36):
        #             sum_ += self.table[idx, self.agent_pos[1]]
        #     else:
        #         for idx in range(up, down + 1):
        #             sum_ += self.table[idx, self.agent_pos[1]]

        # if sum_ >= 1:
        #     return 1
        # else:
        #     return 0

        # return self.table[self.agent_pos[0], self.agent_pos[1]] == 0.0

    # checks if the appointment fits
    def check_and_book(self):

        max_row = self.num_slots + 3
        cells_to_check = self.to_book[self.appt_idx]
        score_ = -1

        if cells_to_check == 1:
            # print('good to check for single')
            if self.table[self.agent_pos[0], self.agent_pos[1]] == 0:
                self.table[self.agent_pos[0], self.agent_pos[1]] = 1
                score_ = self.get_score()
                self.appt_idx += 1

        if cells_to_check == 2:
            # check we're not at the bottom of the grid
            if self.agent_pos[0] < max_row:
                # check the next cells is also 0.0
                # print('good to check for double')
                if self.table[self.agent_pos[0], self.agent_pos[1]] == 0 and \
                        self.table[(self.agent_pos[0] + 1), self.agent_pos[1]] == 0:
                    self.table[self.agent_pos[0], self.agent_pos[1]] = 1
                    self.table[(self.agent_pos[0] + 1), self.agent_pos[1]] = 1
                    score_ = self.get_score()
                    self.appt_idx += 1
                    self.agent_pos = [(self.agent_pos[0] + 1), self.agent_pos[1]]
                    # print('after booking', self.agent_pos)

        if cells_to_check == 3:
            # check we're not at the bottom of the grid
            if self.agent_pos[0] + 1 < max_row:
                # print('good to check for treble')
                if self.table[self.agent_pos[0], self.agent_pos[1]] == 0 and \
                        self.table[(self.agent_pos[0] + 1), self.agent_pos[1]] == 0 \
                        and self.table[(self.agent_pos[0] + 2), self.agent_pos[1]] == 0:
                    self.table[self.agent_pos[0], self.agent_pos[1]] = 1
                    self.table[(self.agent_pos[0] + 1), self.agent_pos[1]] = 1
                    self.table[(self.agent_pos[0] + 2), self.agent_pos[1]] = 1
                    score_ = self.get_score()
                    self.appt_idx += 1
                    self.agent_pos = [(self.agent_pos[0] + 2), self.agent_pos[1]]


        if cells_to_check == 4:
            # check we're not at the bottom of the grid
            if self.agent_pos[0] + 2 < max_row:
                # check the next cells is also 0.0
                # print('good for quad')
                if self.table[self.agent_pos[0], self.agent_pos[1]] == 0 and \
                        self.table[(self.agent_pos[0] + 1), self.agent_pos[1]] == 0 \
                        and self.table[(self.agent_pos[0] + 2), self.agent_pos[1]] == 0 and \
                        self.table[(self.agent_pos[0] + 3), self.agent_pos[1]] == 0:
                    self.table[self.agent_pos[0], self.agent_pos[1]] = 1
                    self.table[(self.agent_pos[0] + 1), self.agent_pos[1]] = 1
                    self.table[(self.agent_pos[0] + 2), self.agent_pos[1]] = 1
                    self.table[(self.agent_pos[0] + 3), self.agent_pos[1]] = 1
                    score_ = self.get_score()
                    self.appt_idx += 1
                    self.agent_pos = [(self.agent_pos[0] + 3), self.agent_pos[1]]

        next_state = self.table[(self.agent_pos[0]-4):(self.agent_pos[0]+5), (self.agent_pos[1]-4):(self.agent_pos[1]+5)]

        return next_state, score_

    def cal_reward(self):
        up = self.agent_pos[0] - self.to_book[self.appt_idx-1]
        down = self.agent_pos[0] + 1
        if up < 4:
            if self.table[down, self.agent_pos[1]] == 0:
                r = 0.1
            else:
                r = 0.25
        else:
            if down > self.num_slots + 3:
                if self.table[up, self.agent_pos[1]] == 0:
                    r = 0.1
                else:
                    r = 0.25
            else:
                if self.table[up, self.agent_pos[1]] == 0:
                    if self.table[down, self.agent_pos[1]] == 0:
                        r = 0.1
                    else:
                        r = 0.25
                else:
                    if self.table[down, self.agent_pos[1]] == 0:
                        r = 0.25
                    else:
                        r = 0.5

        # up = max(4, self.agent_pos[0] - self.to_book[self.appt_idx-1])
        # down = min(self.num_slots+3, self.agent_pos[0]+1)
        # if (self.table[up, self.agent_pos[1]] == 0):
        #     if (self.table[down, self.agent_pos[1]] == 0):
        #         r = 0.1
        #     else:
        #         r = 0.25
        # else:
        #     if (self.table[down, self.agent_pos[1]] == 0):
        #         r = 0.25
        #     else:
        #         r = 0.5

        return r

    def cal_rewards(self):
        length = self.state.shape[0]
        sum_ = 0
        count = 0
        for i in range(length):
            for j in range(length):
                if self.state[i, j] >= 0:
                    sum_ += self.state[i, j]
                    count += 1

        # return 1 - sum(sum(self.state[i]) for i in range(length))/(length * length)
        return 1 - sum_/count
            

    def get_score(self):
        score = 0
        up = self.agent_pos[0] - 1
        down = self.agent_pos[0] + self.to_book[self.appt_idx]
        if up < 4:
            if self.table[down, self.agent_pos[1]] == 0:
                score = 0.5
            else:
                score = 1.0
        else:
            if down > self.num_slots + 3:
                if self.table[up, self.agent_pos[1]] == 0:
                    score = 0.5
                else:
                    score = 1.0
            else:
                if self.table[up, self.agent_pos[1]] == 0:
                    if self.table[down, self.agent_pos[1]] == 0:
                        score = 0
                    else:
                        score = 0.5
                else:
                    if self.table[down, self.agent_pos[1]] == 0:
                        score = 0.5
                    else:
                        score = 1.0

        return score
        

    def step(self, action, old_action):
        # print(action)
        # print(self.agent_pos)

        # # dynamic prebook
        # num = self.num_change
        # if np.random.rand() < 0.1:
        #     while num > 0:
        #         num -= 1
        #         x, y = random.choice(self.pre_booked_pos)
        #         if self.table[x, y] == 1:
        #             self.table[x, y] = 0
        #         else:
        #             self.table[x, y] = 1

        # get new position of agent based on action
        new_agent_pos = self.move_agent(action)
        self.old_agent_pos = self.agent_pos
        self.old_appt_idx = self.appt_idx
        self.scores = -1
        position = (0, 0)
        # print(new_agent_pos)
        # print('new and old pos', new_agent_pos, self.agent_pos)
        # print(self.agent_pos)
        # if the agent is stuck on an edge then move to a new position
        if new_agent_pos == self.old_agent_pos or (old_action+action)==5 or (old_action+action)==1:
        # if new_agent_pos == self.agent_pos:
            # self.agent_pos = [np.random.randint(self.num_slots), np.random.randint(self.num_gps)]
            # return self.state, -0.1, False, self.appt_idx, self.agent_pos, {}
            self.reward = -0.5
            # print('here1', self.agent_pos)
        else:
            self.agent_pos = new_agent_pos
            # self.state[1, self.agent_pos[0], self.agent_pos[1]] = 5
            # print('here2', self.agent_pos)

            # check if it's possible to book then book
            if self.check_bookable():
                # print('checked here')
                self.state, self.scores= self.check_and_book()
                self.reward = self.cal_reward()
            else:
                self.reward = 0.

            # if self.appt_idx > self.old_appt_idx:
            #     self.reward += 0.5
            
            # if self.appt_idx > self.old_appt_idx:
            #     # self.reward = 0.5 + self.cal_rewards()
            #     self.reward = self.cal_reward()
            # else:
            #     # self.reward = self.cal_rewards()
            #     self.reward = 0.

        # work out if episode complete
        if self.appt_idx == len(self.to_book):
            self.done = True

        # print(self.agent_pos)
        agent_state = self.state.copy()
        true_agent_pos = [self.agent_pos[0]-4, self.agent_pos[1]-4]

        info = {}
        return agent_state, self.reward, self.done, self.appt_idx, true_agent_pos, info
