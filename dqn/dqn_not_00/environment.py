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

book_rate = 0.1
# num_to_book = int((32 * 100 - 750) * book_rate)
num_to_book = 50
to_book = get_book(num_to_book)

# starting parameters
num_gps = 100
num_slots = 32
# num_pre_booked = 0.5 * num_gps * num_slots
num_pre_booked = 750
agent_pos = [0, 0]


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

        # set action space to move around the grid
        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right

        # set observation space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, self.num_slots, self.num_gps), dtype=np.int32)

        # creates zero filled dataframe with row per time slot and column per gp
        self.state = np.zeros((2, self.num_slots, self.num_gps), dtype=float)

        # randomly enters a 1 for each pre booked appointments
        pre_booked = self.num_pre_booked
        self.pre_booked_pos = []
        while pre_booked > 0:
            pre_booked -= 1
            x, y = np.random.randint(self.num_slots), np.random.randint(self.num_gps)
            while self.state[0, x, y] == 1:
                x, y = np.random.randint(self.num_slots), np.random.randint(self.num_gps)

            self.state[0, x, y] = 1
            self.pre_booked_pos.append([x, y])

        self.init_state = self.state.copy()

    # creates daily diary for each gp, randomly populates prebooked appointments and resets parameters
    def reset(self):
        # randomly sets the agent start space
        self.agent_pos = [16, 50]

        # self.agent_pos = [np.random.randint(self.num_slots), np.random.randint(self.num_gps)]
        # while self.state[0, self.agent_pos[0], self.agent_pos[1]] == 1:
        #     self.agent_pos = [np.random.randint(self.num_slots), np.random.randint(self.num_gps)]
        '''
        # resets parameters for new episode
        print(self.agent_pos)
        print(self.state[self.agent_pos[0], self.agent_pos[1]])
        '''
        self.done = False
        self.reward = 0
        self.appt_idx = 0
        self.state = self.init_state.copy()
        self.state[1, self.agent_pos[0], self.agent_pos[1]] = 100.0
        self.count = 0
        # left = max(self.agent_pos[1] - 2, 0)
        # right = min(self.agent_pos[1] + 2, self.num_gps - 1)
        # up = max(self.agent_pos[0] - 2, 0)
        # down = min(self.agent_pos[0] + 2, self.num_slots - 1)
        # for i in range(left, right + 1):
        #     for j in range(up, down + 1):
        #         self.state[1, j, i] = 1

        # print('starting state', self.state.sum(), self.state)
        return self.state

    # calculates new position of the agent based on the action
    def move_agent(self, action):

        # set boundaries for the grid
        max_row = self.num_slots - 1
        max_col = self.num_gps - 1

        # setting new co-ordinates for the agent
        new_row = self.agent_pos[0]
        new_col = self.agent_pos[1]

        # calculate what the new position may be based on the action without going out the grid
        if action == 0:
            # print('up')
            new_row = max(self.agent_pos[0] - 1, 0)
        if action == 1:
            # print('down')
            new_row = min(self.agent_pos[0] + 1, max_row)
        if action == 2:
            # print('left')
            new_col = max(self.agent_pos[1] - 1, 0)
        if action == 3:
            # print('right')
            new_col = min(self.agent_pos[1] + 1, max_col)

        new_pos = [new_row, new_col]
        # print('new pos', new_pos)

        return new_pos

    # checks if we can look to book appointment starting here
    def check_bookable(self):
        sum_ = 0
        # up = max(self.agent_pos[0] - 1, 0)
        # down = min(self.agent_pos[0] + self.to_book[self.appt_idx], self.num_slots)

        up = self.agent_pos[0] - 1
        down = self.agent_pos[0] + self.to_book[self.appt_idx]

        if up < 0:
            sum_ += self.state[0, down, self.agent_pos[1]] + 1
        else:
            if down >= self.num_slots:
                sum_ += self.state[0, up, self.agent_pos[1]] + 1
            else:
                sum_ += self.state[0, up, self.agent_pos[1]] + self.state[0, down, self.agent_pos[1]]

        if sum_ >= 1:
            return 1
        else:
            return 0

        # return self.state[0, self.agent_pos[0], self.agent_pos[1]] == 0.0

    # checks if the appointment fits
    def check_and_book(self):

        max_row = self.num_slots - 1
        cells_to_check = self.to_book[self.appt_idx]
        score_ = -1

        k = 0
        if (self.agent_pos[0] + cells_to_check - 1) <= max_row:
            while k < cells_to_check:
                if (self.state[0, self.agent_pos[0] + k, self.agent_pos[1]] == 0):
                    k += 1
                    continue
                else:
                    break

            if k == cells_to_check:
                for j in range(cells_to_check):
                    self.state[0, self.agent_pos[0] + j, self.agent_pos[1]] = 1
                score_ = self.get_score()
                self.appt_idx += 1
                self.agent_pos = [(self.agent_pos[0] + k - 1), self.agent_pos[1]]


        # if cells_to_check == 1:
        #     # print('good to check for single')
        #     if self.state[0, self.agent_pos[0], self.agent_pos[1]] == 0:
        #         self.state[0, self.agent_pos[0], self.agent_pos[1]] = 1
        #         score_ = self.get_score()
        #         self.appt_idx += 1

        # if cells_to_check == 2:
        #     # check we're not at the bottom of the grid
        #     if self.agent_pos[0] < max_row:
        #         # check the next cells is also 0.0
        #         # print('good to check for double')
        #         if self.state[0, self.agent_pos[0], self.agent_pos[1]] == 0 and \
        #                 self.state[0, (self.agent_pos[0] + 1), self.agent_pos[1]] == 0:
        #             self.state[0, self.agent_pos[0], self.agent_pos[1]] = 1
        #             self.state[0, (self.agent_pos[0] + 1), self.agent_pos[1]] = 1
        #             score_ = self.get_score()
        #             self.appt_idx += 1
        #             self.agent_pos = [(self.agent_pos[0] + 1), self.agent_pos[1]]
        #             # print('after booking', self.agent_pos)

        # if cells_to_check == 3:
        #     # check we're not at the bottom of the grid
        #     if self.agent_pos[0] + 1 < max_row:
        #         # print('good to check for treble')
        #         if self.state[0, self.agent_pos[0], self.agent_pos[1]] == 0 and \
        #                 self.state[0, (self.agent_pos[0] + 1), self.agent_pos[1]] == 0 \
        #                 and self.state[0, (self.agent_pos[0] + 2), self.agent_pos[1]] == 0:
        #             self.state[0, self.agent_pos[0], self.agent_pos[1]] = 1
        #             self.state[0, (self.agent_pos[0] + 1), self.agent_pos[1]] = 1
        #             self.state[0, (self.agent_pos[0] + 2), self.agent_pos[1]] = 1
        #             score_ = self.get_score()
        #             self.appt_idx += 1
        #             self.agent_pos = [(self.agent_pos[0] + 2), self.agent_pos[1]]


        # if cells_to_check == 4:
        #     # check we're not at the bottom of the grid
        #     if self.agent_pos[0] + 2 < max_row:
        #         # check the next cells is also 0.0
        #         # print('good for quad')
        #         if self.state[0, self.agent_pos[0], self.agent_pos[1]] == 0 and \
        #                 self.state[0, (self.agent_pos[0] + 1), self.agent_pos[1]] == 0 \
        #                 and self.state[0, (self.agent_pos[0] + 2), self.agent_pos[1]] == 0 and \
        #                 self.state[0, (self.agent_pos[0] + 3), self.agent_pos[1]] == 0:
        #             self.state[0, self.agent_pos[0], self.agent_pos[1]] = 1
        #             self.state[0, (self.agent_pos[0] + 1), self.agent_pos[1]] = 1
        #             self.state[0, (self.agent_pos[0] + 2), self.agent_pos[1]] = 1
        #             self.state[0, (self.agent_pos[0] + 3), self.agent_pos[1]] = 1
        #             score_ = self.get_score()
        #             self.appt_idx += 1
        #             self.agent_pos = [(self.agent_pos[0] + 3), self.agent_pos[1]]

        next_state = self.state

        return next_state, score_

    def find_target_pos(self, book):
        min_row, min_col = 0, 0
        for i in range(self.num_slots):
            for j in range(self.num_gps):
                k = 0
                while (k < book) and ((i+k) < self.num_slots):
                    if self.state[0, i+k, j] != 0:
                        break
                    k += 1
                if (k == book) and ((abs(min_row-self.old_agent_pos[0])+abs(min_col-self.old_agent_pos[1])) > (abs(i-self.old_agent_pos[0])+abs(j-self.old_agent_pos[1]))):
                    min_row, min_col = i, j

        return min_row, min_col

    def cal_reward_1(self, pos, action):
        r = 0
        sub_row, sub_col = pos[0] - self.old_agent_pos[0], pos[1] - self.old_agent_pos[1]
        if sub_row > 0:
            if sub_col > 0:
                if action == 1 or action == 3:
                    r = 0.5
                else:
                    r = -1.0
                    # r = -0.5
            elif sub_col == 0:
                if action == 1:
                    r = 0.5
                else:
                    r = -1.0
                    # r = -0.5
            else:
                if action == 1 or action == 2:
                    r = 0.5
                else:
                    r = -1.0
                    # r = -0.5
        elif sub_row < 0:
            if sub_col > 0:
                if action == 0 or action == 3:
                    r = 0.5
                else:
                    r = -1.0
                    # r = -0.5
            elif sub_col == 0:
                if action == 0:
                    r = 0.5
                else:
                    r = -1.0
                    # r = -0.5
            else:
                if action == 0 or action == 2:
                    r = 0.5
                else:
                    r = -1.0
                    # r = -0.5
        else:
            if ((sub_col > 0) and (action == 3)) or ((sub_col < 0) and (action == 2)):
                r = 0.5
            else:
                r = -1.0
                # r = -0.5

        return r

    def cal_reward(self):
        # up = max(0, self.agent_pos[0]-self.to_book[self.appt_idx-1])
        # down = min(31, self.agent_pos[0]+1)
        up = self.agent_pos[0]-self.to_book[self.appt_idx-1]
        down = self.agent_pos[0]+1
        if up < 0:
            up = 0
            if self.state[0, down, self.agent_pos[1]] == 0:
                r = 0.1
            else:
                r = 0.25
        else:
            if down >= self.num_slots:
                if self.state[0, up, self.agent_pos[1]] == 0:
                    r = 0.1
                else:
                    r = 0.25
            else:
                if (self.state[0, up, self.agent_pos[1]] == 0):
                    if (self.state[0, down, self.agent_pos[1]] == 0):
                        r = 0.1
                    else:
                        r = 0.25
                else:
                    if (self.state[0, down, self.agent_pos[1]] == 0):
                        r = 0.25
                    else:
                        r = 0.5

        return r

    def get_score(self):
        score = 0
        up = max(0, self.agent_pos[0]-1)
        down = min(31, self.agent_pos[0] + self.to_book[self.appt_idx])
        if (self.state[0, up, self.old_agent_pos[1]] == 0):
            if (self.state[0, down, self.agent_pos[1]] == 0):
                score = 0
            else:
                score = 0.5
        else:
            if (self.state[0, down, self.agent_pos[1]] == 0):
                score = 0.5
            else:
                score = 1.0

        return score

    def step(self, action, old_action):
        # print(action)
        # print(self.agent_pos)
        # get new position of agent based on action
        new_agent_pos = self.move_agent(action)
        self.old_agent_pos = self.agent_pos
        self.old_appt_idx = self.appt_idx
        self.scores = -1
        position = (0, 0)
        self.state[1, self.agent_pos[0], self.agent_pos[1]] = 0
        self.agent_pos = new_agent_pos
        # print(new_agent_pos)
        # print('new and old pos', new_agent_pos, self.agent_pos)
        # print(self.agent_pos)
        # if the agent is stuck on an edge then move to a new position
        # if self.agent_pos == self.old_agent_pos or (old_action+action)==5 or (old_action+action)==1:
        if self.agent_pos == self.old_agent_pos:
        # if new_agent_pos == self.agent_pos:
            # self.agent_pos = [np.random.randint(self.num_slots), np.random.randint(self.num_gps)]
            # return self.state, -0.1, False, self.appt_idx, self.agent_pos, {}
            self.reward = -0.5
            # print('here1', self.agent_pos)
        else:
            # left = max(self.agent_pos[1] - 2, 0)
            # right = min(self.agent_pos[1] + 2, self.num_gps - 1)
            # up = max(self.agent_pos[0] - 2, 0)
            # down = min(self.agent_pos[0] + 2, self.num_slots - 1)
            # for i in range(left, right + 1):
            #     for j in range(up, down + 1):
            #         self.state[1, j, i] = 0

            # self.state[1, self.agent_pos[0], self.agent_pos[1]] = 5
            # print('here2', self.agent_pos)

            # self.position = self.find_target_pos(self.to_book[self.appt_idx])
            # self.reward = self.cal_reward_1(self.position, action)

            # check if it's possible to book then book
            if self.check_bookable():
                # print('checked here')
                self.state, self.scores= self.check_and_book()

            if self.appt_idx > self.old_appt_idx:
                self.reward = 0.5
                # self.reward = self.cal_reward()
            else:
                self.reward = -0.1

        # work out if episode complete
        if self.appt_idx == len(self.to_book):
            self.done = True

        # # choose agent position randomly if the agent continuously gets bad reward some times
        # if self.reward < 0.:
        #     self.count += 1
        # else:
        #     self.count = 0
        
        # if self.count >=50:
        #     self.agent_pos = [np.random.randint(self.num_slots), np.random.randint(self.num_gps)]
        #     # self.reward = -0.2
        #     self.count = 0

        # print(self.agent_pos)
        self.state[1, self.agent_pos[0], self.agent_pos[1]] = 100.0
        agent_state = self.state.copy()
        # agent_state[self.agent_pos[0], self.agent_pos[1]] = 5
        # print('agent', agent_state)

        info = {}
        
        return agent_state, self.reward, self.done, self.appt_idx, self.old_agent_pos, info
