import os
import torch
import random
import numpy as np

from model import Model
from environment import SchedulerEnv

seed = 999
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.load('./save_model/sy3_4_model.pkl')

# policy_model = Model(env.observation_space.shape, env.action_space.n).to(device)
# policy_model.load_state_dict(model.state_dict())

def test(policy_model, device):
    env = SchedulerEnv()
    state = env.reset()

    agent_pos_record = []
    bad_choice = []
    action = -5
    episode_reward = 0
    score = []
    book = env.to_book

    for i in range(1000):
        old_action = action
        action = torch.argmax(policy_model(torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device)).cpu()).item()
        next_state, reward, done, index, agent_pos, _ = env.step(action, old_action)
        score.append(env.scores)
        state = next_state

        if (old_action == action) and ((agent_pos[0] == 0) or (agent_pos[1] == 0) or (agent_pos[0] == 31) or (agent_pos[1] == 99)):
            bad_choice.append(agent_pos)

        episode_reward += reward
        agent_pos_record.append(agent_pos)

        if done:
            break

    score_ = []
    for j in range(len(score)):
        if score[j] >= 0:
            score_.append(score[j])
    
    # print("step: ", i)
    # print("book: ", sum(book))
    # print("test index: ", index)
    # print(score_)
    # print(len(score_))
    return sum(score_), episode_reward, i

# score_eval = test()
# print(score_eval)
