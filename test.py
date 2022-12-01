import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from model import Model
from environment import SchedulerEnv
from utils import plot_trajs


def test(model, path, episode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = SchedulerEnv()
    policy_model = Model(env.observation_space.shape, env.action_space.n).to(device)
    policy_model.load_state_dict(model.state_dict())
    # pre_booked_list = env.pre_booked_pos

    state = env.reset()
    agent_pos_record = []
    bad_choice = []
    action = -1
    episode_reward = 0
    for i in range(1000):
        old_action = action
        action = torch.argmax(policy_model(torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device)).cpu()).item()
        next_state, reward, done, index, agent_pos, _ = env.step(action)
        state = next_state

        if (old_action == action) and ((agent_pos[0] == 0) or (agent_pos[1] == 0) or (agent_pos[0] == 31) or (agent_pos[1] == 99)):
                bad_choice.append(agent_pos)

        episode_reward += reward
        agent_pos_record.append(agent_pos)

        if done:
            break
    
    print("test index: ", index)
    if episode_reward < 0:
        plot_trajs(episode_reward, env.num_gps, env.num_slots, agent_pos_record, bad_choice, episode, path)

    