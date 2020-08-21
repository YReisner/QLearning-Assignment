import gym
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import box

import ffmpeg
from model import Network

np.random.seed(42)

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'not_done'))


training_params = {
    'batch_size': 256,
    'gamma': 0.95,
    'epsilon_start': 1.1,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.95,
    'target_update': 'soft',  # use 'soft' or 'hard'
    'tau': 0.01,  # relevant for soft update
    'target_update_period': 15,  # relevant for hard update
    'grad_clip': 0.1,
}

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env,'recording',force=True)

network_params = {
    'state_dim': env.observation_space.shape[0],
    'action_dim': env.action_space.n,
    'hidden_dim': 64}
device = torch.device("cpu")


network_params = box.Box(network_params)
params = box.Box(training_params)

FPS = 25
visualize = 'True'


net = Network(network_params, device).to(device)


print('load best model ...')
net.load_state_dict(torch.load('best.dat'))

print('make movie ...')
state = env.reset()
total_reward = 0.0
c = collections.Counter()

while True:
    start_ts = time.time()
    if visualize:
        env.render()
    state_v = torch.tensor(np.array([state], copy=False)).float()
    q_vals = net(state_v).data.numpy()[0]
    action = np.argmax(q_vals)
    c[action] += 1
    state, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        break
    if visualize:
        delta = 1 / FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
print("Total reward: %.2f" % total_reward)
print("Action counts:", c)
env.close()