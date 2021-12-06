from MineralEnv import MineralEnv

from absl import flags

import tensorflow as tf
import numpy as np

from util import compute_avg_return
from dqn import DeepQNetwork

# Flags needed for creating pysc2 environment
FLAGS = flags.FLAGS
FLAGS([''])

# Hyperparamters
nS = (1, 84, 84)
nA = 7056
alpha = 1e-5
gamma = 0.99
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.9999
batch_size = 64

train_interval = 5

num_train = 100

env = MineralEnv()

agent = DeepQNetwork(
    nS, nA,
    alpha, gamma,
    epsilon,
    epsilon_min,
    epsilon_decay,
    batch_size
)

for eps in range(num_train):
    print("-----------{} Episode------------".format(eps))
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        action = agent.action(obs)
        new_obs, rew, done, info = env.step(action)

        # Update replay memory
        agent.store(obs, action, rew, new_obs, done)

        if steps % train_interval == 0 and len(agent.memory) > batch_size:
            agent.experience_replay()

        obs = new_obs
        steps += 1
    print(agent.epsilon)

env.close()