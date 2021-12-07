from MineralEnv import MineralEnv

from absl import flags
import tensorflow as tf
import numpy as np
import os

from dqn import DeepQNetwork

# Flags needed for creating pysc2 environment
FLAGS = flags.FLAGS
FLAGS([''])

# Hyperparamters
nS = (1, 84, 84)
nA = 7056
alpha = 1e-5
gamma = 0.95
epsilon = 0
epsilon_min = 0.01
epsilon_decay = 0.99995
batch_size = 32

train_interval = 1

num_test = 10

realtime = False
visualize = True
# End Hyperparameters

env = MineralEnv(realtime, visualize)

agent = DeepQNetwork(
    nS, nA,
    alpha, gamma,
    epsilon,
    epsilon_min,
    epsilon_decay,
    batch_size
)

if os.path.exists('training_1.index'):
    print("Loading saved model")
    agent.model.load_weights(agent.checkpoint_dir)
else:
    print("ERROR: NO SAVED MODEL EXISTS")
    exit()

all_rewards = []
for eps in range(num_test):
    print("-----------{} Episode------------".format(eps))
    obs = env.reset()
    done = False
    steps = 0
    reward = 0
    while not done:
        action = agent.action(obs)
        new_obs, rew, done, info = env.step(action)
        obs = new_obs
        steps += 1
        reward += rew

    all_rewards.append(reward)
    print(steps, reward)
    if len(agent.loss) > 0:
        print(agent.loss[-1])

env.close()