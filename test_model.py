import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
from dqn import DeepQNetwork, PreprocessLayer
from MineralEnv import MineralEnv

np.set_printoptions(threshold=np.inf)

# Hyperparamters
nS = (1, 84, 84)
nA = 7056
alpha = 1e-5
gamma = 0.99
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.9999
batch_size = 32

realtime = False
visualize = False

# agent = DeepQNetwork(
#     nS, nA,
#     alpha, gamma,
#     epsilon,
#     epsilon_min,
#     epsilon_decay,
#     batch_size
# )

model = keras.Sequential()
model.add(PreprocessLayer())

env = MineralEnv(realtime, visualize)
obs = env.reset()

output = model(obs)
#print(obs)
#print(output)

# for i in range(len(output[0])):
#     print(output[0][i])

ab = np.zeros((0, 5))
ab = np.append(ab, [1,1,1,1,1], axis=0)
print(ab)
