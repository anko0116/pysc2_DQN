import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
from dqn import DeepQNetwork

# Hyperparamters
nS = (1, 84, 84)
nA = 7056
alpha = 1e-5
gamma = 0.99
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.9999
batch_size = 64

agent = DeepQNetwork(
    nS, nA,
    alpha, gamma,
    epsilon,
    epsilon_min,
    epsilon_decay,
    batch_size
)
model = agent.model

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions

# Testing
test = np.random.random((1, 84, 84))[np.newaxis,...]
layer_outs = [func([test, 1.]) for func in functors]
print(layer_outs)