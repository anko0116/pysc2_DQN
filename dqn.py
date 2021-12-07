import numpy as np
from collections import deque
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import random

np.set_printoptions(threshold=np.inf)

class DeepQNetwork():
    def __init__(self, states, actions, alpha, 
                gamma, epsilon, epsilon_min, 
                epsilon_decay, batch_size, model=None, test=False):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=5000)
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma
        #Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.model_target = self.build_model()

        self.loss = []
        
        self.test = test
        # Used in model fit for model saving
        checkpoint_path = 'training_1/cp.ckpt'
        self.checkpoint_dir = os.path.dirname(checkpoint_path)
        self.cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir,
                                                            save_freq=50,
                                                            save_weights_only=True,
                                                            verbose=0)

    def build_model(self):
        model = keras.Sequential()
        model.add(PreprocessLayer())

        model.add(keras.layers.Conv2D(
            16, 5, strides=1, padding='same', activation='relu'))
        # model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2D(
            32, 3, strides=1, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2D(
            1, 1, strides=1, padding='same', activation=None))

        model.add(keras.layers.Flatten())

        model.compile(loss='mean_squared_error', #Loss function: Mean Squared Error
                      optimizer=keras.optimizers.Adam(learning_rate=self.alpha)) #Optimaizer: Adam (Feel free to check other options)
        return model

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(7056)

        action = self.model.predict(state)
        return np.argmax(action[0])

    def test_action(self, state): #Exploit
        action_vals = self.model.predict(state)
        return action_vals

    def store(self, state, action, reward, nstate, done):
        #Store the experience in memory
        self.memory.append( (state, action, reward, nstate, done) )

    def experience_replay(self):
        #Execute the experience replay
        minibatch = random.sample( self.memory, self.batch_size ) #Randomly sample from memory
        #Convert to numpy for speed by vectorization
        x = []
        y = []
        #np_array = np.array(minibatch)
        np_array = minibatch
        st = np.zeros((1, 84, 84)) #States
        nst = np.zeros((1, 84, 84)) #Next States
        for i in range(len(np_array)): #Creating the state and next state np arrays
            st = np.append( st, np_array[i][0], axis=0)
            nst = np.append( nst, np_array[i][3], axis=0)
        st = np.reshape(st, (self.batch_size+1, 1, 84, 84))
        nst = np.reshape(nst, (self.batch_size+1, 1, 84, 84))
        st_predict = self.model.predict(st) #Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        index = 1
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            #Predict from state
            nst_action_predict_model = nst_predict[index]
            if done == True: #Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:   #Non terminal
                target = reward + self.gamma * np.amax(nst_action_predict_model[0])
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        #Reshape for Keras Fit
        x_reshape = np.array(x)
        x_reshape = np.reshape(x_reshape, (self.batch_size, 1, 84, 84))
        y_reshape = np.array(y)
        epoch_count = 1 #Epochs is the number or iterations
        
        hist = None
        if not self.test:
            hist = self.model.fit(
                x_reshape, y_reshape, epochs=epoch_count, 
                verbose=0,
                callbacks=[self.cp_callback]
            )
        else:
            hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)

        #Graph Losses
        for i in range(epoch_count):
            self.loss.append( hist.history['loss'][i] )

        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            # self.epsilon -= 0.0000495
            self.epsilon -= 0.00003

    def experience_replay_ddqn(self):
        minibatch = random.sample(self.memory, self.batch_size) #Randomly sample from memory

        x = []
        y = []
        np_array = np.array(minibatch)
        st = np.zeros((1,84,84))
        nst = np.zeros((1,84,84))
        for i in range(len(np_array)): #Creating the state and next state np arrays
            st = np.append(st, np_array[i][0], axis=0)
            nst = np.append(nst, np_array[i][3], axis=0)
        st_predict = self.model.predict(st)
        nst_predict = self.model.predict(nst)
        nst_predict_target = self.model_target.predict(nst) #Predict from the TARGET
        index = 1
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            #Predict from state
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]
            if done == True:
                target = reward
            else:   #Non terminal
                target = reward + self.gamma * nst_action_predict_target[np.argmax(nst_action_predict_model)] #Using Q to get T is Double DQN
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        #Reshape for Keras Fit
        x_reshape = np.array(x).reshape((self.batch_size,1,84,84))
        y_reshape = np.array(y)
        epoch_count = 1
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        #Graph Losses
        for i in range(epoch_count):
            self.loss.append( hist.history['loss'][i] )
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            self.epsilon -= 0.000025

# Layers below

class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PreprocessLayer, self).__init__()
        self.map_idx = 0
        self.marine_idx = 1
        self.mineral_idx = 3

    def build(self, arg):
        self.conv = tf.keras.layers.Conv2D(
            1, 1, strides=1, padding='same', activation=None)

    def call(self, inputs):
        # https://www.tensorflow.org/api_docs/python/tf/math/equal
        # flat_inputs = tf.reshape(inputs, (-1, 1, 84*84))
        map_indices = tf.math.equal(inputs, self.map_idx)
        marine_indices = tf.math.equal(inputs, self.marine_idx)
        mineral_indices = tf.math.equal(inputs, self.mineral_idx)
        
        map_img = tf.cast(map_indices, tf.float16)
        marine_img = tf.cast(marine_indices, tf.float16)
        mineral_img = tf.cast(mineral_indices, tf.float16)

        map_img = tf.reshape(map_img, (-1, 84, 84, 1))
        marine_img = tf.reshape(marine_img, (-1, 84, 84, 1))
        mineral_img = tf.reshape(mineral_img, (-1, 84, 84, 1))
        output = tf.concat([map_img, marine_img, mineral_img], -1)
        output = self.conv(output)
        return output

# class PickQLayer(tf.keras.layers.Layer):
#     def __init__(self):
#         super(PickQLayer, self).__init__()
#     def build(self, arg):
#         return
#     def call(self, inputs):
#         # Return location on the map with highest Q value
#         inputs = tf.reshape(inputs, (1, 7056))
#         idx = tf.argmax(inputs, axis=1)
#         action = tf.one_hot(idx, 7056)
#         return action