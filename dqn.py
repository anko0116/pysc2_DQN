import numpy as np
from collections import deque
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import random

class DeepQNetwork():
    def __init__(self, states, actions, alpha, 
                gamma, epsilon, epsilon_min, 
                epsilon_decay, batch_size, model=None, test=False):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma
        #Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
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
            16, 3, strides=1, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(
            1, 1, strides=1, padding='same'))
        #model.add(PickQLayer())
        model.add(keras.layers.Flatten())
        model.compile(loss='mean_squared_error', #Loss function: Mean Squared Error
                      optimizer=keras.optimizers.Adam(learning_rate=self.alpha)) #Optimaizer: Adam (Feel free to check other options)
        return model

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(7056)
        action = self.model.predict(state)
        return np.argmax(action[0])
        return action

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
        nst = np.zeros((1, 84, 84))#Next States
        for i in range(len(np_array)): #Creating the state and next state np arrays
            st = np.append( st, np_array[i][0], axis=0)
            nst = np.append( nst, np_array[i][3], axis=0)
        st = np.reshape(st, (self.batch_size+1, 1, 84, 84))
        nst = np.reshape(nst, (self.batch_size+1, 1, 84, 84))
        st_predict = self.model.predict(st) #Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            #Predict from state
            nst_action_predict_model = nst_predict[index]
            if done == True: #Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:   #Non terminal
                target = reward + self.gamma * np.amax(nst_action_predict_model)
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
            hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, 
                                verbose=0, callbacks=[self.cp_callback])
        else:
            hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)

        #Graph Losses
        for i in range(epoch_count):
            self.loss.append( hist.history['loss'][i] )
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Layers below

class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PreprocessLayer, self).__init__()
        self.marine_idx = 1
        self.mineral_idx = 3

    def build(self, arg):
        self.marine_idx = 1
        self.mineral_idx = 3

    def call(self, inputs):
        # https://www.tensorflow.org/api_docs/python/tf/math/equal
        flat_inputs = tf.reshape(inputs, (-1, 1, 84*84))
        marine_indices = tf.math.equal(inputs, 1) # Returns list of Trues and Falses
        mineral_indices = tf.math.equal(inputs, 3)
        
        marine_indices = tf.cast(marine_indices, tf.uint8)
        mineral_indices = tf.cast(mineral_indices, tf.uint8)

        marine_img = tf.one_hot(marine_indices, 1)
        mineral_img = tf.one_hot(mineral_indices, 1)

        marine_img = tf.reshape(marine_img, (-1, 84, 84, 1))
        mineral_img = tf.reshape(mineral_img, (-1, 84, 84, 1))
        output = tf.concat([marine_img, mineral_img], -1)
        print(output)
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