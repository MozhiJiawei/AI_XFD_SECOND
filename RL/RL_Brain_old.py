# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import tensorflow as tf
import numpy as np
import random
import logging
from collections import deque
from .Memory import ReplayMemory

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 1000.  # timesteps to observe before training
EXPLORE = 200000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.001  # 0.001 # final value of epsilon
INITIAL_EPSILON = 0.3  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 100

try:
    tf.mul
except:
    # For new version of tensorflow
    # tf.mul has been removed in new version of tensorflow
    # Using tf.multiply to replace tf.mul
    tf.mul = tf.multiply


class BrainDQN:

    def __init__(self):

        # init replay memory
        self.replayMemory = ReplayMemory()
        # init some parameters
        self.map_sharp = 12
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = 9
        self.n_channel = 6

        # init Q network
        self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2 = self.createQNetwork()

        # init Target Q Network
        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T = self.createQNetwork()

        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                            self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),
                                            self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),
                                            self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            logging.info("Successfully loaded:{}".format(checkpoint.model_checkpoint_path))
        else:
            logging.info("Could not find old network weights")

    def createQNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([8, 8, self.n_channel, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([256, 128])
        b_fc1 = self.bias_variable([128])

        W_fc2 = self.weight_variable([128, self.actions])
        b_fc2 = self.bias_variable([self.actions])

        # input layer

        stateInput = tf.placeholder("float", [None, self.map_sharp, self.map_sharp, self.n_channel])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 2) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # Q Value layer
        QValue = tf.matmul(h_fc1, W_fc2) + b_fc2

        return stateInput, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def trainQNetwork(self):

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory.memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextState_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        })

        # save network every 100000 iteration
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step=self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setPerception(self, observation, next_observation, action, reward, terminal):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        action_verctor = np.zeros(self.actions)
        action_verctor[action] = 1
        self.replayMemory.push_back((observation.copy(), action_verctor, reward, next_observation.copy(), terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.pop_front()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()

        # logging.info info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if self.timeStep % 1000 == 0:
            logging.error("TIMESTEP {} / STATE {} / EPSILON {}".format(self.timeStep, state, self.epsilon))

        self.timeStep += 1

    def getAction(self, observation):
        QValue = self.QValue.eval(feed_dict={self.stateInput: [observation]})[0]
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.actions)
            randomtmp = True
        else:
            action_index = np.argmax(QValue)
            randomtmp = False

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action_index, randomtmp

    def load_model(self, path):
        pass

    # def setInitState(self, observation):
    #     self.currentState = np.stack((observation, observation, observation, observation), axis=2)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
