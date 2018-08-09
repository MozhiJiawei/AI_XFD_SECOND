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
INITIAL_EPSILON = 0.7  # 0.01 # starting value of epsilon
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

        # bran option
        self._USE_DUELING = True

        # init some parameters
        self.map_sharp = 12
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.n_action = 9
        self.n_channel = 6

        # init Q network
        with tf.variable_scope("q_eval"):
            self.stateInput, self.QValue = self.createQNetwork()
            self.eval_para = tf.trainable_variables()

        # init Target Q Network
        with tf.variable_scope("q_target"):
            self.stateInputT, self.QValueT = self.createQNetwork()
            self.tar_para = tf.trainable_variables()[len(self.eval_para):]

        self.copyTargetQNetworkOperation = [tar.assign(eval) for tar, eval in zip(self.tar_para, self.eval_para)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            logging.info("Successfully loaded:{}".format(checkpoint.model_checkpoint_path))
        else:
            logging.info("Could not find old network weights")

    def createQNetwork(self):
        state_input = tf.placeholder("float", [None, self.map_sharp, self.map_sharp, self.n_channel])

        net = tf.layers.conv2d(state_input, filters=32, kernel_size=2, activation=tf.nn.relu,
                               padding='valid')

        net = tf.layers.conv2d(net, filters=64, kernel_size=2, activation=tf.nn.relu,
                               padding='valid')

        net = tf.layers.flatten(net)

        net = tf.layers.dense(net, units=512, activation=tf.nn.relu)

        if self._USE_DUELING:
            v_net = tf.layers.dense(net, units=512, activation=tf.nn.relu)
            v = tf.layers.dense(v_net, units=1)

            a_net = tf.layers.dense(net, units=512, activation=tf.nn.relu)
            a = tf.layers.dense(a_net, units=self.n_action)

            a_mean = tf.reduce_mean(a, axis=1, keep_dims=True)
            q_value = v + (a - a_mean)
        else:
            q_value = tf.layers.dense(net, units=self.n_action)

        return state_input, q_value

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.n_action])
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
        action_verctor = np.zeros(self.n_action)
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
            action_index = random.randrange(self.n_action)
            randomtmp = True
        else:
            action_index = np.argmax(QValue)
            randomtmp = False

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action_index, randomtmp