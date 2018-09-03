# encoding: utf-8
"""
@author: lijiawei
@email: qetwe0000@gmail.com
@file: RL_Brain.py
@time: 2018/8/25 20:12
@py-version: 3.6
"""

import tensorflow as tf
import numpy as np
import random
import logging
from collections import deque
from .Memory import ReplayMemory

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.70  # decay rate of past observations
OBSERVE = 10000.  # timesteps to observe before training
EXPLORE = 5000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.001  # 0.001 # final value of epsilon
INITIAL_EPSILON = 0.7  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 40000  # number of previous transitions to remember
PRI_EPSILON = 0.001  # Small positive value to avoid zero priority
ALPHA = 0.6  # How much prioritization to use
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 100
BETA_MIN = 0.4

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
        self.epsilon = INITIAL_EPSILON
        self._beta = BETA_MIN
        self._BETA_INC = (1.0 - BETA_MIN) / EXPLORE
        self.replayMemory = ReplayMemory(mem_size=REPLAY_MEMORY,
                                         alpha=ALPHA,
                                         epsilon=PRI_EPSILON)

        # bran option
        self._USE_DUELING = True

        # init some parameters
        self.map_sharp = 12
        self.timeStep = 0
        self.n_action = 8
        self.n_channel = 3
        self._history_loss = []

        self.graph = tf.Graph()
        with self.graph.as_default():
            # init Q network
            with tf.variable_scope("q_eval"):
                self.stateInput, self.key_state_input, self.QValue = self.createQNetwork()
                self.eval_para = tf.trainable_variables()

            # init Target Q Network
            with tf.variable_scope("q_target"):
                self.stateInputT, self.key_state_inputT, self.QValueT = self.createQNetwork()
                self.tar_para = tf.trainable_variables()[len(self.eval_para):]

            self.copyTargetQNetworkOperation = [tar.assign(eval) for tar, eval in zip(self.tar_para, self.eval_para)]

            self._createTrainingMethod()

        self.session = tf.Session(graph=self.graph)
        with self.session.as_default():
            with self.graph.as_default():
                # saving and loading networks
                tf.global_variables_initializer().run()
                self.saver = tf.train.Saver()
                checkpoint = tf.train.get_checkpoint_state("saved_networks")
                if checkpoint and checkpoint.model_checkpoint_path:
                    self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                    logging.error("Successfully loaded:{}".format(checkpoint.model_checkpoint_path))
                else:
                    logging.error("Could not find old network weights")

    def createQNetwork(self):
        state_input = tf.placeholder("float", [None, self.map_sharp, self.map_sharp, self.n_channel])

        key_state_input = tf.placeholder("float", [None, self.n_action])

        net = tf.layers.conv2d(state_input, filters=32, kernel_size=2, activation=tf.nn.relu,
                               padding='same')

        net = tf.layers.conv2d(net, filters=64, kernel_size=2, activation=tf.nn.relu,
                               padding='same')

        net = tf.layers.flatten(net)

        net = tf.concat([net, key_state_input], axis=1)

        net = tf.layers.dense(net, units=512, activation=tf.nn.relu)

        net = tf.layers.dense(net, units=768, activation=tf.nn.relu)

        if self._USE_DUELING:
            v_net = tf.layers.dense(net, units=1024, activation=tf.nn.relu)
            # v_net = tf.layers.dropout(v_net)
            v = tf.layers.dense(v_net, units=1)

            a_net = tf.layers.dense(net, units=1024, activation=tf.nn.relu)
            # a_net = tf.layers.dropout(a_net)
            a = tf.layers.dense(a_net, units=self.n_action)

            a_mean = tf.reduce_mean(a, axis=1, keep_dims=True)
            q_value = v + (a - a_mean)
        else:
            q_value = tf.layers.dense(net, units=self.n_action)

        return state_input, key_state_input, q_value

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def _createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.n_action])
        self._IS_weights = tf.placeholder(tf.float32, [None, ], name="IS_weights")
        self.yInput = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1)
        self.td_err_abs = tf.square(self.yInput - Q_Action)
        self.cost = tf.reduce_mean(self._IS_weights * self.td_err_abs)
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def trainQNetwork(self):

        # Step 1: obtain random minibatch from replay memory
        # minibatch = random.sample(self.replayMemory.memory, BATCH_SIZE)
        batch, IS_weights, tree_indices = self.replayMemory.sample(BATCH_SIZE, self._beta)
        state_batch = [data[0] for data in batch]
        action_batch = [data[1] for data in batch]
        reward_batch = [data[2] for data in batch]
        nextState_batch = [data[3] for data in batch]
        key_state_batch = [data[5] for data in batch]
        next_key_state_batch = [data[6] for data in batch]

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextState_batch,
                                                    self.key_state_inputT: next_key_state_batch})
        for i in range(0, BATCH_SIZE):
            terminal = batch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        _, cost, abs_errs = self.session.run(
            [self.trainStep, self.cost, self.td_err_abs],
            feed_dict={
                self.yInput: y_batch,
                self.actionInput: action_batch,
                self.stateInput: state_batch,
                self.key_state_input: key_state_batch,
                self._IS_weights: IS_weights
            })

        self._history_loss.append(cost)

        self.replayMemory.update(tree_indices, abs_errs)

        if len(self._history_loss) == 1000:
            logging.error("average loss = {}".format(np.mean(self._history_loss)))
            self._history_loss = []

        self._beta = min(1.0, self._beta + self._BETA_INC)

        # save network every 100000 iteration
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step=self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setPerception(self, observation, key_observation, next_observation, next_key_observation, action, reward,
                      terminal):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        action_verctor = np.zeros(self.n_action)
        action_verctor[action] = 1
        self.replayMemory.store((observation.copy(),
                                 action_verctor,
                                 reward,
                                 next_observation.copy(),
                                 terminal,
                                 key_observation.copy(),
                                 next_key_observation.copy())
                                )

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

    def getAction(self, observation, key_observation):
        with self.session.as_default():
            QValue = self.QValue.eval(feed_dict={self.stateInput: [observation],
                                                 self.key_state_input: [key_observation]})[0]
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
