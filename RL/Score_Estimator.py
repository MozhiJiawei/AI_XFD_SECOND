# encoding: utf-8
"""
@author: lijiawei
@email: qetwe0000@gmail.com
@file: Score_Estimator.py
@time: 2018/8/28 22:59
@py-version: 3.6
@describe: 根据当前目标情况进行估分
"""
import random
import logging
from collections import deque

import tensorflow as tf
import numpy as np


class ScoreEstimator:
    def __init__(self):
        self.n_map = 12
        self.n_channel = 3

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.feature, self.score = self._build_graph()
            self.label, self.loss, self.trainer = self._build_trainer()

        self.global_step = 0
        self._history_loss = []

        # Memory
        self.memory = deque()
        self.MEMORY_SIZE = 10000
        self.BATCH_SIZE = 15
        self.SINGLE_STEP = 20

        # 重要记忆，模型异常（走不完）时的记忆
        self.important_memory = deque()
        self.IMPORTANT_MEMORY_SIZE = 2000
        self.IMPORTANT_BATCH_SIZE = 15

        self.session = tf.Session(graph=self.graph)
        with self.session.as_default():
            with self.graph.as_default():
                # saving and loading networks
                tf.global_variables_initializer().run()
                self.saver = tf.train.Saver()
                checkpoint = tf.train.get_checkpoint_state("score_network")
                if checkpoint and checkpoint.model_checkpoint_path:
                    self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                    logging.error("Successfully loaded:{}".format(checkpoint.model_checkpoint_path))
                else:
                    logging.error("Could not find old network weights")
                pass

    def train(self, feature, label, is_normal_finish):
        """
        训练估分网络，模型非法时的记忆标记为特殊
        :param feature: 特征 [12, 12, 3]
        :param label: 标签
        :param is_normal_finish: 模型是否正常结束
        :return: None
        """
        if len(self.memory) < self.BATCH_SIZE or len(self.important_memory) < self.IMPORTANT_BATCH_SIZE:
            if is_normal_finish:
                self.memory.append((feature.copy(), label))
            else:
                self.important_memory.append((feature.copy(), label))
            return

        if is_normal_finish:
            self.memory.append((feature.copy(), label))
        else:
            self.important_memory.append((feature.copy(), label))

        step = 1
        if len(self.memory) > self.MEMORY_SIZE:
            self.memory.popleft()
            step = self.SINGLE_STEP

        if len(self.memory) > self.IMPORTANT_MEMORY_SIZE:
            self.memory.popleft()
            step = self.SINGLE_STEP

        for i in range(step):
            memory_batch = random.sample(self.memory, self.BATCH_SIZE)
            features = [d[0] for d in memory_batch]
            labels = [d[1] for d in memory_batch]

            important_memory_batch = random.sample(self.important_memory, self.IMPORTANT_BATCH_SIZE)
            features.extend([d[0] for d in important_memory_batch])
            labels.extend([d[1] for d in important_memory_batch])

            labels = np.reshape(labels, newshape=(self.BATCH_SIZE + self.IMPORTANT_BATCH_SIZE, 1))

            loss, _ = self.session.run(
                [self.loss, self.trainer],
                feed_dict={
                    self.feature: features,
                    self.label: labels
                }
            )
            self._history_loss.append(loss)
            self.global_step += 1

            if len(self._history_loss) == 1000:
                logging.error("average loss = {}, step = {}".format(np.mean(self._history_loss), self.global_step))
                self._history_loss = []

            if self.global_step % 10000 == 0:
                self.saver.save(self.session, 'score_network/' + 'network' + '-score', global_step=self.global_step)

    def eval(self, feature):
        """
        根据当前状态进行估分
        :param feature:
        :return: score
        """
        with self.session.as_default():
            score = self.score.eval(feed_dict={
                self.feature: [feature]
            })[0][0]

        return score

    def _build_graph(self):
        feature = tf.placeholder("float", [None, self.n_map, self.n_map, self.n_channel])

        net = tf.layers.conv2d(feature, filters=32, kernel_size=2, activation=tf.nn.relu,
                               padding='same')

        net = tf.layers.conv2d(net, filters=64, kernel_size=2, activation=tf.nn.relu,
                               padding='same')

        net = tf.layers.flatten(net)

        net = tf.layers.dense(net, units=512, activation=tf.nn.relu)

        net = tf.layers.dense(net, units=1024, activation=tf.nn.relu)

        score = tf.layers.dense(net, units=1)

        return feature, score

    def _build_trainer(self):
        label = tf.placeholder("float", [None, 1])

        loss = tf.losses.mean_squared_error(label, self.score)

        trainer = tf.train.AdamOptimizer(1e-6).minimize(loss)

        return label, loss, trainer
