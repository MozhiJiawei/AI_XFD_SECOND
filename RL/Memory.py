import logging
import os

import numpy as np

from .sumTree import SumTree


class ReplayMemory:

    def __init__(self, mem_size, alpha, epsilon):
        self._ALPHA = alpha
        self._EPSILON = epsilon

        self._tree = SumTree(mem_size)

    def store(self, transition):
        max_priority = self._tree.max_leaf()
        if max_priority == 0:
            max_priority = self._priority(0)
        self._tree.insert(transition, max_priority)

    def sample(self, num_samples, beta):
        batch = [None] * num_samples
        IS_weights = np.zeros((num_samples,))  # Importance-sampling (IS) weights
        tree_indices = np.zeros((num_samples,), dtype=np.int32)

        len_seg = self._tree.sum() / num_samples
        min_prob = self._tree.min_leaf() / self._tree.sum()

        for i in range(num_samples):
            val = np.random.uniform(len_seg * i, len_seg * (i + 1))
            tree_indices[i], priority, batch[i] = self._tree.retrieve(val)
            prob = priority / self._tree.sum()
            IS_weights[i] = np.power(prob / min_prob, -beta)  # Simplified formula

        return batch, IS_weights, tree_indices

    def update(self, tree_indices, abs_td_errs):
        priorities = self._priority(abs_td_errs)
        for idx, priority in zip(tree_indices, priorities):
            self._tree.update(idx, priority)

    def _priority(self, abs_td_err):
        return np.power(abs_td_err + self._EPSILON, self._ALPHA)

    def dump(self, path="data"):
        observation = [data[0] for data in self._tree.data]
        action = [data[1] for data in self._tree.data]
        reward = [data[2] for data in self._tree.data]
        next_observation = [data[3] for data in self._tree.data]
        done = [data[4] for data in self._tree.data]

        if not os.path.exists(path):
            os.mkdir(path)
        np.savez(os.path.join(path, "observation"), *observation)
        np.savez(os.path.join(path, "action"), *action)
        np.savez(os.path.join(path, "reward"), *reward)
        np.savez(os.path.join(path, "next_observation"), *next_observation)
        np.savez(os.path.join(path, "done"), *done)

    def load(self, path="data"):
        try:
            observation = np.load(os.path.join(path, "observation.npz"))
            action = np.load(os.path.join(path, "action.npz"))
            reward = np.load(os.path.join(path, "reward.npz"))
            next_observation = np.load(os.path.join(path, "next_observation.npz"))
            done = np.load(os.path.join(path, "done.npz"))

            for e in observation:
                self.store((observation[e], action[e], reward[e], next_observation[e], done[e]))
        except:
            logging.exception("load failed.")
