from collections import deque
import logging

import numpy as np
import os


class ReplayMemory(object):
    def __init__(self):
        """
        记忆库存储格式：
        observation
        action
        reward
        next_observation
        done
        """
        self.memory = deque()

    def __len__(self):
        return len(self.memory)

    def push_back(self, obj):
        self.memory.append(obj)

    def pop_front(self):
        self.memory.popleft()

    def dump(self, path="data"):
        observation = [data[0] for data in self.memory]
        action = [data[1] for data in self.memory]
        reward = [data[2] for data in self.memory]
        next_observation = [data[3] for data in self.memory]
        done = [data[4] for data in self.memory]

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
                self.memory.append((observation[e], action[e], reward[e], next_observation[e], done[e]))
        except:
            logging.exception("load failed.")


if __name__ == "__main__":
    pass
