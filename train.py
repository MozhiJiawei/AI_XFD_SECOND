import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

import Env.env as env
from setup_logging import setup_logging
from RL.Memory import ReplayMemory
from RL.RL_Brain import BrainDQN

_MAP_LIST = [np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),

             np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),

             np.array([[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                       [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                       [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
                       [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                       [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                       [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
                       [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                       [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]]),

             np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),

             np.array([[0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
                       [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                       [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                       [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                       [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                       [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
                       [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                       [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]]),

             np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                       [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])]


def gen_eval_data_manually():
    maze = env.Maze(_MAP1, is_show=True)

    key_map = {
        "1": env.LEFT_DOWN,
        "2": env.DOWN,
        "3": env.RIGHT_DOWN,
        "4": env.LEFT,
        "5": env.STOP,
        "6": env.RIGHT,
        "7": env.LEFT_UP,
        "8": env.UP,
        "9": env.RIGHT_UP
    }

    memory = ReplayMemory()
    memory.load("eval_data")

    done = True
    while True:
        if done:
            observation = maze.reset()
        s = input('input your path. Enter "exit" to close\n')
        if s == "exit":
            break

        next_observation, r, done = maze.step(key_map[s], True)

        memory.push_back((observation.copy(), key_map[s], r, next_observation.copy(), done))

        observation = next_observation.copy()

        maze.render(is_sleep=True)

    memory.dump("eval_data")


def eval_q_net():
    """
    连跑训练地图各5000次
    日志输出单步成功率，回合成功率
    :return:
    """
    brain = BrainDQN()
    brain.epsilon = 0

    for index in range(len(_MAP_LIST)):
        maze = env.Maze(_MAP_LIST, is_show=True, is_loop=False, map_index=index)
        for i in range(1001):
            observation = maze.reset()
            while True:
                maze.render(False)
                action, is_random = brain.getAction(observation)
                next_observation, r, done = maze.step(action, is_random)
                observation = next_observation.copy()
                if done:
                    break


def main(is_debug):
    brain = BrainDQN()
    if is_debug:
        brain.epsilon = 0

    maze = env.Maze(_MAP_LIST, is_show=is_debug)

    while True:
        observation = maze.reset()
        while True:
            maze.render(is_debug)
            action, is_random = brain.getAction(observation)
            next_observation, r, done = maze.step(action, is_random)
            brain.setPerception(observation, next_observation, action, r, done)
            observation = next_observation.copy()

            if done:
                break


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--human', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)

    try:
        args = parser.parse_args()
        if args.human:
            gen_eval_data_manually()
        if args.eval:
            eval_q_net()
        elif args.train:
            main(args.debug)
    except:
        logging.exception("")
