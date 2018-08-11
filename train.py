import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

import Env.env as env
from setup_logging import setup_logging
from RL.Memory import ReplayMemory
from RL.RL_Brain import BrainDQN

_MAP1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


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
    maze = env.Maze(_MAP1, is_show=True)

    memory = ReplayMemory()
    memory.load("eval_data")

    brain = BrainDQN()

    # 评估时不随机
    brain.epsilon = 0
    r_list = []

    for data in memory.memory:
        s, action, r, s_, done = data[0], data[1], data[2], data[3], data[4]

        action, _ = brain.getAction(s)

        maze.set_observation(s)

        _, r_eval, done_eval = maze.step(action, False)

        r_list.append(r_eval - r)

    logging.info(sum(r_list) / len(r_list))
    plt.plot(np.arange(len(r_list)), r_list)
    plt.ylabel('reward loss')
    plt.xlabel('sample')
    plt.show()


def main(is_debug):
    brain = BrainDQN()
    if is_debug:
        brain.epsilon = 0

    maze = env.Maze(_MAP1, is_show=is_debug)

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
