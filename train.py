import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

import Env.env as env
from setup_logging import setup_logging
from RL.Score_Estimator import ScoreEstimator
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


def train_score_net():
    brain = BrainDQN()
    brain.epsilon = 0

    estimator = ScoreEstimator()

    maze = env.Maze(_MAP_LIST, is_show=True)
    maze.set_score_mode()

    while True:
        observation, key_observation, attack_value = maze.reset()
        feature = observation.copy()
        feature[:, :, env.MINE_CHANNEL] *= attack_value
        while True:
            maze.render(False)

            action, _ = brain.getAction(observation, key_observation)

            next_observation, next_key_observation, label, done = maze.move(action)

            observation = next_observation.copy()

            key_observation = next_key_observation.copy()

            if done:
                break

        estimator.train(feature, label)


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
        maze.effective_epsilon = 1
        for i in range(1001):
            observation, key_observation = maze.reset()
            while True:
                maze.render(False)

                action, is_random = brain.getAction(observation, key_observation)

                next_observation, next_key_observation, r, done = maze.step(action, is_random)

                observation = next_observation.copy()

                key_observation = next_key_observation.copy()

                if done:
                    break


def main(is_debug):
    brain = BrainDQN()
    if is_debug:
        brain.epsilon = 0

    maze = env.Maze(_MAP_LIST, is_show=True)

    while True:
        observation, key_observation, _ = maze.reset()
        while True:
            maze.render(is_debug)

            action, is_random = brain.getAction(observation, key_observation)

            next_observation, next_key_observation, r, done = maze.step(action, is_random)

            brain.setPerception(observation, key_observation, next_observation, next_key_observation, action, r, done)

            observation = next_observation.copy()

            key_observation = next_key_observation.copy()

            if done:
                break


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--train_score', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)

    try:
        args = parser.parse_args()
        if args.eval:
            eval_q_net()
        if args.train_score:
            train_score_net()
        elif args.train:
            main(args.debug)
    except:
        logging.exception("")
