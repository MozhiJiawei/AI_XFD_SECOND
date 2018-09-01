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


def eval_score_net():
    brain = BrainDQN()
    brain.epsilon = 0

    estimator = ScoreEstimator()

    maze = env.Maze(_MAP_LIST, is_show=True, is_loop=False, map_index=0)
    maze.set_score_mode()

    for i in range(1000):
        observation, key_observation, attack_value = maze.reset()
        feature = observation.copy()
        feature[:, :, env.MINE_CHANNEL] *= attack_value
        while True:
            maze.render(False)

            action, _ = brain.getAction(observation, key_observation)

            next_observation, next_key_observation, label, done, is_normal_finish = maze.move(action)

            observation = next_observation.copy()

            key_observation = next_key_observation.copy()

            if done:
                break

        logging.error("{}\t{}\t{}".format(is_normal_finish, label, estimator.eval(feature)))


def train_score_net():
    brain = BrainDQN()
    brain.epsilon = 0

    estimator = ScoreEstimator()

    maze = env.Maze(_MAP_LIST, is_show=True)
    maze.set_score_mode()

    success = 0
    failed = 0
    success_finish = 0
    failed_finish = 0
    success_abnormal = 0
    failed_abnormal = 0
    while True:
        observation, key_observation, attack_value = maze.reset()
        feature = observation.copy()
        feature[:, :, env.MINE_CHANNEL] *= attack_value
        while True:
            maze.render(False)

            action, _ = brain.getAction(observation, key_observation)

            next_observation, next_key_observation, label, done, is_normal_finish = maze.move(action)

            observation = next_observation.copy()

            key_observation = next_key_observation.copy()

            if done:
                break

        estimator.train(feature, label, is_normal_finish)
        score = estimator.eval(feature)
        # 模型可信度统计
        if abs(score - label) < 1.5:
            success += 1
            if is_normal_finish:
                success_finish += 1
            else:
                success_abnormal += 1
        else:
            failed += 1
            if is_normal_finish:
                failed_finish += 1
            else:
                failed_abnormal += 1

        if success + failed == 1000:
            logging.error("global_trust = {}%".format(success / 10))
            success = 0
            failed = 0

        if success_finish + failed_finish == 500:
            logging.error("normal trust = {}%".format(success_finish / 5))
            success_finish = 0
            failed_finish = 0

        if success_abnormal + failed_abnormal == 300:
            logging.error("abnormal_trust = {}%".format(success_abnormal / 3))
            success_abnormal = 0
            failed_abnormal = 0


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
    parser.add_argument('--eval_score', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)

    try:
        args = parser.parse_args()
        if args.train_score:
            train_score_net()
        elif args.eval_score:
            eval_score_net()
        elif args.eval:
            eval_q_net()
        elif args.train:
            main(args.debug)
    except:
        logging.exception("")
