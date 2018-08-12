"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import time
import sys
import logging

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

WALL_CHANNEL = 0
MINE_CHANNEL = 1
ENEMY_CHANNEL = 2
TREASURE_CHANNEL = 3
PATH_CHANNEL = 4
POISON_CHANNEL = 5

UP = 0
RIGHT_UP = 1
RIGHT = 2
RIGHT_DOWN = 3
DOWN = 4
LEFT_DOWN = 5
LEFT = 6
LEFT_UP = 7
STOP = 8

UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Pos(object):
    def __init__(self, board=12):
        self.board = board
        self.x = 0
        self.y = 0
        self.random_init()

    def __copy__(self):
        p = Pos()
        p.x = self.x
        p.y = self.y
        return p

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return "x[{}], y[{}]".format(self.x, self.y)

    def distance(self, other):
        return np.linalg.norm(np.array([self.x, self.y]) - np.array([other.x, other.y]))

    def random_init(self):
        self.x = np.random.randint(0, self.board)
        self.y = np.random.randint(0, self.board)
        while not self.is_valid():
            self.x = np.random.randint(0, self.board)
            self.y = np.random.randint(0, self.board)

    def move(self, action):
        if action == UP:
            self.x -= 1
        if action == RIGHT_UP:
            self.x -= 1
            self.y += 1
        if action == RIGHT:
            self.y += 1
        if action == RIGHT_DOWN:
            self.x += 1
            self.y += 1
        if action == DOWN:
            self.x += 1
        if action == LEFT_DOWN:
            self.y -= 1
            self.x += 1
        if action == LEFT:
            self.y -= 1
        if action == LEFT_UP:
            self.y -= 1
            self.x -= 1
        if action == 8:
            pass

    def is_valid(self):
        if 0 <= self.x < self.board and 0 <= self.y < self.board:
            return True
        else:
            return False


class Maze(tk.Tk, object):
    def __init__(self, map_in, is_show=False, is_loop=True, map_index=0):
        """
        :param map_in: list[ np.array(12, 12) * 6]
        :param is_show: 是否显示TK界面
        :param is_loop: 是否循环训练
        :param map_index: 选择训练的地图
        """
        super(Maze, self).__init__()
        self.is_show = is_show

        # 地图信息
        self.n_action = 9
        self.n_map = 12
        self.n_channel = 6
        self.map_index = map_index  # 当前地图索引
        self.is_loop = is_loop  # 是否循环训练
        self.loop_step = 0  # 当前地图已经训练了几次
        self.loop_count = 500  # 训练几次更新地图
        self.map_input = map_in

        self.observation = np.zeros((self.n_map, self.n_map, self.n_channel))

        self._WITH_WALL = True
        self._WITH_TREASURE = True
        self._WITH_STOP_ACTION = False

        self.player = Pos()
        self.last_player = Pos()
        self.enemy = Pos()
        self.enemy_count = 0
        if self._WITH_TREASURE:
            self.treasure_1 = Pos()
            self.treasure_2 = Pos()

        self.reward_sum = 0

        self.success_time = [0 for i in range(len(self.map_input))]
        self.failed_time = [0 for i in range(len(self.map_input))]

        if self.is_show:
            self.player_ret = None
            self.enemy_ret = None
            self.t1_ret = None
            self.t2_ret = None
            self.wall_list = []
            self._init_tk()

        self.reset()

    def _init_tk(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=self.n_map * UNIT,
                                width=self.n_map * UNIT)
        for c in range(0, self.n_map * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.n_map * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.n_map * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.n_map * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        self._init_wall()
        self.wall_list = []
        for i in range(self.map_input[self.map_index].shape[0]):
            for j in range(self.map_input[self.map_index].shape[1]):
                if self.observation[i, j, WALL_CHANNEL] != 0 and (i != self.enemy.x or j != self.enemy.y):
                    self.wall_list.append(self._create_rectangle(i, j, "black", is_ret=True))

        self.enemy_ret = self._create_rectangle(self.enemy.x, self.enemy.y, "yellow")
        self.player_ret = self._create_rectangle(self.player.x, self.player.y, "red")
        if self._WITH_TREASURE:
            self.t1_ret = self._create_rectangle(self.treasure_1.x, self.treasure_1.y, "green")
            self.t2_ret = self._create_rectangle(self.treasure_2.x, self.treasure_2.y, "green")

        self.canvas.pack()
        self.update()

    def reset(self):
        self._init_wall()
        self._init_player()
        self._init_enemy()
        self._init_path()
        # self._init_poison()
        if self._WITH_TREASURE:
            self._init_treasure()
        self.reward_sum = 0

        if self.is_show:
            self._reset_tk()

        return self.observation

    def step(self, action, is_random):
        """
        当前规则描述如下：
        墙： -1
        瞎停： -1
        砍人： 0.5
        捡宝藏： 0.5
        停对位置：周边墙数量 * 0.125

        :param action:
        :param is_random:
        :return:
        """
        reward = 0
        done = False

        self.last_player = self.player.__copy__()
        self.player.move(action)

        # 走错路
        if not self.player.is_valid() or \
                self.observation[self.player.x, self.player.y, WALL_CHANNEL] != 0 or \
                self.observation[self.player.x, self.player.y, PATH_CHANNEL] != 0:
            reward = -1
            done = True
            if not is_random:
                self.failed_time[self.map_index] += 1

            if not self.player.is_valid():
                logging.info("out of board!!!")
            elif self.observation[self.player.x, self.player.y, WALL_CHANNEL] != 0:
                logging.info("hit wall!!!")
            elif self.observation[self.player.x, self.player.y, PATH_CHANNEL] != 0:
                logging.info("repeat path!!!")

            self.reset()
            return self.observation, reward, done

        if self.is_show:
            self.canvas.delete(self.player_ret)
            self.player_ret = self._create_rectangle(self.player.x, self.player.y, "red")
            self.update()

        # 尝试停止
        if self._WITH_STOP_ACTION:
            if action == STOP:
                logging.info("try stop!!!")
                done = True
                # 没有砍到人就停止，瞎搞！
                if self.reward_sum == 0:
                    if not is_random:
                        self.failed_time[self.map_index] += 1

                    reward = -1
                    self.reset()
                    return self.observation, reward, done

                # 停在正确的位置需要加分
                for i in range(0, 8, 2):
                    reward += self._try_stop(self.player, i) * 0.125

                # 停在敌人旁边要扣分
                if self.player.distance(self.enemy) == 1:
                    reward -= 0.25

        self.observation[self.last_player.x, self.last_player.y, MINE_CHANNEL] = 0
        self.observation[self.last_player.x, self.last_player.y, PATH_CHANNEL] = 1
        self.observation[self.player.x, self.player.y, MINE_CHANNEL] = 1

        if self.observation[self.player.x, self.player.y, ENEMY_CHANNEL] != 0:
            logging.info("kill someone!!!")
            reward += 1
            self.reward_sum += reward
            self.observation[self.player.x, self.player.y, ENEMY_CHANNEL] = 0
            self.enemy_count -= 1
            if self.enemy_count == 0:
                logging.error("finish!!!!!! map_index = {}".format(self.map_index))
                done = True
        else:
            pass
            # 根据和敌人之间的距离计算reward，距离越近reward越大
            # dis = self.player.distance(self.enemy)
            # reward += (0.5 / (dis*dis))

        # 去除TREASURE CHANNEL，与ENEMY统一判断即可
        # if self._WITH_TREASURE:
        #     if self.player.distance(self.treasure_1) == 0:
        #         reward += 0.5
        #         self.observation[self.player.x, self.player.y, TREASURE_CHANNEL] -= 1
        #         logging.info("pick treasure!!!")
        #
        #     if self.player.distance(self.treasure_2) == 0:
        #         reward += 0.5
        #         self.observation[self.player.x, self.player.y, TREASURE_CHANNEL] -= 1
        #         logging.info("pick treasure!!!")

        if not is_random:
            self.success_time[self.map_index] += 1

        if self.success_time[self.map_index] + self.failed_time[self.map_index] >= 1000:
            logging.error("success rat = {} map_index = {}".format(
                self.success_time[self.map_index] / (
                        self.success_time[self.map_index] + self.failed_time[self.map_index]),
                self.map_index))
            self.success_time[self.map_index] = 0
            self.failed_time[self.map_index] = 0

        logging.info("reward = {}".format(reward))
        return self.observation, reward, done

    def set_observation(self, observation):
        """
        评估时，设置当前状态
        :param observation:
        :return:
        """
        self.observation = observation.copy()

        has_player = False
        has_enemy = False
        for i in range(self.n_map):
            for j in range(self.n_map):
                if self.observation[i, j, MINE_CHANNEL] == 1:
                    self.player.x = i
                    self.player.y = j
                    has_player = True
                if self.observation[i, j, ENEMY_CHANNEL] == 1:
                    self.enemy.x = i
                    self.enemy.y = j
                    has_enemy = True
                # 去除TREASURE CHANNEL
                # if self.observation[i, j, TREASURE_CHANNEL] == 1:
                #     self.treasure_1.x = i
                #     self.treasure_1.y = j
                #     has_treasure_1 = True
                # if has_treasure_1 and self.observation[i, j, TREASURE_CHANNEL] == 1:
                #     self.treasure_2.x = i
                #     self.treasure_2.y = j

        if not has_player or not has_enemy:
            logging.error("player")
            logging.error(observation[:, :, MINE_CHANNEL])
            logging.error("enemy")
            logging.error(observation[:, :, ENEMY_CHANNEL])
            raise ValueError("input observation has no player or enemy!!!")

    def render(self, is_sleep=False):
        if is_sleep:
            time.sleep(0.3)
        self.update()

    def _try_stop(self, player, action):
        tmp = player.__copy__()
        tmp.move(action)
        if not tmp.is_valid() or self.observation[tmp.x, tmp.y, WALL_CHANNEL] != 0:
            return 1
        else:
            return 0

    def _reset_tk(self):
        self.canvas.delete(self.enemy_ret)
        self.canvas.delete(self.player_ret)
        self.canvas.delete(self.t1_ret)
        self.canvas.delete(self.t2_ret)
        for wall in self.wall_list:
            self.canvas.delete(wall)

        self.wall_list = []
        for i in range(self.map_input[self.map_index].shape[0]):
            for j in range(self.map_input[self.map_index].shape[1]):
                if self.observation[i, j, WALL_CHANNEL] != 0 and (i != self.enemy.x or j != self.enemy.y):
                    self.wall_list.append(self._create_rectangle(i, j, "black", is_ret=True))

        self.enemy_ret = self._create_rectangle(self.enemy.x, self.enemy.y, "yellow")
        self.player_ret = self._create_rectangle(self.player.x, self.player.y, "red")
        if self._WITH_TREASURE:
            self.t1_ret = self._create_rectangle(self.treasure_1.x, self.treasure_1.y, "green")
            self.t2_ret = self._create_rectangle(self.treasure_2.x, self.treasure_2.y, "green")

        self.canvas.pack()
        self.update()

    def _create_rectangle(self, x, y, color, is_ret=None):
        origin = np.array([20, 20])
        enemy_center = origin + np.array([UNIT * y, UNIT * x])
        if is_ret is None:
            return self.canvas.create_oval(
                enemy_center[0] - 15, enemy_center[1] - 15,
                enemy_center[0] + 15, enemy_center[1] + 15,
                fill=color)
        else:
            return self.canvas.create_rectangle(
                enemy_center[0] - 15, enemy_center[1] - 15,
                enemy_center[0] + 15, enemy_center[1] + 15,
                fill=color)

    def _init_wall(self):
        if self.is_loop and self.loop_step > self.loop_count:
            self.loop_step = 0
            self.map_index = (self.map_index + 1) % len(self.map_input)
        elif self.is_loop:
            self.loop_step += 1

        for i in range(self.map_input[self.map_index].shape[0]):
            for j in range(self.map_input[self.map_index].shape[1]):
                if self._WITH_WALL:
                    self.observation[i, j, WALL_CHANNEL] = self.map_input[self.map_index][i, j]
                else:
                    self.observation[i, j, WALL_CHANNEL] = 0

    def _init_player(self):
        self.player.random_init()
        while (self.observation[self.player.x, self.player.y, WALL_CHANNEL] == 1) or (
                self.observation[self.player.x, self.player.y, ENEMY_CHANNEL] == 1):
            self.player.random_init()
        self.observation[:, :, MINE_CHANNEL] = np.zeros((12, 12))
        self.observation[self.player.x, self.player.y, MINE_CHANNEL] = 1

    def _init_enemy(self):
        self.enemy_count = 0
        self.enemy.random_init()
        while (self.observation[self.enemy.x, self.enemy.y, WALL_CHANNEL] == 1) or \
                (self.observation[self.enemy.x, self.enemy.y, MINE_CHANNEL] == 1):
            self.enemy.random_init()

        while (self.observation[self.enemy.x, self.enemy.y, WALL_CHANNEL] == 1) or (
                self.observation[self.enemy.x, self.enemy.y, MINE_CHANNEL] == 1):
            self.enemy.random_init()
        self.observation[:, :, ENEMY_CHANNEL] = np.zeros((12, 12))

        # 敌人上面位置 后面改成毒气层
        enemy_tmp = Pos()
        enemy_tmp.x = self.enemy.x - 1
        enemy_tmp.y = self.enemy.y

        if enemy_tmp.is_valid() and \
                self.observation[enemy_tmp.x, enemy_tmp.y, WALL_CHANNEL] != 1 and \
                self.observation[enemy_tmp.x, enemy_tmp.y, PATH_CHANNEL] != 1 and \
                self.observation[enemy_tmp.x, enemy_tmp.y, MINE_CHANNEL] != 1:
            self.observation[enemy_tmp.x, enemy_tmp.y, ENEMY_CHANNEL] = 1
            self.enemy_count += 1

        # 敌人右面位置 后面改成毒气层
        enemy_tmp.x = self.enemy.x
        enemy_tmp.y = self.enemy.y + 1
        if enemy_tmp.is_valid() and \
                self.observation[enemy_tmp.x, enemy_tmp.y, WALL_CHANNEL] != 1 and \
                self.observation[enemy_tmp.x, enemy_tmp.y, PATH_CHANNEL] != 1 and \
                self.observation[enemy_tmp.x, enemy_tmp.y, MINE_CHANNEL] != 1:
            self.observation[enemy_tmp.x, enemy_tmp.y, ENEMY_CHANNEL] = 1
            self.enemy_count += 1

        # 敌人下面位置 后面改成毒气层
        enemy_tmp.x = self.enemy.x + 1
        enemy_tmp.y = self.enemy.y
        if enemy_tmp.is_valid() and \
                self.observation[enemy_tmp.x, enemy_tmp.y, WALL_CHANNEL] != 1 and \
                self.observation[enemy_tmp.x, enemy_tmp.y, PATH_CHANNEL] != 1 and \
                self.observation[enemy_tmp.x, enemy_tmp.y, MINE_CHANNEL] != 1:
            self.observation[enemy_tmp.x, enemy_tmp.y, ENEMY_CHANNEL] = 1
            self.enemy_count += 1

        # 敌人左面位置 后面改成毒气层
        enemy_tmp.x = self.enemy.x
        enemy_tmp.y = self.enemy.y - 1
        if enemy_tmp.is_valid() and \
                self.observation[enemy_tmp.x, enemy_tmp.y, WALL_CHANNEL] != 1 and \
                self.observation[enemy_tmp.x, enemy_tmp.y, PATH_CHANNEL] != 1 and \
                self.observation[enemy_tmp.x, enemy_tmp.y, MINE_CHANNEL] != 1:
            self.observation[enemy_tmp.x, enemy_tmp.y, ENEMY_CHANNEL] = 1
            self.enemy_count += 1

        # 把敌人位置置为墙
        self.observation[self.enemy.x, self.enemy.y, WALL_CHANNEL] = 1

    def _init_treasure(self):
        self.treasure_1.random_init()
        while (self.observation[self.treasure_1.x, self.treasure_1.y, WALL_CHANNEL] == 1) or \
                (self.observation[self.treasure_1.x, self.treasure_1.y, MINE_CHANNEL] == 1):
            self.treasure_1.random_init()

        self.treasure_2.random_init()
        while (self.observation[self.treasure_2.x, self.treasure_2.y, WALL_CHANNEL] == 1) or \
                (self.observation[self.treasure_2.x, self.treasure_2.y, MINE_CHANNEL] == 1):
            self.treasure_2.random_init()

        # 去除TREASURE CHANNEL
        # self.observation[:, :, TREASURE_CHANNEL] = np.zeros((12, 12))
        if self.observation[self.treasure_1.x, self.treasure_1.y, ENEMY_CHANNEL] == 0:
            self.observation[self.treasure_1.x, self.treasure_1.y, ENEMY_CHANNEL] = 1
            self.enemy_count += 1

        if self.observation[self.treasure_2.x, self.treasure_2.y, ENEMY_CHANNEL] == 0:
            self.observation[self.treasure_2.x, self.treasure_2.y, ENEMY_CHANNEL] = 1
            self.enemy_count += 1

    def _init_path(self):
        self.observation[:, :, PATH_CHANNEL] = np.zeros((12, 12))
