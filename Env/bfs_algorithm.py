import env


class node:
    def __init__(self, x=0, y=0, t=0):
        self.x = x
        self.y = y
        self.t = t  # t表示走到这个格子用的步数


class father:
    def __init__(self, x=0, y=0, cz=[]):
        self.x = x  # 当前格子的父节点坐标
        self.y = y
        self.cz = cz  # 由什么操作到达的这个格子
        self.action = 0


mmap = [[0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0]]

shape_len = 12

xx = [-1, -1, 0, 1, 1, 1, 0, -1]  # 右、下、左、上
yy = [0, 1, 1, 1, 0, -1, -1, -1]


def bfs(start, dest, observation, path):
    q = []
    q.append(start)

    path[start.x][start.y].x = 1000
    path[start.x][start.y].y = 1000
    path[start.x][start.y].cz = 0

    vis = []
    for i in range(0, shape_len + 2):
        vis += [[]]
        for j in range(0, shape_len + 2):
            vis[i] += [False]

    vis[start][start] = True  # 标为已经访问过
    # print("vis={}".format(vis))
    while q:
        now = q[0]
        q.pop(0)
        for i in range(0, 8):
            new = node()
            new.x = now.x + xx[i]
            new.y = now.y + yy[i]
            new.t = now.t + 1
            if new.x < 0 or new.y < 0 or new.x >= shape_len or new.y >= shape_len or vis[new.x][new.y] == True or \
                    observation[new.x, new.y, env.WALL_CHANNEL] == 1:  # 下标越界或者访问过或者是障碍物
                continue

            q.append(new)
            path[new.x][new.y].x = now.x
            path[new.x][new.y].y = now.y
            if i == 0:
                path[new.x][new.y].cz = 'U'
            elif i == 1:
                path[new.x][new.y].cz = 'RU'
            elif i == 2:
                path[new.x][new.y].cz = 'R'
            elif i == 3:
                path[new.x][new.y].cz = 'RD'
            elif i == 4:
                path[new.x][new.y].cz = 'D'
            elif i == 5:
                path[new.x][new.y].cz = 'LD'
            elif i == 6:
                path[new.x][new.y].cz = 'L'
            elif i == 7:
                path[new.x][new.y].cz = 'LU'

            path[new.x][new.y].action = i
            vis[new.x][new.y] = True
            # print("value={} ({},{}) {}\n".format(mmap[new.x][new.y], new.x, new.y, path[new.x][new.y].cz))
            # print("=============================================================")
            if new.x == dest.x and new.y == dest.y:
                return True  # 到达终点
    return False


def dfs(x, y, start, action, path):
    if x == 0 and y == 0:
        return
    else:
        dfs(path[x][y].x, path[x][y].y, start, action, path)
    # print(lj[x][y].cz)
    action.append(path[x][y].action)


def get_path(start, dest, observation, action):
    path = []
    for i in range(0, shape_len + 2):
        path += [[]]
        for j in range(0, shape_len + 2):
            path[i] += [father()]

    if bfs(start, dest, observation, path):
        dfs(dest.x, dest.y, start, action, path)
        print("迷宫行走方式{}".format(action))
        return True
    return False


if __name__ == '__main__':
    # print(mmap)
    pass
