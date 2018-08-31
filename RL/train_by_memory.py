from RL_Brain import BrainDQN

STEP = 1000


def main():
    brain = BrainDQN()

    brain.replayMemory.load(path="data")

    for i in range(STEP):
        brain._trainQNetwork()


if __name__ == "__main__":
    main()
