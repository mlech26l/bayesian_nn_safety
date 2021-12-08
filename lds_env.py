import gym
from gym import spaces
import numpy as np
from os import path


class LDSEnv(gym.Env):
    def __init__(self, g=10.0):
        self.steps = None
        self.state = None
        self.has_render = False

        init = np.array([0.4, 0.4], np.float32)
        self.init_space = spaces.Box(low=-init, high=init, dtype=np.float32)
        init_train = np.array([0.75, 0.75], np.float32)
        self.init_space_train = spaces.Box(
            low=-init_train, high=init_train, dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        self.reset()

    def step(self, action):
        self.steps += 1
        x, y = self.state
        action = float(np.clip(action, -1, 1))

        new_y = y + action * 0.2
        new_x = x + new_y * 0.3 + action * 0.05

        self.state = np.array([new_x, new_y])
        crash = np.abs(new_x) > 1.0
        done = self.steps >= 200 or crash

        reward = 1.0 - np.square(new_x) - np.square(new_y)

        return self.state, reward, done, {}

    def reset(self, state=None):
        if state is None:
            state = self.init_space_train.sample()
        self.state = state
        self.steps = 0
        return self.state