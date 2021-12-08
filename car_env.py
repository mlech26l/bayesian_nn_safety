import gym
from gym import spaces
import numpy as np
from os import path


class CarEnv(gym.Env):
    def __init__(self):
        self.steps = None
        self.state = None

        self.bound_x = 2
        self.bound_y = 5
        self.speed_y = 1
        self.speed_x = 1
        self.safety_ball = 1
        self.has_render = False

        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.ones(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )
        self.reset()

    def step(self, action):
        self.steps += 1
        px, ax, ay = self.state

        if action == 2:
            action = -1

        px += action * self.speed_x
        px = np.clip(px, -self.bound_x, self.bound_x)

        ay -= self.speed_y

        self.state = np.array([px, ax, ay])

        done = self.steps >= 50
        reward = 0
        reward += 0.1 * np.abs(px - ax)

        if ay <= 0:
            done = True
            if np.abs(px - ax) <= self.safety_ball:
                reward -= 100
        return self.state, reward, done, {}

    def reset(self, state=None, seed=None):
        if state is None:
            rng = (
                np.random.default_rng() if seed is None else np.random.default_rng(seed)
            )
            px = rng.integers(-self.bound_x, self.bound_x + 1)
            ax = rng.integers(-self.bound_x, self.bound_x + 1)
            ay = rng.integers(self.bound_y - 1, self.bound_y + 1)
            # ay = self.bound_y

            state = np.array([px, ax, ay])
        self.state = state
        self.steps = 0
        return self.state
