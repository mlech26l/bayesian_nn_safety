import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


def angular(x):
    x = float(x)
    while x < np.pi / 2:
        x += 2 * np.pi
    while x > 5 * np.pi / 2:
        x -= 2 * np.pi
    if x > 3 * np.pi / 2:
        return (x - 2 * np.pi) / (np.pi / 2)
    if x > np.pi / 2:
        return 2 - x / (np.pi / 2)
    return x / (np.pi / 2)


class PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0):
        self.has_render = True
        self.max_speed = 8
        self.max_torque = 2.5
        self.dt = 0.05
        self.g = g
        self.m = 0.8  # 1.0
        self.l = 1.0
        self.steps = 0
        self.viewer = None

        init = np.array([np.pi / 6, 0.2], np.float32)
        self.init_space = spaces.Box(low=-init, high=init, dtype=np.float32)
        init_train = np.array([np.pi / 3, 0.4], np.float32)
        self.init_space_train = spaces.Box(
            low=-init_train, high=init_train, dtype=np.float32
        )

        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self.steps += 1
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -1, 1)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        newthdot = (
            thdot
            + (
                -3 * g / (2 * l) * angular(th + np.pi)
                + 3.0 / (m * l ** 2) * self.max_torque * u
            )
            * dt
        )
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        # print("state: ", self.state)
        done = self.steps >= 200 or np.abs(newth) > np.pi / 2
        return self._get_obs(), -costs + 1, done, {}

    def reset(self, state=None):
        if state is None:
            state = self.init_space_train.sample()
        self.state = state
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot])

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


if __name__ == "__main__":
    env = PendulumEnv()
    done = False
    x = env.reset()
    total_reward = 0
    step = 0
    while not done:
        x = env.action_space.sample()
        x, r, done, info = env.step(x)
        env.render()
        total_reward += r
        step += 1
