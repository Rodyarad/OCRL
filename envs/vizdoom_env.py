import gym

from .vizdoom_wrappers import wrap_lightzero


class VizdoomEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config, seed=0):
        self._env = wrap_lightzero(config)
        self._seed = seed
        self.seed(seed)

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def seed(self, seed=None):
        self._seed = seed
        if hasattr(self._env, "seed"):
            self._env.seed(seed)
        if hasattr(self.action_space, "seed"):
            self.action_space.seed(seed)
        return seed

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def render(self, mode=None):
        return self._env.render(mode=mode)
