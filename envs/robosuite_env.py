import gym
import numpy as np

from .robosuite_wrappers import wrap_lightzero


def _sanitize_box_for_sb3_pickling(space: gym.spaces.Space) -> gym.spaces.Space:
    """
    Rebuild Box bounds as plain C-contiguous arrays so SubprocVecEnv can pickle
    observation_space/action_space (avoids NumPy 2.x / robosuite ndarray pickling bugs).
    """
    if not isinstance(space, gym.spaces.Box):
        return space
    dtype = np.dtype(space.dtype)
    low = np.array(space.low, dtype=dtype, copy=True, order="C")
    high = np.array(space.high, dtype=dtype, copy=True, order="C")
    return gym.spaces.Box(low=low, high=high, dtype=dtype.type)


class RobosuiteSlotEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config, seed=0):
        self._env = wrap_lightzero(config, seed=seed)
        self.observation_space = _sanitize_box_for_sb3_pickling(self._env.observation_space)
        self.action_space = _sanitize_box_for_sb3_pickling(self._env.action_space)
        self._seed = seed
        self.seed(seed)

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
