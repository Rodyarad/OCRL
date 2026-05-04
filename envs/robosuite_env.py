import gym
import gymnasium.spaces
import numpy as np

from .robosuite_wrappers import wrap_lightzero


def _reconstruct_box_for_subproc(low, high, dtype_str):
    """Top-level helper so pickle can rebuild a Gymnasium Box in the parent process."""
    dtype = np.dtype(dtype_str)
    # NumPy < 2: asarray() has no `copy=` kwarg; use np.array(..., copy=True).
    low = np.array(low, dtype=dtype, copy=True, order="C")
    high = np.array(high, dtype=dtype, copy=True, order="C")
    return gymnasium.spaces.Box(low=low, high=high, dtype=dtype.type)


class _PickleSafeBox(gymnasium.spaces.Box):
    """
    Box whose pickle round-trip sends only bounds + dtype.

    SubprocVecEnv pickles observation_space/action_space from workers; default
    Box can carry RNG state that breaks unpickling. Stable-Baselines3 expects
    Gymnasium spaces, not legacy gym.spaces.Box.
    """

    def __init__(self, space):
        dtype = np.dtype(space.dtype)
        low = np.array(space.low, dtype=dtype, copy=True, order="C")
        high = np.array(space.high, dtype=dtype, copy=True, order="C")
        super().__init__(low=low, high=high, dtype=dtype.type)

    def __reduce__(self):
        dtype = np.dtype(self.dtype)
        low = np.array(self.low, dtype=dtype, copy=True, order="C")
        high = np.array(self.high, dtype=dtype, copy=True, order="C")
        return (_reconstruct_box_for_subproc, (low, high, dtype.str))


def _as_pickle_safe_box(space):
    if isinstance(space, (gym.spaces.Box, gymnasium.spaces.Box)):
        return _PickleSafeBox(space)
    return space


class RobosuiteSlotEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config, seed=0):
        self._env = wrap_lightzero(config, seed=seed)
        self.observation_space = _as_pickle_safe_box(self._env.observation_space)
        self.action_space = _as_pickle_safe_box(self._env.action_space)
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
