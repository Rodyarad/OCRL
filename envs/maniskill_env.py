import gym
from packaging import version
import stable_baselines3 as sb3

from .maniskill_wrappers import wrap_lightzero


_SB3_GYMNASIUM_API = version.parse(sb3.__version__) >= version.parse("2.0.0")


class ManiSkillEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config, seed=0):
        self._env = wrap_lightzero(config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._seed = seed
        self.seed(seed)

    def seed(self, seed=None):
        self._seed = seed
        if hasattr(self._env, "seed"):
            self._env.seed(seed)
        if hasattr(self.action_space, "seed"):
            self.action_space.seed(seed)
        return seed

    def reset(self, seed=None, options=None, **kwargs):
        try:
            obs = self._env.reset(seed=seed, options=options, **kwargs)
        except TypeError:
            if seed is not None:
                self.seed(seed)
            obs = self._env.reset()
        return (obs, {}) if _SB3_GYMNASIUM_API else obs

    def step(self, action):
        out = self._env.step(action)
        if _SB3_GYMNASIUM_API:
            if len(out) == 5:
                return out
            obs, reward, done, info = out
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(done) and not truncated
            return obs, reward, terminated, truncated, info
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, reward, done, info
        return out

    def render(self, mode=None):
        return self._env.render(mode=mode)
