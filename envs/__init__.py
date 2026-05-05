from .synthetic_envs import RandomObjsEnv, OddOneOutEnv, TargetEnv, PushEnv, MazeEnv
from .cw_envs import CwTargetEnv
from .vizdoom_env import VizdoomEnv
from .maniskill_env import ManiSkillEnv

# Keep robosuite optional so non-robosuite runs do not import DINOSAUR dependencies.
try:
    from .robosuite_env import RobosuiteSlotEnv
except Exception:
    pass
