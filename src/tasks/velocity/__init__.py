"""Velocity task type: shared MDP core plus robot profiles."""

from .core import TASK_GROUP_LAINLAB as TASK_GROUP_LAINLAB
from .core import TASK_GROUP_UNITREE as TASK_GROUP_UNITREE
from .core import ActionScale as ActionScale
from .core import CommandRange as CommandRange
from .core import CommandRanges as CommandRanges
from .core import RobotKind as RobotKind
from .core import TerrainName as TerrainName
from .core import VelocityRobotProfile as VelocityRobotProfile
from .core import VelocityScaling as VelocityScaling
from .core import humanoid_velocity_scaling as humanoid_velocity_scaling
from .core import make_flat_env_cfg as make_flat_env_cfg
from .core import make_rough_env_cfg as make_rough_env_cfg
from .core import quadruped_velocity_scaling as quadruped_velocity_scaling
from .core import register_velocity_profile as register_velocity_profile
