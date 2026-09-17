"""Velocity task definitions and registration facade."""

from .velocity_env import make_flat_env_cfg as make_flat_env_cfg
from .velocity_env import make_rough_env_cfg as make_rough_env_cfg
from .velocity_profiles import PROFILES as PROFILES
from .velocity_registry import (
  register_velocity_profile as register_velocity_profile,
)
from .velocity_registry import register_velocity_tasks as register_velocity_tasks
from .velocity_types import TASK_GROUP_LAINLAB as TASK_GROUP_LAINLAB
from .velocity_types import TASK_GROUP_UNITREE as TASK_GROUP_UNITREE
from .velocity_types import ActionScale as ActionScale
from .velocity_types import CommandRange as CommandRange
from .velocity_types import CommandRanges as CommandRanges
from .velocity_types import RobotKind as RobotKind
from .velocity_types import TerrainName as TerrainName
from .velocity_types import VelocityRobotProfile as VelocityRobotProfile
from .velocity_types import VelocityScaling as VelocityScaling
