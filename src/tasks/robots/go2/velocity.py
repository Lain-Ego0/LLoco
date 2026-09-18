"""Go2 velocity profile."""

from src.assets.robots import get_go2_robot_cfg
from src.tasks.robots.common import _QUAD_FEET, _QUAD_GEOMS
from src.tasks.velocity import (
  TASK_GROUP_UNITREE,
  VelocityRobotProfile,
  quadruped_velocity_scaling,
)

GO2_PROFILE = VelocityRobotProfile(
  "Go2",
  get_go2_robot_cfg,
  "quadruped",
  "base_link",
  "base_link",
  _QUAD_FEET,
  _QUAD_GEOMS,
  _QUAD_GEOMS,
  scaling=quadruped_velocity_scaling(),
  task_group=TASK_GROUP_UNITREE,
)


def register_velocity_tasks() -> None:
  from src.tasks.velocity import register_velocity_profile

  register_velocity_profile(GO2_PROFILE)
