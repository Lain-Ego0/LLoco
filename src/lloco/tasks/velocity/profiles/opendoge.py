"""OpenDoge (OpenDog V1.1) velocity robot profile."""

from lloco.assets.robots import get_opendoge_robot_cfg

from ..core import (
  TASK_GROUP_LAINLAB,
  VelocityRobotProfile,
  VelocityScaling,
)
from .common import _QUAD_FEET, _QUAD_GEOMS

OPENDOGE_VELOCITY_SCALING = VelocityScaling(
  action_scale=0.25,
  command_z_offset=0.2,
  max_iterations=9_000,
  viewer_distance=0.8,
  foot_radius=0.015,
  foot_sample_count=4,
  clearance_target_height=0.05,
  reset_height_range=(0.004, 0.018),
  command_ranges=((-0.8, 0.8), (-0.5, 0.5), (-0.8, 0.8)),
  play_command_ranges=((-0.8, 0.8), (-0.5, 0.5), (-0.8, 0.8)),
  command_velocity_stages=(
    {
      "step": 0,
      "lin_vel_x": (-0.8, 0.8),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-0.8, 0.8),
    },
    {
      "step": 3_000 * 24,
      "lin_vel_x": (-1.0, 1.0),
      "lin_vel_y": (-0.6, 0.6),
      "ang_vel_z": (-1.0, 1.0),
    },
    {
      "step": 6_000 * 24,
      "lin_vel_x": (-1.2, 1.2),
      "lin_vel_y": (-0.7, 0.7),
      "ang_vel_z": (-1.2, 1.2),
    },
  ),
  base_com_scale=0.5,
  decimation=2,
)

OPENDOGE_PROFILES = (
  VelocityRobotProfile(
    "OpenDoge",
    get_opendoge_robot_cfg,
    "quadruped",
    "base_link",
    "base_link",
    _QUAD_FEET,
    _QUAD_GEOMS,
    _QUAD_GEOMS,
    scaling=OPENDOGE_VELOCITY_SCALING,
    task_group=TASK_GROUP_LAINLAB,
    terrains=("Flat",),
  ),
)
