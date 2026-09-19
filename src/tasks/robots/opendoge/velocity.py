"""OpenDoge (OpenDog V1.1) velocity robot profiles."""

from dataclasses import replace

from mjlab.sensor import GridPatternCfg

from src.assets.robots import get_opendoge_robot_cfg
from src.tasks.robots.common import _QUAD_FEET, _QUAD_GEOMS
from src.tasks.velocity import (
  TASK_GROUP_LAINLAB,
  RoughTerrainOverrides,
  RoughVariantCfg,
  SensorOverrideCfg,
  SimOverrides,
  SubTerrainOverrideCfg,
  VelocityRobotProfile,
  VelocityScaling,
)

OPENDOGE_FLAT_SCALING = VelocityScaling(
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

OPENDOGE_ROUGH_SCALING = replace(
  OPENDOGE_FLAT_SCALING,
  max_iterations=10_000,
  clearance_target_height=0.06,
  reset_height_range=(0.004, 0.02),
  command_ranges=((-0.5, 0.6), (-0.4, 0.4), (-0.6, 0.6)),
  play_command_ranges=((-0.6, 0.8), (-0.4, 0.4), (-0.7, 0.7)),
  command_velocity_stages=(
    {
      "step": 0,
      "lin_vel_x": (-0.4, 0.4),
      "lin_vel_y": (-0.3, 0.3),
      "ang_vel_z": (-0.4, 0.4),
    },
    {
      "step": 2_000 * 24,
      "lin_vel_x": (-0.6, 0.6),
      "lin_vel_y": (-0.4, 0.4),
      "ang_vel_z": (-0.6, 0.6),
    },
    {
      "step": 4_000 * 24,
      "lin_vel_x": (-0.8, 0.8),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-0.8, 0.8),
    },
  ),
)


OPENDOGE_ROUGH_VARIANT = RoughVariantCfg(
  scaling=OPENDOGE_ROUGH_SCALING,
  terrain=RoughTerrainOverrides(
    max_init_terrain_level=2,
    generator_size=(6.0, 6.0),
    generator_border_width=10.0,
    generator_difficulty_range=(0.0, 1.0),
    # common_step_counter advances by 24 per PPO iteration.
    max_level_stages=((0, 3), (72_000, 5), (144_000, 7), (216_000, 9)),
    promotion_distance_scale=1.0,
    sub_terrains={
      "pyramid_stairs": SubTerrainOverrideCfg(
        step_height_range=(0.0, 0.10),
        step_width=0.22,
      ),
      "pyramid_stairs_inv": SubTerrainOverrideCfg(
        step_height_range=(0.0, 0.10),
        step_width=0.22,
      ),
      "random_rough": SubTerrainOverrideCfg(
        noise_range=(0.02, 0.10),
        scale_with_difficulty=True,
      ),
      "hf_pyramid_slope": SubTerrainOverrideCfg(
        slope_range=(0.0, 1.00),
      ),
      "hf_pyramid_slope_inv": SubTerrainOverrideCfg(
        slope_range=(0.0, 1.00),
      ),
      "wave_terrain": SubTerrainOverrideCfg(
        amplitude_range=(0.0, 0.20),
      ),
    },
  ),
  sensors=(
    SensorOverrideCfg(
      name="terrain_scan",
      pattern=GridPatternCfg(size=(0.7, 0.5), resolution=0.05),
    ),
  ),
  sim=SimOverrides(
    nconmax=64,
    njmax=600,
    contact_sensor_maxmatch=128,
    mujoco_ccd_iterations=200,
  ),
)


OPENDOGE_VELOCITY_PROFILES = (
  VelocityRobotProfile(
    "OpenDoge",
    get_opendoge_robot_cfg,
    "quadruped",
    "base_link",
    "base_link",
    _QUAD_FEET,
    _QUAD_GEOMS,
    _QUAD_GEOMS,
    scaling=OPENDOGE_FLAT_SCALING,
    rough=OPENDOGE_ROUGH_VARIANT,
    task_group=TASK_GROUP_LAINLAB,
    terrains=("Flat", "Rough"),
  ),
)


def register_velocity_tasks() -> None:
  from src.tasks.velocity import register_velocity_profile

  for profile in OPENDOGE_VELOCITY_PROFILES:
    register_velocity_profile(profile)
