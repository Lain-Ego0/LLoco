"""OpenDoge (OpenDog V1.1) velocity robot profiles."""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import GridPatternCfg, RayCastSensorCfg

from src.assets.robots import get_opendoge_robot_cfg
from src.tasks.robots.common import _QUAD_FEET, _QUAD_GEOMS
from src.tasks.velocity import (
  TASK_GROUP_LAINLAB,
  RoughVariantCfg,
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
  max_iterations=15_000,
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


def configure_opendoge_rough_env(cfg: ManagerBasedRlEnvCfg) -> None:
  """Scale mjlab's rough terrain curriculum to the OpenDoge body."""
  terrain = cfg.scene.terrain
  assert terrain is not None
  generator = terrain.terrain_generator
  assert generator is not None

  # OpenDoge is smaller than Go2, so use a compact terrain grid, reduced
  # roughness, and a lower initial curriculum level for reliable early
  # adaptation.
  generator.size = (6.0, 6.0)
  generator.border_width = 10.0
  terrain.max_init_terrain_level = 2

  sub = generator.sub_terrains
  sub["pyramid_stairs"] = replace(
    sub["pyramid_stairs"],
    step_height_range=(0.0, 0.04),
    step_width=0.22,
  )
  sub["pyramid_stairs_inv"] = replace(
    sub["pyramid_stairs_inv"],
    step_height_range=(0.0, 0.04),
    step_width=0.22,
  )
  sub["random_rough"] = replace(
    sub["random_rough"],
    noise_range=(0.005, 0.025),
  )
  sub["hf_pyramid_slope"] = replace(
    sub["hf_pyramid_slope"],
    slope_range=(0.0, 0.45),
  )
  sub["hf_pyramid_slope_inv"] = replace(
    sub["hf_pyramid_slope_inv"],
    slope_range=(0.0, 0.45),
  )
  sub["wave_terrain"] = replace(
    sub["wave_terrain"],
    amplitude_range=(0.0, 0.08),
  )

  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, RayCastSensorCfg) and sensor.name == "terrain_scan":
      sensor.pattern = GridPatternCfg(size=(0.7, 0.5), resolution=0.05)

  # 4096-env rough training is memory bound on 16 GB. Keep contact buffers near
  # the generic rough-velocity baseline instead of Go2-scaled large buffers.
  cfg.sim.nconmax = 64
  cfg.sim.njmax = 600
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.ccd_iterations = 200


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
    rough=RoughVariantCfg(scaling=OPENDOGE_ROUGH_SCALING),
    rough_env_hook=configure_opendoge_rough_env,
    task_group=TASK_GROUP_LAINLAB,
    terrains=("Flat", "Rough"),
  ),
)


def register_velocity_tasks() -> None:
  from src.tasks.velocity import register_velocity_profile

  for profile in OPENDOGE_VELOCITY_PROFILES:
    register_velocity_profile(profile)
