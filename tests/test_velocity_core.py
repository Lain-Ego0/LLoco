"""Contract tests for the velocity environment builders."""

from dataclasses import fields, replace

import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.robots.opendoge.velocity import OPENDOGE_VELOCITY_PROFILES
from src.tasks.velocity import (
  RoughTerrainOverrides,
  RoughVariantCfg,
  SensorOverrideCfg,
  SimOverrides,
  make_rough_env_cfg,
)

_FORBIDDEN_OVERRIDE_FIELDS = {
  "actions",
  "commands",
  "events",
  "observations",
  "rewards",
  "terminations",
}


def test_flat_and_rough_share_base_config() -> None:
  flat = load_env_cfg("LainLab-OpenDoge-Flat")
  rough = load_env_cfg("LainLab-OpenDoge-Rough")

  assert flat.actions == rough.actions
  assert flat.decimation == rough.decimation
  assert flat.episode_length_s == rough.episode_length_s
  assert flat.events.keys() == rough.events.keys()
  assert flat.rewards.keys() == rough.rewards.keys()
  assert flat.viewer == rough.viewer
  assert flat.sim.mujoco.timestep == rough.sim.mujoco.timestep
  assert set(flat.observations["actor"].terms) - {"height_scan"} == set(
    rough.observations["actor"].terms
  ) - {"height_scan"}


def test_rough_variant_drives_runner_and_play_config() -> None:
  flat_rl = load_rl_cfg("LainLab-OpenDoge-Flat")
  rough_rl = load_rl_cfg("LainLab-OpenDoge-Rough")
  assert flat_rl.max_iterations == 9_000
  assert rough_rl.max_iterations == 15_000

  play_cfg = load_env_cfg("LainLab-OpenDoge-Rough", play=True)
  command = play_cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == (-0.6, 0.8)
  assert command.ranges.lin_vel_y == (-0.4, 0.4)
  assert command.ranges.ang_vel_z == (-0.7, 0.7)


def test_rough_override_types_exclude_high_level_config_sections() -> None:
  override_types = (
    RoughVariantCfg,
    RoughTerrainOverrides,
    SensorOverrideCfg,
    SimOverrides,
  )
  for override_type in override_types:
    assert {field.name for field in fields(override_type)}.isdisjoint(
      _FORBIDDEN_OVERRIDE_FIELDS
    )


def test_rough_overrides_do_not_touch_env_behavior_config() -> None:
  profile = OPENDOGE_VELOCITY_PROFILES[0]
  rough = profile.rough
  assert rough is not None
  control_profile = replace(
    profile,
    rough=replace(rough, terrain=None, sensors=(), sim=None),
  )

  configured = make_rough_env_cfg(profile)
  control = make_rough_env_cfg(control_profile)

  assert configured.actions == control.actions
  assert configured.observations == control.observations
  assert configured.rewards == control.rewards
  assert configured.commands == control.commands
  assert configured.events == control.events
  assert configured.terminations == control.terminations

  assert configured.scene.terrain is not None
  assert control.scene.terrain is not None
  assert configured.scene.terrain.terrain_generator is not None
  assert control.scene.terrain.terrain_generator is not None
  assert configured.scene.terrain.terrain_generator.size == (6.0, 6.0)
  assert control.scene.terrain.terrain_generator.size != (6.0, 6.0)
