from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.raycast_sensor import RayCastSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _go2_reward_contact_mask(
  sensor: ContactSensor,
  threshold: float,
  vertical_only: bool = False,
) -> torch.Tensor:
  """Return source-style foot contacts, preferring measured force fields."""
  force = sensor.data.force
  if force is not None:
    if force.ndim != 3 or force.shape[-1] != 3:
      raise ValueError(f"Contact force must be [B, feet, 3], got {tuple(force.shape)}")
    magnitude = force[..., 2].abs() if vertical_only else torch.linalg.vector_norm(force, dim=-1)
    return magnitude > threshold
  found = sensor.data.found
  if found is None:
    raise ValueError("ContactSensor must expose either force or found fields")
  return found > 0


def _go2_trigger_state(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
  """Read optional delayed-jump state from a Go2 command term.

  The fallback keeps the reward functions reusable with the ordinary velocity
  command (for example in unit tests or a generic task configuration).
  """
  try:
    term = env.command_manager.get_term(command_name)
  except (KeyError, AttributeError):
    return None, None, None
  return (
    getattr(term, "triggered", None),
    getattr(term, "was_in_flight", None),
    getattr(term, "has_jumped", None),
  )


def go2_trot_phase_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
  cycle_time: float = 0.5,
  contact_threshold: float = 0.0,
) -> torch.Tensor:
  """Reward the diagonal stance pattern used by the source Go2 trot task."""
  sensor: ContactSensor = env.scene[sensor_name]
  contact = _go2_reward_contact_mask(sensor, threshold=max(contact_threshold, 5.0), vertical_only=True)
  if contact.shape[1] != 4:
    raise ValueError(f"Expected four foot contacts, got shape {tuple(contact.shape)}")

  phase = (
    env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
  ) % cycle_time / cycle_time
  stance_a = phase < 0.5
  stance_b = phase > 0.5
  # Contact sensor order is (FR, FL, RR, RL).  The source mask is expressed as
  # (FL, FR, RL, RR), so its first stance diagonal maps to (FL, RR) here.
  expected = torch.stack((stance_b, stance_a, stance_a, stance_b), dim=1)
  diagonal = (contact[:, 0] == contact[:, 3]) & (contact[:, 1] == contact[:, 2])
  phase_match = torch.all(contact == expected, dim=1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  moving = torch.linalg.vector_norm(command[:, :3], dim=1) > 0.1
  # The source stores ``self.trot`` as a batch-wide moving-gait fraction for
  # commanded motion, while idle environments use an all-four-feet contact
  # bit.  Tracking terms consume this state on the following reward pass.
  idle_contact = torch.all(
    _go2_reward_contact_mask(sensor, threshold=0.1), dim=1
  )
  moving_gate = diagonal.to(torch.float32).mean() > 0.7
  env._go2_trot_ready = torch.where(moving, moving_gate, idle_contact)
  return (diagonal & phase_match & moving).to(torch.float32)


def go2_default_hip_position_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize abduction/hip displacement as in the source trot task."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(joint_pos[:, 0::3] - default[:, 0::3]), dim=1)


def go2_stand_still_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint displacement when no velocity command is active."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.sum(
    torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
  )
  return error * (torch.linalg.vector_norm(command[:, :3], dim=1) < 0.1)


def go2_contact_without_command(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
) -> torch.Tensor:
  """Penalize all four feet being in contact while standing without a command."""
  sensor: ContactSensor = env.scene[sensor_name]
  contact = _go2_reward_contact_mask(sensor, threshold=0.1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  all_contact = torch.all(contact, dim=1)
  idle = torch.linalg.vector_norm(command[:, :3], dim=1) < 0.1
  return (all_contact & idle).to(torch.float32)


def go2_source_tracking_linear_velocity(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  sigma: float = 0.25,
  trot_gate: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Source tracking kernel over commanded planar velocity only."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.sum(torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=-1)
  reward = torch.exp(-error / sigma)
  if trot_gate:
    ready = getattr(env, "_go2_trot_ready", None)
    if ready is None:
      ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reward = reward * ready.to(reward.dtype)
  return reward


def go2_source_tracking_angular_velocity(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  sigma: float = 0.25,
  trot_gate: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Source tracking kernel over yaw rate only."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  reward = torch.exp(-error / sigma)
  if trot_gate:
    ready = getattr(env, "_go2_trot_ready", None)
    if ready is None:
      ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reward = reward * ready.to(reward.dtype)
  return reward


def go2_jump_tracking_linear_velocity(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  sigma: float = 0.25,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Source Jump tracking kernel, including its idle-velocity branch."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  moving = torch.linalg.vector_norm(command[:, :3], dim=1) > command_threshold
  moving_error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=-1
  )
  moving_reward = torch.exp(-moving_error / sigma)
  idle_reward = torch.exp(
    -torch.linalg.vector_norm(asset.data.root_link_lin_vel_b[:, :2], dim=1) / sigma
  )
  return torch.where(moving, moving_reward, idle_reward)


def go2_jump_tracking_angular_velocity(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  sigma: float = 0.25,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Source Jump yaw tracking kernel, including its idle branch."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  moving = torch.linalg.vector_norm(command[:, :3], dim=1) > command_threshold
  moving_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  moving_reward = torch.exp(-moving_error / sigma)
  idle_reward = torch.exp(-torch.abs(asset.data.root_link_ang_vel_b[:, 2]) / sigma)
  return torch.where(moving, moving_reward, idle_reward)


def go2_jump_vertical_velocity_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Positive vertical-velocity kernel used by the source jump task."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.exp(-torch.abs(asset.data.root_link_lin_vel_b[:, 2]))


def go2_jump_angular_velocity_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Positive roll/pitch angular-velocity kernel from Go2 jump."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.exp(-torch.abs(asset.data.root_link_ang_vel_b[:, :2]).sum(dim=-1))


def go2_jump_orientation_reward(
  env: ManagerBasedRlEnv,
  sharpness: float = 10.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Positive uprightness kernel used while jumping."""
  asset: Entity = env.scene[asset_cfg.name]
  tilt = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=-1)
  return torch.exp(-sharpness * tilt)


def go2_default_position_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute displacement from the nominal Go2 joint pose."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  ).sum(dim=-1)


def go2_stand_zero_linear_velocity_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  target_height: float = 0.47,
  ready_threshold: float = 0.78,
  forward_sign: float = 1.0,
  sigma: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Legged-stand tracking kernel active only for near-zero commands."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error_x = command[:, 0] - forward_sign * asset.data.root_link_lin_vel_b[:, 2]
  error_y = command[:, 1] - asset.data.root_link_lin_vel_b[:, 1]
  reward = torch.exp(-(error_x.square() + error_y.square()) / sigma)
  idle = torch.linalg.vector_norm(command[:, :2], dim=-1) < 0.1
  return reward * idle * _go2_stand_ready_mask(env, target_height, ready_threshold, asset_cfg)


def go2_stand_zero_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  target_height: float = 0.47,
  ready_threshold: float = 0.78,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Legged-stand yaw/roll error penalty for near-zero commands."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.square(command[:, 2] + asset.data.root_link_ang_vel_b[:, 0])
  idle = torch.abs(command[:, 2]) < 0.1
  return error * idle * _go2_stand_ready_mask(env, target_height, ready_threshold, asset_cfg)


def go2_special_collision_penalty(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
) -> torch.Tensor:
  """Aggregate non-foot contacts into the source count-based term.

  Isaac Gym sums every penalized body contact whose force exceeds ``0.1``;
  preserve that count rather than reducing each sensor to a single boolean.
  """
  penalty = torch.zeros(env.num_envs, device=env.device)
  for sensor_name in sensor_names:
    sensor: ContactSensor = env.scene[sensor_name]
    force = sensor.data.force
    if force is not None:
      if force.shape[-1] != 3:
        raise ValueError(f"{sensor_name} force must end in XYZ, got {tuple(force.shape)}")
      penalty = penalty + (
        torch.linalg.vector_norm(force, dim=-1) > 0.1
      ).to(torch.float32).reshape(env.num_envs, -1).sum(dim=-1)
      continue
    found = sensor.data.found
    assert found is not None
    penalty = penalty + found.to(torch.float32).reshape(env.num_envs, -1).sum(dim=-1)
  return penalty


def go2_trot_feet_clearance_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  cycle_time: float = 0.5,
  target_height: float = 0.06,
  sharpness: float = 10.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Source trot swing-foot clearance shaping term.

  The legacy implementation selects the two diagonal pairs, subtracts the
  0.02 m foot-radius offset, and gates each pair with the complementary gait
  phase.  The target height follows ``abs(sin(2π phase))`` rather than staying
  fixed at its peak value.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  moving = torch.linalg.vector_norm(command[:, :3], dim=-1) > 0.1
  phase = (env.episode_length_buf.to(torch.float32) * env.step_dt) % cycle_time / cycle_time
  # Site order is explicitly passed by the config as (FR, FL, RR, RL).
  heights = (
    asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    - env.scene.env_origins[:, 2:3]
    - 0.02
  )
  target = torch.abs(torch.sin(2.0 * torch.pi * phase)) * target_height
  # Source feet_indices are (FL, FR, RL, RR), hence the corresponding site
  # pairs here are (FL, RR)=(1,2) and (FR, RL)=(0,3).
  source_left = heights[:, (1, 2)]
  source_right = heights[:, (0, 3)]
  swing_left = (~(phase < 0.5)).to(heights.dtype).unsqueeze(-1)
  swing_right = (~(phase > 0.5)).to(heights.dtype).unsqueeze(-1)
  pair_left = torch.exp(-sharpness * torch.sum(torch.abs(source_left - target.unsqueeze(-1)) * swing_left, dim=-1))
  pair_right = torch.exp(-sharpness * torch.sum(torch.abs(source_right - target.unsqueeze(-1)) * swing_right, dim=-1))
  return (pair_left + pair_right) * moving


def go2_linear_velocity_z_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared vertical base velocity from the source locomotion reward."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def go2_angular_velocity_xy_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared roll/pitch angular velocity from the source reward."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=-1)


def go2_orientation_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared projected-gravity tilt penalty used by source rough tasks."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)


def go2_base_height_penalty(
  env: ManagerBasedRlEnv,
  target_height: float = 0.4,
  sensor_name: str = "terrain_scan",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track base height above the local raycast terrain, like Isaac Gym."""
  asset: Entity = env.scene[asset_cfg.name]
  root_z = asset.data.root_link_pos_w[:, 2]
  try:
    sensor = env.scene[sensor_name]
  except KeyError:
    sensor = None
  if isinstance(sensor, RayCastSensor):
    hit_z = sensor.data.hit_pos_w[..., 2]
    distances = sensor.data.distances
    valid_hit_z = torch.where(distances < 0.0, root_z.unsqueeze(-1), hit_z)
    local_height = root_z - valid_hit_z.mean(dim=-1)
  else:
    local_height = root_z - env.scene.env_origins[:, 2]
  return torch.square(local_height - target_height)


def go2_joint_acceleration_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared joint acceleration term used in all source rough tasks."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=-1)


def go2_action_smoothness_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Second-order raw-action smoothness penalty from the source runner."""
  action = env.action_manager.action
  previous = env.action_manager.prev_action
  previous_previous = env.action_manager.prev_prev_action
  acceleration = action - 2.0 * previous + previous_previous
  return torch.sum(torch.square(acceleration), dim=-1)


def go2_base_height_error(
  env: ManagerBasedRlEnv,
  target_height: float = 0.29,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared base-height error used by jump and standing variants."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return torch.square(height - target_height)


def go2_vertical_velocity_reward(
  env: ManagerBasedRlEnv,
  target_velocity: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Smooth reward for tracking vertical base velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.exp(-torch.square(asset.data.root_link_lin_vel_b[:, 2] - target_velocity) / 0.25)


def go2_idle_base_height_reward(
  env: ManagerBasedRlEnv,
  target_height: float = 0.3,
  sharpness: float = 10.0,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Source jump base-height reward, active while no motion is commanded."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  idle = torch.linalg.vector_norm(command[:, :3], dim=1) < command_threshold
  return torch.exp(-torch.abs(height - target_height) * sharpness) * idle


def go2_all_feet_airborne(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  contact_threshold: float = 0.0,
  command_name: str | None = None,
) -> torch.Tensor:
  """Return the source flight flag, or current all-feet contact state."""
  if command_name is not None:
    _triggered, was_in_flight, _has_jumped = _go2_trigger_state(env, command_name)
    if was_in_flight is not None:
      return was_in_flight.to(torch.float32)
  sensor: ContactSensor = env.scene[sensor_name]
  contact = _go2_reward_contact_mask(sensor, threshold=contact_threshold)
  return (~contact).all(dim=1).to(torch.float32)


def go2_any_foot_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  contact_threshold: float = 0.0,
) -> torch.Tensor:
  """Return one after at least one foot has contacted the ground."""
  sensor: ContactSensor = env.scene[sensor_name]
  contact = _go2_reward_contact_mask(sensor, threshold=contact_threshold)
  return contact.any(dim=1).to(torch.float32)


def go2_stumble_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  horizontal_ratio: float = 5.0,
) -> torch.Tensor:
  """Detect a foot striking a near-vertical surface.

  This is the source Go2 ``_reward_stumble`` kernel: a foot is considered to
  stumble when its horizontal contact force is more than five times its
  vertical force.  ContactSensor keeps the same per-foot XYZ force layout, so
  the term can be shared by CTS, DreamWaQ and TS variants without duplicating
  task directories.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  if force is None:
    return torch.zeros(env.num_envs, device=env.device)
  if force.ndim != 3 or force.shape[-1] != 3:
    raise ValueError(f"{sensor_name} force must have shape [B, feet, 3], got {tuple(force.shape)}")
  horizontal = torch.linalg.vector_norm(force[..., :2], dim=-1)
  vertical = force[..., 2].abs()
  return (horizontal > horizontal_ratio * vertical).any(dim=-1).to(force.dtype)


def go2_base_height_phase_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  phase: str,
  sharpness: float = 5.0,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
  gain: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track a base height during the source airborne/landing phases."""
  if phase not in ("airborne", "contact"):
    raise ValueError(f"phase must be 'airborne' or 'contact', got {phase!r}")
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  _triggered, was_in_flight, has_jumped = _go2_trigger_state(env, command_name)
  if was_in_flight is not None and has_jumped is not None:
    # Source rewards are phase/state gated: flight shaping remains active after
    # the first airborne frame, while stance shaping starts only after landing
    # and excludes a robot that has fallen below 0.2 m.
    if phase == "airborne":
      mask = was_in_flight & ~has_jumped
    else:
      mask = has_jumped & (height > 0.2)
  else:
    airborne = go2_all_feet_airborne(env, sensor_name)
    mask = airborne if phase == "airborne" else 1.0 - airborne
  return gain * torch.exp(-torch.abs(height - target_height) * sharpness) * mask


def go2_upward_velocity_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward positive vertical base velocity before the first landing."""
  asset: Entity = env.scene[asset_cfg.name]
  triggered, _was_in_flight, has_jumped = _go2_trigger_state(env, command_name)
  reward = asset.data.root_link_lin_vel_b[:, 2].clamp_min(0.0)
  if triggered is not None and has_jumped is not None:
    reward = reward * (triggered & ~has_jumped)
  return reward


def go2_pitch_angular_velocity_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward source backflip pitch velocity before and during flight."""
  asset: Entity = env.scene[asset_cfg.name]
  velocity = asset.data.root_link_ang_vel_b[:, 1].clamp_min(0.0)
  triggered, was_in_flight, has_jumped = _go2_trigger_state(env, command_name)
  if triggered is not None and was_in_flight is not None and has_jumped is not None:
    velocity = velocity * (
      triggered.to(velocity.dtype) * (~was_in_flight & ~has_jumped).to(velocity.dtype)
      + 3.0 * (was_in_flight & ~has_jumped).to(velocity.dtype)
    )
  return velocity.clamp_max(20.0)


def go2_backflip_orientation_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Smooth absolute-orientation reward after a successful backflip."""
  asset: Entity = env.scene[asset_cfg.name]
  roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
  angles = torch.stack((roll, pitch, yaw), dim=-1)
  reward = torch.exp(-torch.abs(angles).sum(dim=-1))
  _triggered, _was_in_flight, has_jumped = _go2_trigger_state(env, command_name)
  if has_jumped is not None:
    term = env.command_manager.get_term(command_name)
    max_pitch = getattr(term, "max_pitch_ang_vel", None)
    if max_pitch is not None:
      reward = reward * (has_jumped & (max_pitch > 7.0))
    else:
      reward = reward * has_jumped
  return reward


def go2_landing_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  target_offset_scale: float = 1.0,
  min_jump_height: float | None = None,
  max_orientation_error: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward returning near the source task's expected landing position.

  Spring-Jump additionally requires a sufficiently high jump and a settled
  orientation; Backflip keeps the looser ``has_jumped``-only condition by
  leaving these optional gates unset.
  """
  try:
    term = env.command_manager.get_term(command_name)
  except (KeyError, AttributeError):
    return torch.zeros(env.num_envs, device=env.device)
  landing_pos = getattr(term, "landing_pos", None)
  start_pos = getattr(term, "start_pos", None)
  has_jumped = getattr(term, "has_jumped", None)
  if landing_pos is None or start_pos is None or has_jumped is None:
    return torch.zeros(env.num_envs, device=env.device)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  target = start_pos + target_offset_scale * command[:, :2]
  error = torch.abs(target - landing_pos).sum(dim=-1)
  success = has_jumped
  if min_jump_height is not None:
    max_height = getattr(term, "max_height", None)
    if max_height is not None:
      success = success & (max_height > min_jump_height)
  if max_orientation_error is not None:
    asset: Entity = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
    orientation_error = torch.abs(torch.stack((roll, pitch, yaw), dim=-1)).sum(dim=-1)
    success = success & (orientation_error < max_orientation_error)
  return torch.exp(-error) * success


def go2_before_setting_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  denominator: float = 2.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward holding the default pose before the jump trigger."""
  asset: Entity = env.scene[asset_cfg.name]
  error = torch.abs(asset.data.joint_pos - asset.data.default_joint_pos).sum(dim=-1)
  triggered, _was_in_flight, _has_jumped = _go2_trigger_state(env, command_name)
  active = torch.ones_like(error, dtype=torch.bool) if triggered is None else ~triggered
  return torch.exp(-error / denominator) * active


def go2_joint_position_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute joint displacement penalty used by flip tasks after landing."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  ).sum(dim=-1)


def go2_hip_position_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute displacement of the four abduction/hip joints."""
  asset: Entity = env.scene[asset_cfg.name]
  joints = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.abs(joints[:, 0::3] - default[:, 0::3]).sum(dim=-1)


def go2_hip_position_squared_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared abduction-joint error used by the source CTS reward."""
  asset: Entity = env.scene[asset_cfg.name]
  joints = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.square(joints - default).sum(dim=-1)


def go2_rear_hip_limit_penalty(
  env: ManagerBasedRlEnv,
  limit: float = 0.4,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Count a source AMP-DreamWaQ penalty when either rear hip exceeds ``limit``."""
  asset: Entity = env.scene[asset_cfg.name]
  joints = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return ((joints.abs() > limit).any(dim=-1)).to(torch.float32)


def go2_line_velocity_stance_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize planar drift once the source flip has landed."""
  asset: Entity = env.scene[asset_cfg.name]
  penalty = torch.abs(asset.data.root_link_lin_vel_b[:, :2]).sum(dim=-1)
  _triggered, _was_in_flight, has_jumped = _go2_trigger_state(env, command_name)
  if has_jumped is not None:
    penalty = penalty * has_jumped
  return penalty


def go2_flight_linear_velocity_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  gain: float = 1.6,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Source Spring-Jump forward-speed reward during the flight phase."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  reward = torch.exp(
    -torch.square(command[:, 0] * gain - asset.data.root_link_lin_vel_b[:, 0])
  )
  _triggered, was_in_flight, has_jumped = _go2_trigger_state(env, command_name)
  if was_in_flight is not None and has_jumped is not None:
    reward = reward * (was_in_flight & ~has_jumped)
  return reward


def go2_flip_foot_clearance_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  gain: float = 6.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Source flip-task clearance error during the airborne phase."""
  asset: Entity = env.scene[asset_cfg.name]
  positions = asset.data.site_pos_w[:, asset_cfg.site_ids]
  root_pos = asset.data.root_link_pos_w.unsqueeze(1)
  # ``quat_apply_inverse`` operates on matching flat leading dimensions;
  # expand the root quaternion over each selected foot before flattening.
  num_feet = positions.shape[1]
  quaternion = asset.data.root_link_quat_w[:, None, :].expand(-1, num_feet, -1)
  relative = quat_apply_inverse(
    quaternion.reshape(-1, 4), (positions - root_pos).reshape(-1, 3)
  ).reshape(positions.shape[0], num_feet, 3)
  error = torch.abs(relative[..., 2] + 0.20).sum(dim=-1)
  _triggered, was_in_flight, has_jumped = _go2_trigger_state(env, command_name)
  if was_in_flight is not None and has_jumped is not None:
    error = error * (was_in_flight & ~has_jumped)
  return gain * error


def go2_ang_vel_xy_penalty(
  env: ManagerBasedRlEnv,
  include_pitch: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalty for residual body angular velocity after a special action."""
  asset: Entity = env.scene[asset_cfg.name]
  if include_pitch:
    return torch.abs(asset.data.root_link_ang_vel_b).sum(dim=-1)
  return torch.abs(asset.data.root_link_ang_vel_b[:, (0, 2)]).sum(dim=-1)


def go2_joint_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared joint velocity penalty used by source flip configurations."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]).sum(dim=-1)


def go2_joint_velocity_limit_penalty(
  env: ManagerBasedRlEnv,
  limit_factor: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Soft joint-velocity limit penalty with a one-radian cap per joint."""
  asset: Entity = env.scene[asset_cfg.name]
  velocity = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
  # MuJoCo does not expose Isaac Gym's velocity limits through EntityData; Go2
  # uses the conservative 30 rad/s limit from its URDF configuration.
  excess = (velocity - 30.0 * limit_factor).clamp_min(0.0).clamp_max(1.0)
  return excess.sum(dim=-1)


def go2_torque_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute actuator-force penalty for special-action controllers."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(asset.data.actuator_force[:, asset_cfg.joint_ids]).sum(dim=-1)


def go2_foot_contact_force_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  max_contact_force: float = 150.0,
) -> torch.Tensor:
  """Penalize foot contact force above the source safety threshold."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  excess = (torch.linalg.vector_norm(force, dim=-1) - max_contact_force).clamp_min(0.0)
  return excess.sum(dim=-1)


def go2_orientation_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute orientation reward used while a spring jump is settling."""
  asset: Entity = env.scene[asset_cfg.name]
  roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
  return torch.exp(-torch.abs(torch.stack((roll, pitch, yaw), dim=-1)).sum(dim=-1))


def go2_orientation_before_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward upright orientation before a backflip is triggered."""
  reward = go2_orientation_reward(env, asset_cfg)
  triggered, _was_in_flight, _has_jumped = _go2_trigger_state(env, command_name)
  if triggered is not None:
    reward = reward * ~triggered
  return reward


def _go2_stand_ready_mask(
  env: ManagerBasedRlEnv,
  target_height: float,
  threshold: float,
  asset_cfg: SceneEntityCfg,
  readiness_sharpness: float | None = None,
) -> torch.Tensor:
  """Return the source stand-task readiness gate for every environment.

  The legacy Isaac Gym code stores ``rew_hanstand`` as the mean base-height
  reward over the whole vectorized batch and gates several terms with that one
  scalar.  Handstand uses a height sharpness of ``5`` while Leggedstand uses
  ``10``.  Infer the legacy choice from the task's target height when callers
  do not provide it explicitly, while returning a batch-shaped mask for the
  mjlab reward API.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  if readiness_sharpness is None:
    readiness_sharpness = 5.0 if target_height >= 0.5 else 10.0
  readiness = torch.exp(-torch.abs(height - target_height) * readiness_sharpness)
  batch_ready = readiness.mean() > threshold
  return torch.full_like(readiness, batch_ready, dtype=torch.bool)


def go2_stand_linear_velocity_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  target_height: float = 0.52,
  ready_threshold: float = 0.70,
  forward_sign: float = -1.0,
  sigma: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track stand-task forward/lateral velocity in the rotated body frame."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  # In a handstand, world-up maps to -base-z; in a legged stand it maps to
  # +base-z. ``forward_sign`` selects that source convention.
  error_x = command[:, 0] - forward_sign * asset.data.root_link_lin_vel_b[:, 2]
  error_y = command[:, 1] - asset.data.root_link_lin_vel_b[:, 1]
  reward = torch.exp(-(error_x.square() + error_y.square()) / sigma)
  ready = _go2_stand_ready_mask(env, target_height, ready_threshold, asset_cfg)
  return reward * ready


def go2_stand_angular_velocity_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  target_height: float = 0.52,
  ready_threshold: float = 0.70,
  forward_sign: float = -1.0,
  angular_sign: float | None = None,
  sigma: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track source stand yaw command, which maps to base roll rate."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  # The linear mapping and angular mapping have opposite signs in the source:
  # handstand uses ``cmd - roll_rate`` while leggedstand uses
  # ``cmd + roll_rate``.  ``forward_sign`` describes the linear mapping, so
  # use its opposite by default and allow an explicit override for variants.
  if angular_sign is None:
    angular_sign = -forward_sign
  error = command[:, 2] - angular_sign * asset.data.root_link_ang_vel_b[:, 0]
  reward = torch.exp(-error.square() / sigma)
  ready = _go2_stand_ready_mask(env, target_height, ready_threshold, asset_cfg)
  return reward * ready


def go2_stand_linear_z_velocity_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Keep the stand's vertical body-frame velocity near zero."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.exp(-torch.abs(asset.data.root_link_lin_vel_b[:, 0]) * 10.0)


def go2_stand_ang_vel_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return the source positive reward for low roll/yaw angular velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.exp(-torch.linalg.vector_norm(torch.abs(asset.data.root_link_ang_vel_b[:, 1:]), dim=-1))


def go2_stand_roll_penalty(
  env: ManagerBasedRlEnv,
  ready_target_height: float = 0.52,
  ready_threshold: float = 0.70,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize roll error after a stand has reached its target height."""
  asset: Entity = env.scene[asset_cfg.name]
  roll, _pitch, _yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
  ready = _go2_stand_ready_mask(env, ready_target_height, ready_threshold, asset_cfg)
  return torch.abs(roll) * ready


def go2_stand_foot_clearance_reward(
  env: ManagerBasedRlEnv,
  foot_indices: tuple[int, int],
  cycle_time: float = 1.6,
  target_height: float = 0.06,
  ready_target_height: float = 0.52,
  ready_threshold: float = 0.70,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track the sinusoidal clearance of the two non-support feet."""
  asset: Entity = env.scene[asset_cfg.name]
  phase = (env.episode_length_buf.to(torch.float32) * env.step_dt) % cycle_time / cycle_time
  target = torch.abs(torch.sin(2.0 * torch.pi * phase)) * target_height
  # Isaac Gym subtracts the 0.02 m foot-radius offset before comparing the
  # rigid-state height with the sinusoidal target.
  heights = (
    asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    - env.scene.env_origins[:, 2:3]
    - 0.02
  )
  selected = heights[:, list(foot_indices)]
  # The source builds a two-column gait mask: selected leg 0 swings in the
  # second half, selected leg 1 swings in the first half.  Keeping the pair
  # phases separate is important for the alternating stand motion.
  swing = torch.stack((phase >= 0.5, phase <= 0.5), dim=-1).to(selected.dtype)
  error = torch.abs(selected - target.unsqueeze(-1))
  reward = torch.exp(-10.0 * error) * swing
  reward = reward.sum(dim=-1)
  ready = _go2_stand_ready_mask(env, ready_target_height, ready_threshold, asset_cfg)
  return reward * ready


def go2_stand_contact_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  foot_indices: tuple[int, int] = (2, 3),
  ready_target_height: float = 0.52,
  ready_threshold: float = 0.70,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward exactly one supporting foot in a stand task."""
  sensor: ContactSensor = env.scene[sensor_name]
  contacts = _go2_reward_contact_mask(sensor, threshold=1.0, vertical_only=True)[:, list(foot_indices)]
  ready = _go2_stand_ready_mask(env, ready_target_height, ready_threshold, asset_cfg)
  return (contacts.sum(dim=-1) == 1).to(torch.float32) * ready


def go2_stand_foot_air_time_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  foot_indices: tuple[int, int] = (2, 3),
  ready_target_height: float = 0.52,
  ready_threshold: float = 0.70,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the first landing after a sufficiently long swing.

  The source reward is event-based: it contributes ``last_air_time - 0.4``
  only on the first contact frame.  Returning the continuously accumulated
  air time would produce a different scale and would reward feet that never
  land, so use ContactSensor's first-contact bookkeeping here.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.current_air_time is None or data.last_air_time is None:
    return torch.zeros(env.num_envs, device=env.device)
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  first_contact &= _go2_reward_contact_mask(
    sensor, threshold=1.0, vertical_only=True
  )
  ready = _go2_stand_ready_mask(env, ready_target_height, ready_threshold, asset_cfg)
  landing_reward = torch.sum(
    (data.last_air_time[:, list(foot_indices)] - 0.4)
    * first_contact[:, list(foot_indices)].to(data.last_air_time.dtype),
    dim=-1,
  )
  return landing_reward * ready


def go2_stand_orientation_symmetry_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Left/right roll symmetry penalty used by both stand variants."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b[:, 1].square()


def go2_stand_feet_height_symmetry_penalty(
  env: ManagerBasedRlEnv,
  foot_indices: tuple[int, int] = (0, 1),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute height difference between the two swing/support feet."""
  asset: Entity = env.scene[asset_cfg.name]
  heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  return torch.abs(heights[:, foot_indices[0]] - heights[:, foot_indices[1]])


def go2_stand_hand_height_reward(
  env: ManagerBasedRlEnv,
  target_height: float = 0.67,
  sharpness: float = 10.0,
  foot_indices: tuple[int, ...] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping source-selected feet/hands near the support height."""
  asset: Entity = env.scene[asset_cfg.name]
  heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - env.scene.env_origins[:, 2:3]
  if foot_indices is not None:
    if any(index < 0 or index >= heights.shape[-1] for index in foot_indices):
      raise ValueError(f"Invalid foot_indices {foot_indices} for {heights.shape[-1]} sites")
    heights = heights[:, list(foot_indices)]
  return torch.exp(-sharpness * torch.abs(heights - target_height).sum(dim=-1))


def go2_alive_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Constant alive bonus used by the source standing tasks."""
  return torch.ones(env.num_envs, device=env.device)


def go2_joint_symmetry_penalty(
  env: ManagerBasedRlEnv,
  leg_indices: tuple[int, ...] = (0, 1, 2, 3),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Pairwise left/right joint symmetry penalty from source stand tasks.

  Handstand compares both leg pairs, while Leggedstand's source implementation
  intentionally compares only the rear/support pair.  ``leg_indices`` keeps
  that distinction explicit instead of applying an extra penalty to the other
  pair.
  """
  asset: Entity = env.scene[asset_cfg.name]
  joints = asset.data.joint_pos[:, asset_cfg.joint_ids]
  if joints.shape[-1] != 12:
    raise ValueError(f"Go2 symmetry expects 12 joints, got {joints.shape[-1]}")
  joints = joints.view(joints.shape[0], 4, 3).clone()
  # Hip sign conventions are mirrored on the right legs in the source model.
  joints[:, 1, 0] *= -1.0
  joints[:, 3, 0] *= -1.0
  if len(leg_indices) == 2:
    pairs = (leg_indices,)
  elif len(leg_indices) == 4:
    pairs = ((leg_indices[0], leg_indices[1]), (leg_indices[2], leg_indices[3]))
  else:
    raise ValueError("leg_indices must contain one or two left/right pairs")
  terms = []
  for left, right in pairs:
    if not 0 <= left < 4 or not 0 <= right < 4 or left == right:
      raise ValueError(f"Invalid left/right leg pair {(left, right)}")
    terms.append(torch.abs(joints[:, left] - joints[:, right]).sum(dim=-1))
  return torch.stack(terms, dim=-1).sum(dim=-1)


def go2_target_joint_position_penalty(
  env: ManagerBasedRlEnv,
  target_joint_pos: tuple[float, ...],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute joint-position error for a source special-action target pose."""
  asset: Entity = env.scene[asset_cfg.name]
  joints = asset.data.joint_pos[:, asset_cfg.joint_ids]
  target = torch.as_tensor(target_joint_pos, device=env.device, dtype=joints.dtype)
  if target.numel() != joints.shape[-1]:
    raise ValueError(f"Expected {joints.shape[-1]} target joints, got {target.numel()}")
  return torch.abs(joints - target).sum(dim=-1)


def go2_target_joint_position_reward(
  env: ManagerBasedRlEnv,
  target_joint_pos: tuple[float, ...],
  joint_slice: tuple[int, int] = (0, 12),
  sharpness: float = 1.0,
  ready_target_height: float | None = None,
  ready_threshold: float = 0.70,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exponential target-pose reward used after a stand configuration settles."""
  asset: Entity = env.scene[asset_cfg.name]
  joints = asset.data.joint_pos[:, asset_cfg.joint_ids]
  target = torch.as_tensor(target_joint_pos, device=env.device, dtype=joints.dtype)
  if target.numel() != joints.shape[-1]:
    raise ValueError(f"Expected {joints.shape[-1]} target joints, got {target.numel()}")
  start, end = joint_slice
  if not 0 <= start < end <= joints.shape[-1]:
    raise ValueError(f"Invalid joint_slice {joint_slice} for {joints.shape[-1]} joints")
  error = torch.abs(joints[:, start:end] - target[start:end]).sum(dim=-1)
  reward = torch.exp(-sharpness * error)
  if ready_target_height is not None:
    reward = reward * _go2_stand_ready_mask(
      env, ready_target_height, ready_threshold, asset_cfg
    )
  return reward


def go2_inverted_orientation_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward an upside-down base orientation for the backflip scaffold."""
  asset: Entity = env.scene[asset_cfg.name]
  gravity_b = asset.data.projected_gravity_b
  return torch.clamp(-gravity_b[:, 2], min=0.0)


def go2_target_gravity_error(
  env: ManagerBasedRlEnv,
  target_gravity: tuple[float, float, float] = (-1.0, 0.0, 0.0),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared projected-gravity error for handstand/leggedstand targets."""
  asset: Entity = env.scene[asset_cfg.name]
  target = torch.as_tensor(target_gravity, device=env.device, dtype=asset.data.projected_gravity_b.dtype)
  return torch.sum(torch.square(asset.data.projected_gravity_b - target), dim=-1)


def go2_target_base_height_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  sharpness: float = 10.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exponential reward for the source task's target base height."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return torch.exp(-torch.abs(height - target_height) * sharpness)


def go2_selected_feet_airborne(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  foot_indices: tuple[int, ...] = (0, 1),
) -> torch.Tensor:
  """Reward selected feet staying off the ground (source stand tasks)."""
  sensor: ContactSensor = env.scene[sensor_name]
  selected = _go2_reward_contact_mask(sensor, threshold=1.0)[:, list(foot_indices)]
  return (~selected).to(torch.float32).prod(dim=-1)


def go2_exactly_one_foot_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  foot_indices: tuple[int, ...] = (2, 3),
) -> torch.Tensor:
  """Reward exactly one contact in a selected foot pair."""
  sensor: ContactSensor = env.scene[sensor_name]
  contacts = _go2_reward_contact_mask(sensor, threshold=1.0, vertical_only=True)[:, list(foot_indices)].to(torch.int32)
  return (contacts.sum(dim=-1) == 1).to(torch.float32)


def go2_jump_contact_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
  cycle_time: float = 1.5,
) -> torch.Tensor:
  """Reward synchronized four-foot contact during the jump gait phase."""
  sensor: ContactSensor = env.scene[sensor_name]
  contact = _go2_reward_contact_mask(sensor, threshold=5.0, vertical_only=True)
  phase = (
    env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
  ) % cycle_time / cycle_time
  stance = phase < 0.6
  synchronized = (contact[:, 0] == contact[:, 1])
  synchronized &= contact[:, 1] == contact[:, 2]
  synchronized &= contact[:, 2] == contact[:, 3]
  synchronized &= contact[:, 0] == stance
  command = env.command_manager.get_command(command_name)
  assert command is not None
  active = torch.linalg.vector_norm(command[:, :3], dim=1) > 0.1
  return (synchronized & active).to(torch.float32)


def go2_jump_feet_clearance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str = "twist",
  target_height: float = 0.05,
) -> torch.Tensor:
  """Reward swing-foot clearance using the flat-terrain site positions."""
  asset: Entity = env.scene[asset_cfg.name]
  # Source uses rigid-state height minus the 0.02 m foot-radius offset, then
  # clips the remaining clearance to [0, target_height].
  foot_height = (
    asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    - env.scene.env_origins[:, 2:3]
    - 0.02
  )
  phase = (env.episode_length_buf.to(dtype=torch.float32) * env.step_dt) % 1.5 / 1.5
  swing = (phase >= 0.6).to(torch.float32).unsqueeze(1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  active = (torch.linalg.vector_norm(command[:, :3], dim=1) > 0.1).to(torch.float32)
  return torch.sum(torch.clamp(foot_height, min=0.0, max=target_height) * swing, dim=1) * active


def go2_jump_feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
  command_threshold: float = 0.1,
  target_air_time: float = 0.5,
) -> torch.Tensor:
  """Reward first landings using the source Jump air-time bookkeeping."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.current_air_time is None or data.last_air_time is None:
    return torch.zeros(env.num_envs, device=env.device)
  # ``compute_first_contact`` uses the same found-bit state that updates
  # ``last_air_time`` and avoids introducing a Python-side last-contact cache.
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  first_contact &= _go2_reward_contact_mask(
    sensor, threshold=1.0, vertical_only=True
  )
  landing_reward = torch.sum(
    (data.last_air_time - target_air_time) * first_contact.to(data.last_air_time.dtype),
    dim=-1,
  )
  command = env.command_manager.get_command(command_name)
  assert command is not None
  active = torch.linalg.vector_norm(command[:, :3], dim=-1) > command_threshold
  return landing_reward * active.to(landing_reward.dtype)


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
