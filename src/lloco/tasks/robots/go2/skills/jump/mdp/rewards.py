"""Jump-specific reward equations."""

import torch
from mjlab.entity import Entity
from mjlab.managers import RewardTermCfg
from mjlab.sensor import ContactSensor

from ...shared import rl as shared
from ...shared.contacts import joint_ids, source_vertical_contact
from .observations import jump_stance_mask

absolute_torques = shared.absolute_torques
action_rate = shared.action_rate
collision = shared.collision
contact_without_command = shared.contact_without_command
default_pos = shared.default_pos
stand_still = shared.stand_still


def jump_tracking_lin_vel(env, command_name: str, sigma: float) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  cmd = shared.command(env, command_name)
  is_moving = torch.linalg.vector_norm(cmd[:, :3], dim=1) > 0.1
  error = torch.square(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2]).sum(dim=1)
  return (
    torch.exp(-error / sigma) * is_moving
    + torch.exp(
      -torch.linalg.vector_norm(robot.data.root_link_lin_vel_b[:, :2], dim=1) / sigma
    )
    * ~is_moving
  )


def jump_tracking_ang_vel(env, command_name: str, sigma: float) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  cmd = shared.command(env, command_name)
  is_moving = torch.linalg.vector_norm(cmd[:, :3], dim=1) > 0.1
  error = torch.square(cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
  return (
    torch.exp(-error / sigma) * is_moving
    + torch.exp(-torch.abs(robot.data.root_link_ang_vel_b[:, 2]) / sigma) * ~is_moving
  )


def jump_lin_vel_z(env) -> torch.Tensor:
  return torch.exp(-torch.abs(env.scene["robot"].data.root_link_lin_vel_b[:, 2]))


def jump_ang_vel_xy(env) -> torch.Tensor:
  return torch.exp(
    -torch.linalg.vector_norm(
      torch.abs(env.scene["robot"].data.root_link_ang_vel_b[:, :2]), dim=1
    )
  )


def jump_orientation(env) -> torch.Tensor:
  return torch.exp(
    -10.0
    * torch.linalg.vector_norm(
      env.scene["robot"].data.projected_gravity_b[:, :2], dim=1
    )
  )


def jump_base_height(env, command_name: str, target_height: float) -> torch.Tensor:
  return torch.exp(
    -10.0 * torch.abs(env.scene["robot"].data.root_link_pos_w[:, 2] - target_height)
  ) * ~shared.moving(env, command_name)


class JointVelocityDifference:
  """Jump source acceleration term omits division by policy dt."""

  def __init__(self, cfg: RewardTermCfg, env) -> None:
    del cfg
    self._last_velocity = torch.zeros(
      env.num_envs, len(joint_ids(env.scene["robot"])), device=env.device
    )

  def __call__(self, env) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    velocity = robot.data.joint_vel[:, joint_ids(robot)]
    value = torch.square(self._last_velocity - velocity).sum(dim=1)
    self._last_velocity.copy_(velocity)
    return value

  def reset(self, env_ids=None) -> None:
    self._last_velocity[env_ids] = 0.0


def jump_contact_match(
  env, sensor_name: str, command_name: str, cycle_time: float
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  contact = source_vertical_contact(sensor, 5.0)
  all_equal = (
    (contact[:, 0] == contact[:, 1])
    & (contact[:, 1] == contact[:, 2])
    & (contact[:, 2] == contact[:, 3])
  )
  return (
    all_equal
    & (contact[:, 3] == jump_stance_mask(env, cycle_time)[:, 0].bool())
    & shared.moving(env, command_name)
  )


def jump_feet_clearance(
  env, command_name: str, cycle_time: float, max_height: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids, names = robot.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)
  if tuple(names) != ("FL", "FR", "RL", "RR"):
    raise RuntimeError(f"Go2 foot site order mismatch: {names}")
  height = (robot.data.site_pos_w[:, ids, 2] - 0.02).clamp(min=0.0, max=max_height)
  return (height * (1.0 - jump_stance_mask(env, cycle_time)[:, :1])).sum(
    dim=1
  ) * shared.moving(env, command_name)


def jump_default_hip_pos(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  error = torch.abs(robot.data.joint_pos[:, joint_ids(robot)][:, (0, 3, 6, 9)]).sum(
    dim=1
  )
  return torch.exp(-4.0 * error)


class FeetAirTime:
  def __init__(self, cfg: RewardTermCfg, env) -> None:
    del cfg
    self._air_time = torch.zeros((env.num_envs, 4), device=env.device)
    self._last_contact = torch.zeros(
      (env.num_envs, 4), dtype=torch.bool, device=env.device
    )

  def __call__(self, env, sensor_name: str, command_name: str) -> torch.Tensor:
    contact = source_vertical_contact(env.scene[sensor_name], 1.0)
    filtered = contact | self._last_contact
    self._last_contact.copy_(contact)
    first_contact = (self._air_time > 0.0) & filtered
    self._air_time += env.step_dt
    reward = ((self._air_time - 0.5) * first_contact).sum(dim=1) * shared.moving(
      env, command_name
    )
    self._air_time *= ~filtered
    return reward

  def reset(self, env_ids=None) -> None:
    self._air_time[env_ids] = 0.0
    self._last_contact[env_ids] = False


def feet_contact_forces(
  env, sensor_name: str, max_contact_force: float
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  order = [
    sensor.primary_names.index(name)
    for name in (
      "FL_foot_collision",
      "FR_foot_collision",
      "RL_foot_collision",
      "RR_foot_collision",
    )
  ]
  return (
    (torch.linalg.vector_norm(force[:, order], dim=-1) - max_contact_force)
    .clamp(min=0.0)
    .sum(dim=1)
  )
