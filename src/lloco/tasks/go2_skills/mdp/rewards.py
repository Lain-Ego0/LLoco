"""Reward equations translated from the registered Trot and Jump tasks."""

import torch
from mjlab.entity import Entity
from mjlab.managers import RewardTermCfg
from mjlab.sensor import ContactSensor

from .observations import (
  joint_ids,
  jump_stance_mask,
  phase,
  source_contact,
  source_vertical_contact,
  stance_mask,
)


def command(env, command_name: str) -> torch.Tensor:
  value = env.command_manager.get_command(command_name)
  assert value is not None
  return value


def moving(env, command_name: str) -> torch.Tensor:
  return torch.linalg.vector_norm(command(env, command_name)[:, :3], dim=1) > 0.1


def trot_match(env, sensor_name: str, cycle_time: float) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  contact = source_contact(sensor, 5.0)
  stance = stance_mask(env, cycle_time).bool()
  return (
    (contact[:, 0] == contact[:, 3])
    & (contact[:, 1] == contact[:, 2])
    & (contact[:, 0] == stance[:, 0])
    & (contact[:, 1] == stance[:, 1])
  )


def source_trot_gate(
  env, sensor_name: str, command_name: str, cycle_time: float
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  is_moving = moving(env, command_name)
  match = trot_match(env, sensor_name, cycle_time).float()
  all_contact = source_contact(sensor, 0.1).sum(dim=1) == 4
  # The scalar batch mean is intentional: this is exactly what self.trot stores.
  return match.mean() * is_moving + all_contact * ~is_moving


def trot(env, sensor_name: str, command_name: str, cycle_time: float) -> torch.Tensor:
  return trot_match(env, sensor_name, cycle_time) * moving(env, command_name)


def tracking_lin_vel(
  env, sensor_name: str, command_name: str, cycle_time: float, sigma: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  error = torch.square(
    command(env, command_name)[:, :2] - robot.data.root_link_lin_vel_b[:, :2]
  ).sum(dim=1)
  gate = source_trot_gate(env, sensor_name, command_name, cycle_time) > 0.7
  return torch.exp(-error / sigma) * gate


def tracking_ang_vel(
  env, sensor_name: str, command_name: str, cycle_time: float, sigma: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  error = torch.square(
    command(env, command_name)[:, 2] - robot.data.root_link_ang_vel_b[:, 2]
  )
  gate = source_trot_gate(env, sensor_name, command_name, cycle_time) > 0.7
  return torch.exp(-error / sigma) * gate


def lin_vel_z(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.square(robot.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.square(robot.data.root_link_ang_vel_b[:, :2]).sum(dim=1)


def orientation(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.square(robot.data.projected_gravity_b[:, :2]).sum(dim=1)


def torques(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.square(robot.data.qfrc_actuator[:, joint_ids(robot)]).sum(dim=1)


class DofAcceleration:
  """Source finite difference using the previous policy-step joint velocity."""

  def __init__(self, cfg: RewardTermCfg, env) -> None:
    del cfg
    self._last_velocity = torch.zeros(
      env.num_envs, len(joint_ids(env.scene["robot"])), device=env.device
    )

  def __call__(self, env) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    velocity = robot.data.joint_vel[:, joint_ids(robot)]
    value = torch.square((self._last_velocity - velocity) / env.step_dt).sum(dim=1)
    self._last_velocity.copy_(velocity)
    return value

  def reset(self, env_ids=None) -> None:
    self._last_velocity[env_ids] = 0.0


def collision(env, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  return (torch.linalg.vector_norm(force, dim=-1) > 0.1).float().sum(dim=1)


def action_rate(env) -> torch.Tensor:
  return torch.square(env.action_manager.prev_action - env.action_manager.action).sum(1)


def stand_still(env, command_name: str) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = joint_ids(robot)
  error = torch.abs(
    robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
  ).sum(dim=1)
  return error * ~moving(env, command_name)


def base_height(env, target_height: float) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.square(robot.data.root_link_pos_w[:, 2] - target_height)


def default_hip_pos(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  pos = robot.data.joint_pos[:, joint_ids(robot)]
  return torch.abs(pos[:, (0, 3, 6, 9)]).sum(dim=1)


def default_pos(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = joint_ids(robot)
  return torch.abs(
    robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
  ).sum(dim=1)


def contact_without_command(env, sensor_name: str, command_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  all_contact = source_contact(sensor, 0.1).sum(dim=1) == 4
  return all_contact * ~moving(env, command_name)


def feet_clearance(
  env, command_name: str, cycle_time: float, target_foot_height: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  site_ids, names = robot.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)
  if tuple(names) != ("FL", "FR", "RL", "RR"):
    raise RuntimeError(f"Go2 foot site order mismatch: {names}")
  height = robot.data.site_pos_w[:, site_ids, 2] - 0.02
  diagonal_a = height[:, (0, 3)]
  diagonal_b = height[:, (1, 2)]
  swing = 1.0 - stance_mask(env, cycle_time)
  target = (
    torch.abs(torch.sin(2.0 * torch.pi * phase(env, cycle_time)))
    * target_foot_height
  ).unsqueeze(1).repeat(1, 2)
  reward = torch.exp(
    -10.0 * (torch.abs(diagonal_a - target) * swing[:, 0:1]).sum(dim=1)
  )
  reward += torch.exp(
    -10.0 * (torch.abs(diagonal_b - target) * swing[:, 1:2]).sum(dim=1)
  )
  return reward * moving(env, command_name)


# Jump uses a different family of reward equations despite sharing names with Trot.


def jump_tracking_lin_vel(
  env, command_name: str, sigma: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  cmd = command(env, command_name)
  is_moving = torch.linalg.vector_norm(cmd[:, :3], dim=1) > 0.1
  moving_error = torch.square(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2]).sum(
    dim=1
  )
  return torch.exp(-moving_error / sigma) * is_moving + torch.exp(
    -torch.linalg.vector_norm(robot.data.root_link_lin_vel_b[:, :2], dim=1) / sigma
  ) * ~is_moving


def jump_tracking_ang_vel(
  env, command_name: str, sigma: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  cmd = command(env, command_name)
  is_moving = torch.linalg.vector_norm(cmd[:, :3], dim=1) > 0.1
  moving_error = torch.square(cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-moving_error / sigma) * is_moving + torch.exp(
    -torch.abs(robot.data.root_link_ang_vel_b[:, 2]) / sigma
  ) * ~is_moving


def jump_lin_vel_z(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.exp(-torch.abs(robot.data.root_link_lin_vel_b[:, 2]))


def jump_ang_vel_xy(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.exp(
    -torch.linalg.vector_norm(torch.abs(robot.data.root_link_ang_vel_b[:, :2]), dim=1)
  )


def jump_orientation(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.exp(
    -10.0 * torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=1)
  )


def jump_base_height(
  env, command_name: str, target_height: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.exp(
    -10.0 * torch.abs(robot.data.root_link_pos_w[:, 2] - target_height)
  ) * ~moving(env, command_name)


def absolute_torques(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.abs(robot.data.qfrc_actuator[:, joint_ids(robot)]).sum(dim=1)


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
  expected = jump_stance_mask(env, cycle_time)[:, 0].bool()
  return all_equal & (contact[:, 3] == expected) & moving(env, command_name)


def jump_feet_clearance(
  env, command_name: str, cycle_time: float, max_height: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  site_ids, names = robot.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)
  if tuple(names) != ("FL", "FR", "RL", "RR"):
    raise RuntimeError(f"Go2 foot site order mismatch: {names}")
  feet_height = (robot.data.site_pos_w[:, site_ids, 2] - 0.02).clamp(
    min=0.0, max=max_height
  )
  swing = 1.0 - jump_stance_mask(env, cycle_time)[:, :1]
  return (feet_height * swing).sum(dim=1) * moving(env, command_name)


def jump_default_hip_pos(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  pos = robot.data.joint_pos[:, joint_ids(robot)]
  hip_error = torch.abs(pos[:, (0, 3, 6, 9)]).sum(dim=1)
  return torch.exp(-4.0 * hip_error)


class FeetAirTime:
  """Source contact-filtered foot air-time state and first-contact reward."""

  def __init__(self, cfg: RewardTermCfg, env) -> None:
    del cfg
    self._air_time = torch.zeros((env.num_envs, 4), device=env.device)
    self._last_contact = torch.zeros(
      (env.num_envs, 4), dtype=torch.bool, device=env.device
    )

  def __call__(self, env, sensor_name: str, command_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    contact = source_vertical_contact(sensor, 1.0)
    contact_filtered = contact | self._last_contact
    self._last_contact.copy_(contact)
    first_contact = (self._air_time > 0.0) & contact_filtered
    self._air_time += env.step_dt
    reward = ((self._air_time - 0.5) * first_contact).sum(dim=1)
    reward *= moving(env, command_name)
    self._air_time *= ~contact_filtered
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
  order = [sensor.primary_names.index(name) for name in (
    "FL_foot_collision",
    "FR_foot_collision",
    "RL_foot_collision",
    "RR_foot_collision",
  )]
  magnitude = torch.linalg.vector_norm(force[:, order], dim=-1)
  return (magnitude - max_contact_force).clamp(min=0.0).sum(dim=1)
