"""Trot-specific reward equations."""

import torch
from mjlab.entity import Entity
from mjlab.sensor import ContactSensor

from ...shared import rl as shared
from ...shared.contacts import joint_ids, phase, source_contact, stance_mask

DofAcceleration = shared.DofAcceleration
action_rate = shared.action_rate
ang_vel_xy = shared.ang_vel_xy_squared
collision = shared.collision
contact_without_command = shared.contact_without_command
default_pos = shared.default_pos
lin_vel_z = shared.lin_vel_z_squared
orientation = shared.orientation_squared
stand_still = shared.stand_still
torques = shared.torques_squared


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
  is_moving = shared.moving(env, command_name)
  match = trot_match(env, sensor_name, cycle_time).float()
  all_contact = source_contact(sensor, 0.1).sum(dim=1) == 4
  return match.mean() * is_moving + all_contact * ~is_moving


def trot(env, sensor_name: str, command_name: str, cycle_time: float) -> torch.Tensor:
  return trot_match(env, sensor_name, cycle_time) * shared.moving(env, command_name)


def tracking_lin_vel(
  env, sensor_name: str, command_name: str, cycle_time: float, sigma: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  error = torch.square(
    shared.command(env, command_name)[:, :2] - robot.data.root_link_lin_vel_b[:, :2]
  ).sum(dim=1)
  return torch.exp(-error / sigma) * (
    source_trot_gate(env, sensor_name, command_name, cycle_time) > 0.7
  )


def tracking_ang_vel(
  env, sensor_name: str, command_name: str, cycle_time: float, sigma: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  error = torch.square(
    shared.command(env, command_name)[:, 2] - robot.data.root_link_ang_vel_b[:, 2]
  )
  return torch.exp(-error / sigma) * (
    source_trot_gate(env, sensor_name, command_name, cycle_time) > 0.7
  )


def base_height(env, target_height: float) -> torch.Tensor:
  return torch.square(env.scene["robot"].data.root_link_pos_w[:, 2] - target_height)


def default_hip_pos(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.abs(robot.data.joint_pos[:, joint_ids(robot)][:, (0, 3, 6, 9)]).sum(
    dim=1
  )


def feet_clearance(
  env, command_name: str, cycle_time: float, target_foot_height: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids, names = robot.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)
  if tuple(names) != ("FL", "FR", "RL", "RR"):
    raise RuntimeError(f"Go2 foot site order mismatch: {names}")
  height = robot.data.site_pos_w[:, ids, 2] - 0.02
  swing = 1.0 - stance_mask(env, cycle_time)
  target = (
    (torch.abs(torch.sin(2.0 * torch.pi * phase(env, cycle_time))) * target_foot_height)
    .unsqueeze(1)
    .repeat(1, 2)
  )
  reward = torch.exp(
    -10.0 * (torch.abs(height[:, (0, 3)] - target) * swing[:, :1]).sum(dim=1)
  )
  reward += torch.exp(
    -10.0 * (torch.abs(height[:, (1, 2)] - target) * swing[:, 1:2]).sum(dim=1)
  )
  return reward * shared.moving(env, command_name)
