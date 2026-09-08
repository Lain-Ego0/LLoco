"""DreamWaQ reward formulas copied from ``Go2_DreamWaQ.py``."""

import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply_inverse

from ...shared.contacts import JOINT_NAMES


def _ids(robot: Entity) -> list[int]:
  ids, _ = robot.find_joints(JOINT_NAMES, preserve_order=True)
  return ids


def tracking_lin_vel(env, command_name: str, sigma: float) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.square(command[:, :2] - env.scene["robot"].data.root_link_lin_vel_b[:, :2]).sum(1)
  return torch.exp(-error / sigma)


def tracking_ang_vel(env, command_name: str, sigma: float) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.square(command[:, 2] - env.scene["robot"].data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / sigma)


def base_height(env, target_height: float) -> torch.Tensor:
  # Gym CTS averages a dense local probe below the base.  The available 17x11
  # ray grid is spaced at 0.1 m, so its central 3x3 samples are the closest
  # equivalent; averaging the entire 1.6x1.0 m scan incorrectly rewards a
  # crouched robot on slopes.
  scan = env.scene["terrain_scan"].data
  hits = scan.hit_pos_w[..., 2].reshape(env.num_envs, 17, 11)
  terrain_z = hits[:, 7:10, 4:7].mean((1, 2))
  return torch.square(env.scene["robot"].data.root_link_pos_w[:, 2] - terrain_z - target_height)


def dof_acc(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = _ids(robot)
  last = getattr(env, "_dreamwaq_last_dof_vel", None)
  if last is None:
    last = torch.zeros_like(robot.data.joint_vel[:, ids])
    env._dreamwaq_last_dof_vel = last
  value = torch.square((last - robot.data.joint_vel[:, ids]) / env.step_dt).sum(1)
  last.copy_(robot.data.joint_vel[:, ids])
  return value


def lin_vel_z(env) -> torch.Tensor:
  return torch.square(env.scene["robot"].data.root_link_lin_vel_b[:, 2])


def ang_vel_xy(env) -> torch.Tensor:
  return torch.square(env.scene["robot"].data.root_link_ang_vel_b[:, :2]).sum(1)


def orientation(env) -> torch.Tensor:
  return torch.square(env.scene["robot"].data.projected_gravity_b[:, :2]).sum(1)


def torques(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.square(robot.data.qfrc_actuator[:, _ids(robot)]).sum(1)


def action_rate(env) -> torch.Tensor:
  return torch.square(env.action_manager.action - env.action_manager.prev_action).sum(1)


def dof_pos_limits(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = _ids(robot)
  position = robot.data.joint_pos[:, ids]
  limits = robot.data.soft_joint_pos_limits[:, ids]
  return (-(position - limits[..., 0]).clamp(max=0.0) + (position - limits[..., 1]).clamp(min=0.0)).sum(1)


def action_smoothness(env) -> torch.Tensor:
  previous = env.action_manager.prev_action
  before_previous = getattr(env, "_dreamwaq_last_last_action", None)
  if before_previous is None:
    before_previous = torch.zeros_like(previous)
    env._dreamwaq_last_last_action = before_previous
  value = torch.square(env.action_manager.action - 2 * previous + before_previous).sum(1)
  before_previous.copy_(previous)
  return value


def collision(env, sensor_name: str) -> torch.Tensor:
  force = env.scene[sensor_name].data.force
  assert force is not None
  return (torch.linalg.vector_norm(force, dim=-1) > 0.1).float().sum(1)


def stumble(env, sensor_name: str) -> torch.Tensor:
  force = env.scene[sensor_name].data.force
  assert force is not None
  return (torch.linalg.vector_norm(force[..., :2], dim=-1) > 5 * torch.abs(force[..., 2])).any(1).float()


def foot_clearance(env, sensor_name: str) -> torch.Tensor:
  # This is the source CTS calculation in the base frame, rather than a
  # terrain-relative/world-frame proxy.
  del sensor_name
  robot: Entity = env.scene["robot"]
  sites, _ = robot.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)
  pos = robot.data.site_pos_w[:, sites]
  vel = robot.data.site_vel_w[:, sites, :3]
  root = robot.data.root_link_pos_w[:, None]
  root_vel = robot.data.root_link_lin_vel_w[:, None]
  quat = robot.data.root_link_quat_w[:, None].expand(-1, len(sites), -1)
  pos_b = quat_apply_inverse(
    quat.reshape(-1, 4), (pos - root).reshape(-1, 3)
  ).reshape_as(pos)
  vel_b = quat_apply_inverse(
    quat.reshape(-1, 4), (vel - root_vel).reshape(-1, 3)
  ).reshape_as(vel)
  height_error = torch.square(pos_b[..., 2] + 0.2)
  lateral_speed = torch.linalg.vector_norm(vel_b[..., :2], dim=-1)
  return (height_error * lateral_speed).sum(1)
