"""Go2 joint ordering, contact mapping and gait primitives."""

import math

import torch
from mjlab.entity import Entity
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat

JOINT_NAMES = (
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
)
SOURCE_FOOT_GEOMS = (
  "FL_foot_collision",
  "FR_foot_collision",
  "RL_foot_collision",
  "RR_foot_collision",
)


def joint_ids(robot: Entity) -> list[int]:
  ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
  if tuple(names) != JOINT_NAMES:
    raise RuntimeError(f"Go2 joint order mismatch: {names}")
  return ids


def source_contact(sensor: ContactSensor, threshold: float) -> torch.Tensor:
  force = sensor.data.force
  assert force is not None
  order = [sensor.primary_names.index(name) for name in SOURCE_FOOT_GEOMS]
  return torch.linalg.vector_norm(force[:, order], dim=-1) > threshold


def source_vertical_contact(sensor: ContactSensor, threshold: float) -> torch.Tensor:
  force = sensor.data.force
  assert force is not None
  order = [sensor.primary_names.index(name) for name in SOURCE_FOOT_GEOMS]
  return -force[:, order, 2] > threshold


def phase(env, cycle_time: float) -> torch.Tensor:
  return torch.remainder(env.episode_length_buf * env.step_dt, cycle_time) / cycle_time


def stance_mask(env, cycle_time: float) -> torch.Tensor:
  gait_phase = phase(env, cycle_time)
  return torch.stack((gait_phase < 0.5, gait_phase > 0.5), dim=1).float()


def root_euler(robot: Entity) -> torch.Tensor:
  return torch.stack(euler_xyz_from_quat(robot.data.root_link_quat_w), dim=1)


def phase_command(env, command_name: str, cycle_time: float) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  gait_phase = phase(env, cycle_time)
  return torch.cat(
    (
      torch.sin(2.0 * math.pi * gait_phase).unsqueeze(1),
      torch.cos(2.0 * math.pi * gait_phase).unsqueeze(1),
      command[:, :2] * 2.0,
      command[:, 2:3] * 0.25,
    ),
    dim=1,
  )
