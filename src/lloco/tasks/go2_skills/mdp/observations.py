"""Observations translated from ``Go2_Trot.py`` without field reordering."""

import math

import torch
from mjlab.entity import Entity
from mjlab.managers import ObservationTermCfg
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


def phase(env, cycle_time: float) -> torch.Tensor:
  return torch.remainder(env.episode_length_buf * env.step_dt, cycle_time) / cycle_time


def phase_command(env, command_name: str, cycle_time: float) -> torch.Tensor:
  gait_phase = phase(env, cycle_time)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.cat(
    (
      torch.sin(2.0 * math.pi * gait_phase).unsqueeze(1),
      torch.cos(2.0 * math.pi * gait_phase).unsqueeze(1),
      command[:, :2] * 2.0,
      command[:, 2:3] * 0.25,
    ),
    dim=1,
  )


def joint_ids(robot: Entity) -> list[int]:
  ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
  if tuple(names) != JOINT_NAMES:
    raise RuntimeError(f"Go2 joint order mismatch: {names}")
  return ids


def source_contact(sensor: ContactSensor, threshold: float) -> torch.Tensor:
  """Contacts in source FL, FR, RL, RR order.

  Isaac Gym uses world-frame force Z. The mjlab net-force sensor exposes a
  resultant vector; on this flat task its norm is the stable equivalent.
  """
  force = sensor.data.force
  assert force is not None
  order = [sensor.primary_names.index(name) for name in SOURCE_FOOT_GEOMS]
  return torch.linalg.vector_norm(force[:, order], dim=-1) > threshold


def source_vertical_contact(
  sensor: ContactSensor, threshold: float
) -> torch.Tensor:
  """Isaac Gym contact test using the world-frame vertical force component."""
  force = sensor.data.force
  assert force is not None
  order = [sensor.primary_names.index(name) for name in SOURCE_FOOT_GEOMS]
  return force[:, order, 2] > threshold


def contact_observation(env, sensor_name: str, threshold: float = 5.0) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  return source_contact(sensor, threshold).float()


def stance_mask(env, cycle_time: float) -> torch.Tensor:
  gait_phase = phase(env, cycle_time)
  return torch.stack((gait_phase < 0.5, gait_phase > 0.5), dim=1).float()


def root_euler(robot: Entity) -> torch.Tensor:
  return torch.stack(euler_xyz_from_quat(robot.data.root_link_quat_w), dim=1)


def actor_frame(env, command_name: str, cycle_time: float) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = joint_ids(robot)
  return torch.cat(
    (
      phase_command(env, command_name, cycle_time),
      robot.data.root_link_ang_vel_b * 0.25,
      root_euler(robot),
      robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids],
      robot.data.joint_vel[:, ids] * 0.05,
      env.action_manager.action,
    ),
    dim=1,
  )


def critic_frame(
  env, command_name: str, sensor_name: str, cycle_time: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = joint_ids(robot)
  return torch.cat(
    (
      phase_command(env, command_name, cycle_time),
      robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids],
      robot.data.joint_pos[:, ids],
      robot.data.joint_vel[:, ids] * 0.05,
      env.action_manager.action,
      robot.data.root_link_lin_vel_b * 2.0,
      robot.data.root_link_ang_vel_b * 0.25,
      root_euler(robot),
      stance_mask(env, cycle_time),
      contact_observation(env, sensor_name),
    ),
    dim=1,
  )


class _SourceHistory:
  """Frame-major, oldest-to-newest history with source-style zero reset."""

  frame_dim: int
  history_length: int

  def __init__(self, cfg: ObservationTermCfg, env) -> None:
    del cfg
    self._history = torch.zeros(
      env.num_envs, self.history_length, self.frame_dim, device=env.device
    )

  def _append(self, frame: torch.Tensor) -> torch.Tensor:
    self._history = torch.roll(self._history, shifts=-1, dims=1)
    self._history[:, -1] = frame
    return self._history.reshape(frame.shape[0], -1)

  def reset(self, env_ids=None) -> None:
    self._history[env_ids] = 0.0


class TrotActorHistory(_SourceHistory):
  frame_dim = 47
  history_length = 10

  def __call__(
    self, env, command_name: str, cycle_time: float, add_noise: bool
  ) -> torch.Tensor:
    frame = actor_frame(env, command_name, cycle_time)
    if add_noise:
      _, upper = single_frame_noise_bounds()
      amplitude = torch.tensor(upper, device=env.device)
      frame = frame + (2.0 * torch.rand_like(frame) - 1.0) * amplitude
    return self._append(frame)


class TrotCriticHistory(_SourceHistory):
  frame_dim = 68
  history_length = 3

  def __call__(
    self, env, command_name: str, sensor_name: str, cycle_time: float
  ) -> torch.Tensor:
    return self._append(critic_frame(env, command_name, sensor_name, cycle_time))


def jump_phase(env, cycle_time: float) -> torch.Tensor:
  """Unwrapped source Jump phase (the stance transition happens only once)."""
  return env.episode_length_buf * env.step_dt / cycle_time


def jump_stance_mask(env, cycle_time: float) -> torch.Tensor:
  gait_phase = jump_phase(env, cycle_time)
  return torch.stack((gait_phase < 0.6, gait_phase > 0.6), dim=1).float()


def jump_phase_command(env, command_name: str, cycle_time: float) -> torch.Tensor:
  gait_phase = jump_phase(env, cycle_time)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.cat(
    (
      torch.sin(2.0 * math.pi * gait_phase).unsqueeze(1),
      torch.cos(2.0 * math.pi * gait_phase).unsqueeze(1),
      command[:, :2] * 2.0,
      command[:, 2:3] * 0.25,
    ),
    dim=1,
  )


def jump_actor_frame(env, command_name: str, cycle_time: float) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = joint_ids(robot)
  return torch.cat(
    (
      jump_phase_command(env, command_name, cycle_time),
      robot.data.root_link_ang_vel_b * 0.25,
      root_euler(robot),
      robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids],
      robot.data.joint_vel[:, ids] * 0.05,
      env.action_manager.action,
    ),
    dim=1,
  )


def jump_critic_frame(
  env, command_name: str, sensor_name: str, cycle_time: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = joint_ids(robot)
  sensor: ContactSensor = env.scene[sensor_name]
  # The source samples one friction bucket per environment and applies it to all
  # robot shapes. Reading the first robot geom therefore recovers the label.
  friction = robot.data.model.geom_friction[
    :, robot.indexing.geom_ids[0], 0
  ].unsqueeze(1)
  # Go2_Jump allocates body_mass but never writes to it; its critic observes zero.
  source_body_mass = torch.zeros((env.num_envs, 1), device=env.device)
  return torch.cat(
    (
      jump_phase_command(env, command_name, cycle_time),
      robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids],
      robot.data.joint_pos[:, ids],
      robot.data.joint_vel[:, ids] * 0.05,
      env.action_manager.action,
      robot.data.root_link_lin_vel_b * 2.0,
      robot.data.root_link_ang_vel_b * 0.25,
      root_euler(robot),
      friction,
      source_body_mass,
      jump_stance_mask(env, cycle_time),
      source_vertical_contact(sensor, 5.0).float(),
    ),
    dim=1,
  )


class JumpActorHistory(_SourceHistory):
  frame_dim = 47
  history_length = 10

  def __call__(
    self, env, command_name: str, cycle_time: float, add_noise: bool
  ) -> torch.Tensor:
    frame = jump_actor_frame(env, command_name, cycle_time)
    if add_noise:
      _, upper = single_frame_noise_bounds()
      amplitude = torch.tensor(upper, device=env.device)
      frame = frame + (2.0 * torch.rand_like(frame) - 1.0) * amplitude
    return self._append(frame)


class JumpCriticHistory(_SourceHistory):
  frame_dim = 70
  history_length = 3

  def __call__(
    self, env, command_name: str, sensor_name: str, cycle_time: float
  ) -> torch.Tensor:
    return self._append(
      jump_critic_frame(env, command_name, sensor_name, cycle_time)
    )


def single_frame_noise_bounds() -> tuple[tuple[float, ...], tuple[float, ...]]:
  amplitudes = (
    [0.0] * 5
    + [0.2 * 0.25] * 3
    + [0.1] * 3
    + [0.01] * 12
    + [1.5 * 0.05] * 12
    + [0.0] * 12
  )
  return tuple(-value for value in amplitudes), tuple(amplitudes)
