"""Teacher, student-history and critic observations for source CTS."""

import torch
from mjlab.entity import Entity

from ...dreamwaq.mdp.observations import _actor_frame, _joint_ids


def _heights(env) -> torch.Tensor:
  scan = env.scene["terrain_scan"].data
  return torch.clamp(
    scan.frame_pos_w[:, 0, 2:3] - 0.5 - scan.hit_pos_w[..., 2], -1.0, 1.0
  ).mul(5.0)


def _labels(env, robot: Entity) -> torch.Tensor:
  friction = robot.data.model.geom_friction[:, robot.indexing.geom_ids[0], 0:1]
  restitution = getattr(
    env, "_cts_restitution", getattr(env, "_dreamwaq_restitution", torch.zeros_like(friction))
  )
  base_mass = getattr(env, "_cts_base_mass", torch.zeros_like(friction))
  com = getattr(env, "_cts_base_com", torch.zeros((env.num_envs, 3), device=env.device))
  p_gain = getattr(
    env, "_dreamwaq_p_gain", torch.ones((env.num_envs, 12), device=env.device)
  )
  d_gain = getattr(
    env, "_dreamwaq_d_gain", torch.ones((env.num_envs, 12), device=env.device)
  )
  torque = getattr(
    env, "_cts_torque_multiplier", torch.ones((env.num_envs, 12), device=env.device)
  )
  return torch.cat(
    (friction, restitution, base_mass, com, p_gain, d_gain, torque), dim=1
  )


def _teacher(env, command_name: str, sensor_name: str) -> torch.Tensor:
  del command_name
  robot: Entity = env.scene["robot"]
  ids = _joint_ids(robot)
  sensor = env.scene[sensor_name].data
  contacts = (sensor.force[..., 2] > 1.0).to(dtype=robot.data.joint_pos.dtype)
  value = torch.cat((_labels(env, robot), contacts, _heights(env)), dim=1)
  if value.shape[1] != 233:
    raise RuntimeError(f"CTS teacher observation must be 233-D, got {value.shape}")
  del ids
  return value


class CtsTeacherObservation:
  def __init__(self, cfg, env):
    del cfg, env

  def __call__(self, env, command_name: str, sensor_name: str) -> torch.Tensor:
    return _teacher(env, command_name, sensor_name)


class CtsCriticObservation:
  def __init__(self, cfg, env):
    del cfg, env

  def __call__(self, env, command_name: str, sensor_name: str) -> torch.Tensor:
    value = torch.cat(
      (_actor_frame(env, command_name), _teacher(env, command_name, sensor_name)), dim=1
    )
    if value.shape[1] != 278:
      raise RuntimeError(f"CTS critic observation must be 278-D, got {value.shape}")
    return value
