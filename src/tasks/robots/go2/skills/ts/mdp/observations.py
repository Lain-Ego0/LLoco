"""Go2 TS observations matching the source's 45/187/74/309 split."""

import torch
from mjlab.entity import Entity
from mjlab.managers import ObservationTermCfg

from ...dreamwaq.mdp.observations import _actor_frame


def _terrain(env) -> torch.Tensor:
  scan = env.scene["terrain_scan"].data
  return torch.clamp(
    scan.frame_pos_w[:, 0, 2:3] - 0.5 - scan.hit_pos_w[..., 2], -1.0, 1.0
  ).mul(5.0)


def _domain(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  friction = robot.data.model.geom_friction[:, robot.indexing.geom_ids[0], 0:1]
  restitution = getattr(
    env,
    "_ts_restitution",
    getattr(env, "_dreamwaq_restitution", torch.zeros_like(friction)),
  )
  base_mass = getattr(
    env, "_ts_base_mass", getattr(env, "_cts_base_mass", torch.zeros_like(friction))
  )
  link_mass = getattr(
    env, "_ts_link_mass", torch.ones((env.num_envs, 28), device=env.device)
  )
  com = getattr(env, "_ts_base_com", torch.zeros((env.num_envs, 3), device=env.device))
  p_gain = getattr(
    env,
    "_ts_p_gain",
    getattr(env, "_dreamwaq_p_gain", torch.ones((env.num_envs, 12), device=env.device)),
  )
  d_gain = getattr(
    env,
    "_ts_d_gain",
    getattr(env, "_dreamwaq_d_gain", torch.ones((env.num_envs, 12), device=env.device)),
  )
  torque = getattr(env, "_ts_torque", torch.ones((env.num_envs, 12), device=env.device))
  return torch.cat(
    (friction, restitution, base_mass, link_mass, com, p_gain, d_gain, torque), 1
  )


class TsActor:
  def __init__(self, cfg: ObservationTermCfg, env) -> None:
    del cfg
    self._noise = True

  def __call__(self, env, command_name: str, add_noise: bool = True) -> torch.Tensor:
    frame = _actor_frame(env, command_name)
    amplitude = torch.tensor(
      # Gym noise is applied before the observation scales: angular velocity
      # noise is 0.2 * 0.25 = 0.05 (not 0.075; 0.075 belongs to dq noise,
      # whose source scale is 1.5 * 0.05).
      [0.0] * 3 + [0.05] * 3 + [0.05] * 3 + [0.01] * 12 + [0.075] * 12 + [0.0] * 12,
      device=frame.device,
    )
    return (
      frame + (2.0 * torch.rand_like(frame) - 1.0) * amplitude if add_noise else frame
    )


class TsTerrain:
  def __init__(self, cfg: ObservationTermCfg, env) -> None:
    del cfg, env

  def __call__(self, env) -> torch.Tensor:
    return _terrain(env)


class TsDomain:
  def __init__(self, cfg: ObservationTermCfg, env) -> None:
    del cfg, env

  def __call__(self, env) -> torch.Tensor:
    # The source teacher encoder receives domain labels plus four foot contacts.
    robot: Entity = env.scene["robot"]
    sensor = env.scene["feet_ground_contact"].data
    contacts = (sensor.force[..., 2] > 1.0).to(robot.data.joint_pos.dtype)
    return torch.cat((_domain(env), contacts), 1)


class TsCritic:
  def __init__(self, cfg: ObservationTermCfg, env) -> None:
    del cfg, env

  def __call__(self, env, command_name: str) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    sensor = env.scene["feet_ground_contact"].data
    contacts = (sensor.force[..., 2] > 1.0).to(robot.data.joint_pos.dtype)
    value = torch.cat(
      (
        robot.data.root_link_lin_vel_b * 2.0,
        _actor_frame(env, command_name),
        _domain(env),
        contacts,
        _terrain(env),
      ),
      1,
    )
    if value.shape[1] != 309:
      raise RuntimeError(f"TS critic observation must be 309-D, got {value.shape}")
    return value
