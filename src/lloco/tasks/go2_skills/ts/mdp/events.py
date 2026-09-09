"""Gym TS domain randomization expressed through mjlab event APIs."""

import torch
from mjlab.actuator import IdealPdActuator
from mjlab.entity import Entity
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields


def _ids(env, env_ids):
  return resolve_env_ids(env, env_ids)


@requires_model_fields("geom_friction")
def friction(env, env_ids, low=0.05, high=3.0, entity_name="robot", num_buckets=None):
  del num_buckets
  ids = _ids(env, env_ids)
  robot: Entity = env.scene[entity_name]
  value = torch.empty((len(ids), 1), device=env.device).uniform_(low, high)
  env.sim.model.geom_friction[ids[:, None], robot.indexing.geom_ids, 0] = value


def restitution(env, env_ids, low=0.0, high=0.5, attribute_name="_ts_restitution"):
  ids = _ids(env, env_ids)
  value = torch.empty((len(ids), 1), device=env.device).uniform_(low, high)
  setattr(
    env,
    attribute_name,
    getattr(env, attribute_name, torch.zeros((env.num_envs, 1), device=env.device)),
  )
  getattr(env, attribute_name)[ids] = value


def base_mass(env, env_ids, ranges, asset_cfg: SceneEntityCfg, operation="add"):
  ids = _ids(env, env_ids)
  robot: Entity = env.scene[asset_cfg.name]
  if not hasattr(env, "_ts_default_mass"):
    env._ts_default_mass = env.sim.model.body_mass.clone()
  dr.body_mass(env, ids, ranges, asset_cfg=asset_cfg, operation=operation)
  body_ids, _ = robot.find_bodies(("base_link",), preserve_order=True)
  model_ids = robot.indexing.body_ids[body_ids]
  value = (
    env.sim.model.body_mass[ids[:, None], model_ids]
    - env._ts_default_mass[ids[:, None], model_ids]
  )
  env._ts_base_mass = getattr(
    env, "_ts_base_mass", torch.zeros((env.num_envs, 1), device=env.device)
  )
  env._ts_base_mass[ids] = value


@requires_model_fields("body_ipos", recompute=RecomputeLevel.set_const)
def base_com(env, env_ids, ranges, asset_cfg: SceneEntityCfg, operation="add"):
  ids = _ids(env, env_ids)
  robot: Entity = env.scene[asset_cfg.name]
  if not hasattr(env, "_ts_default_ipos"):
    env._ts_default_ipos = env.sim.model.body_ipos.clone()
  dr.body_com_offset(env, ids, ranges, asset_cfg=asset_cfg, operation=operation)
  body_ids, _ = robot.find_bodies(("base_link",), preserve_order=True)
  model_ids = robot.indexing.body_ids[body_ids]
  value = (
    env.sim.model.body_ipos[ids[:, None], model_ids]
    - env._ts_default_ipos[ids[:, None], model_ids]
  ).squeeze(1)
  env._ts_base_com = getattr(
    env, "_ts_base_com", torch.zeros((env.num_envs, 3), device=env.device)
  )
  env._ts_base_com[ids] = value


def link_mass(env, env_ids, ranges, asset_cfg: SceneEntityCfg, operation="scale"):
  ids = _ids(env, env_ids)
  robot: Entity = env.scene[asset_cfg.name]
  if not hasattr(env, "_ts_default_mass"):
    env._ts_default_mass = env.sim.model.body_mass.clone()
  dr.body_mass(env, ids, ranges, asset_cfg=asset_cfg, operation=operation)
  body_ids, _ = robot.find_bodies(asset_cfg.body_names, preserve_order=True)
  model_ids = robot.indexing.body_ids[body_ids]
  ratio = (
    env.sim.model.body_mass[ids[:, None], model_ids]
    / env._ts_default_mass[ids[:, None], model_ids]
  )
  env._ts_link_mass = getattr(
    env, "_ts_link_mass", torch.ones((env.num_envs, 28), device=env.device)
  )
  # The public Go2 MJCF exposes 12 articulated leg bodies; Gym's URDF label
  # contains 28 link masses.  Preserve the available ratios and leave the
  # remaining fixed links at their neutral ratio.
  env._ts_link_mass[ids] = 1.0
  env._ts_link_mass[ids, : ratio.shape[1]] = ratio


def pd_and_torque(env, env_ids, low=0.8, high=1.2, entity_name="robot"):
  ids = _ids(env, env_ids)
  robot: Entity = env.scene[entity_name]
  p = torch.empty((len(ids), 12), device=env.device).uniform_(low, high)
  d = torch.empty_like(p).uniform_(low, high)
  torque = torch.empty_like(p).uniform_(low, high)
  for actuator in robot.actuators:
    if not isinstance(actuator, IdealPdActuator):
      raise TypeError("TS requires IdealPdActuator")
    targets = actuator.target_ids
    actuator.set_gains(
      ids,
      kp=actuator.default_stiffness[ids] * p[:, targets] * torque[:, targets],
      kd=actuator.default_damping[ids] * d[:, targets] * torque[:, targets],
    )
  for name, value in (("_ts_p_gain", p), ("_ts_d_gain", d), ("_ts_torque", torque)):
    labels = getattr(env, name, torch.ones((env.num_envs, 12), device=env.device))
    labels[ids] = value
    setattr(env, name, labels)
