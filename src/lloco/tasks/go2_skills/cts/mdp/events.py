"""CTS-only domain randomization with matching privileged labels."""

import torch
from mjlab.actuator import IdealPdActuator
from mjlab.entity import Entity
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields


@requires_model_fields("geom_friction")
def randomize_friction(
  env,
  env_ids,
  low: float = 0.2,
  high: float = 1.0,
  entity_name: str = "robot",
) -> None:
  """Assign one independently sampled friction coefficient per CTS environment."""
  ids = resolve_env_ids(env, env_ids)
  robot: Entity = env.scene[entity_name]
  values = torch.empty((len(ids), 1), device=env.device).uniform_(low, high)
  env.sim.model.geom_friction[ids[:, None], robot.indexing.geom_ids, 0] = values


def randomize_pd_and_torque(
  env,
  env_ids,
  low: float = 0.9,
  high: float = 1.1,
  entity_name: str = "robot",
) -> None:
  """Apply Gym CTS's independent P/D/torque multipliers and expose labels."""
  ids = resolve_env_ids(env, env_ids)
  robot: Entity = env.scene[entity_name]
  p = torch.empty((len(ids), 12), device=env.device).uniform_(low, high)
  d = torch.empty((len(ids), 12), device=env.device).uniform_(low, high)
  torque = torch.empty((len(ids), 12), device=env.device).uniform_(low, high)
  for actuator in robot.actuators:
    if not isinstance(actuator, IdealPdActuator):
      raise TypeError(f"CTS requires IdealPdActuator, got {type(actuator).__name__}")
    assert actuator.default_stiffness is not None
    assert actuator.default_damping is not None
    target_ids = actuator.target_ids
    # Gym applies torque_multiplier after PD. Scaling both gains yields the
    # same unclipped torque and retains the original effort limit.
    actuator.set_gains(
      ids,
      kp=actuator.default_stiffness[ids] * p[:, target_ids] * torque[:, target_ids],
      kd=actuator.default_damping[ids] * d[:, target_ids] * torque[:, target_ids],
    )
  for name, values in (
    ("_dreamwaq_p_gain", p),
    ("_dreamwaq_d_gain", d),
    ("_cts_torque_multiplier", torque),
  ):
    labels = getattr(env, name, None)
    if labels is None:
      labels = torch.ones((env.num_envs, 12), device=env.device)
      setattr(env, name, labels)
    labels[ids] = values


def randomize_base_mass(
  env,
  env_ids,
  ranges: tuple[float, float],
  asset_cfg,
  operation: str = "add",
  entity_name: str = "robot",
) -> None:
  """Apply source additive base-mass DR and retain its exact 1-D label."""
  if operation != "add":
    raise ValueError("CTS source randomizes base mass additively")
  ids = resolve_env_ids(env, env_ids)
  robot: Entity = env.scene[entity_name]
  local_ids, _ = robot.find_bodies(("base_link",), preserve_order=True)
  model_ids = robot.indexing.body_ids[local_ids]
  if not hasattr(env, "_cts_default_body_mass"):
    env._cts_default_body_mass = env.sim.model.body_mass.clone()
  default_mass = env._cts_default_body_mass
  dr.body_mass(env, ids, ranges, asset_cfg=asset_cfg, operation="add")
  labels = getattr(env, "_cts_base_mass", None)
  if labels is None:
    labels = torch.zeros((env.num_envs, 1), device=env.device)
    env._cts_base_mass = labels
  labels[ids] = (
    env.sim.model.body_mass[ids[:, None], model_ids]
    - default_mass[ids[:, None], model_ids]
  )


@requires_model_fields("body_ipos", recompute=RecomputeLevel.set_const)
def randomize_base_com(
  env,
  env_ids,
  ranges,
  asset_cfg,
  operation: str = "add",
  entity_name: str = "robot",
) -> None:
  """Apply base COM DR and expose the actual offset to the CTS teacher."""
  if operation != "add":
    raise ValueError("CTS source randomizes base COM additively")
  ids = resolve_env_ids(env, env_ids)
  robot: Entity = env.scene[entity_name]
  local_ids, _ = robot.find_bodies(("base_link",), preserve_order=True)
  model_ids = robot.indexing.body_ids[local_ids]
  if not hasattr(env, "_cts_default_body_ipos"):
    env._cts_default_body_ipos = env.sim.model.body_ipos.clone()
  default_ipos = env._cts_default_body_ipos
  dr.body_com_offset(
    env, ids, ranges, asset_cfg=asset_cfg, operation="add"
  )
  labels = getattr(env, "_cts_base_com", None)
  if labels is None:
    labels = torch.zeros((env.num_envs, 3), device=env.device)
    env._cts_base_com = labels
  labels[ids] = (
    env.sim.model.body_ipos[ids[:, None], model_ids]
    - default_ipos[ids[:, None], model_ids]
  ).squeeze(1)
