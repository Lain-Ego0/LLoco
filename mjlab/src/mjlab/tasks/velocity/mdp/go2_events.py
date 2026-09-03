"""Go2-specific reset events that mirror the source Isaac Gym semantics."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_joints_by_scale(
  env,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset joints as ``default_q * U(scale_range)`` and zero/random velocity.

  Several Go2 source environments use multiplicative joint initialization
  rather than mjlab's additive ``reset_joints_by_offset`` event.  Keeping this
  as a separate event avoids changing the generic reset behavior used by other
  velocity tasks.  The result is clamped to the articulation's soft limits,
  matching the normal mjlab reset event's safety behavior.
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert default_joint_pos is not None
  assert default_joint_vel is not None
  assert soft_joint_pos_limits is not None

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)
  joint_pos = default_joint_pos[env_ids][:, joint_ids].clone()
  scales = sample_uniform(*scale_range, joint_pos.shape, env.device)
  joint_pos = joint_pos * scales
  limits = soft_joint_pos_limits[env_ids][:, joint_ids]
  joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

  joint_vel = default_joint_vel[env_ids][:, joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)
  asset.write_joint_state_to_sim(
    joint_pos.reshape(len(env_ids), -1),
    joint_vel.reshape(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )


def go2_torque_multiplier(
  env,
  env_ids: torch.Tensor | None,
  torque_multiplier_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Scale each Go2 position actuator's complete PD torque at startup.

  The source Isaac Gym controller multiplies ``kp*(q_target-q)-kd*dq`` by a
  per-environment torque multiplier after the PD calculation.  For MuJoCo's
  native position actuator this is exactly represented by multiplying the
  actuator gain and both corresponding bias coefficients by one shared sample
  per target.  It intentionally runs after the independent PD-gain event.
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  actuator_ids = asset_cfg.actuator_ids
  if isinstance(actuator_ids, list):
    actuators = [asset.actuators[i] for i in actuator_ids]
  elif isinstance(actuator_ids, slice):
    actuators = list(asset.actuators[actuator_ids])
  else:
    actuators = [asset.actuators[actuator_ids]]

  gainprm = env.sim.model.actuator_gainprm
  biasprm = env.sim.model.actuator_biasprm
  default_gainprm = env.sim.get_default_field("actuator_gainprm")
  default_biasprm = env.sim.get_default_field("actuator_biasprm")
  joint_count = asset.data.joint_pos.shape[-1]
  if not hasattr(env, "_go2_pd_kp_multiplier"):
    env._go2_pd_kp_multiplier = torch.ones(
      (env.num_envs, joint_count), device=env.device
    )
    env._go2_pd_kd_multiplier = torch.ones_like(env._go2_pd_kp_multiplier)
    env._go2_torque_multiplier = torch.ones_like(env._go2_pd_kp_multiplier)
  for actuator in actuators:
    ctrl_ids = actuator.global_ctrl_ids
    target_ids = actuator.target_ids
    n_targets = len(ctrl_ids)
    # Capture the independent PD samples before applying the complete-torque
    # multiplier.  Reading gainprm afterwards would conflate two distinct
    # source privileged-observation fields.
    env._go2_pd_kp_multiplier[env_ids[:, None], target_ids] = (
      gainprm[env_ids[:, None], ctrl_ids, 0]
      / default_gainprm[ctrl_ids, 0].clamp_min(1.0e-6)
    )
    env._go2_pd_kd_multiplier[env_ids[:, None], target_ids] = (
      -biasprm[env_ids[:, None], ctrl_ids, 2]
      / (-default_biasprm[ctrl_ids, 2]).clamp_min(1.0e-6)
    )
    multipliers = sample_uniform(
      torque_multiplier_range[0],
      torque_multiplier_range[1],
      (len(env_ids), n_targets),
      env.device,
    )
    env._go2_torque_multiplier[env_ids[:, None], target_ids] = multipliers
    gainprm[env_ids[:, None], ctrl_ids, 0] *= multipliers
    biasprm[env_ids[:, None], ctrl_ids, 1] *= multipliers
    biasprm[env_ids[:, None], ctrl_ids, 2] *= multipliers
