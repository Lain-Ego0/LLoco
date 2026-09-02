"""Go2 action terms that preserve source per-decimation delay semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat


@dataclass(kw_only=True)
class Go2DelayedJointPositionActionCfg(JointPositionActionCfg):
  """Joint-position action with a random substep switch point.

  The source Go2 tasks sample one integer in ``[0, decimation)`` at every
  policy step, keep applying the previous target before that point, and apply
  the new target afterwards.  ``delay=False`` makes this a drop-in equivalent
  of :class:`JointPositionActionCfg` for tasks that do not use the source
  latency model.
  """

  delay: bool = True
  delay_mode: Literal["switch", "buffer"] = "switch"
  delay_steps_range: tuple[int, int] | None = None
  observation_latency: bool = False
  # Backflip's source observer uses projected gravity; the other special
  # action observers use Euler angles in their six-dimensional IMU block.
  observation_imu_orientation: Literal["euler", "gravity"] = "euler"
  observation_motor_latency_range: tuple[int, int] = (1, 3)
  observation_imu_latency_range: tuple[int, int] = (1, 3)

  def build(self, env) -> Go2DelayedJointPositionAction:
    return Go2DelayedJointPositionAction(self, env)


class Go2DelayedJointPositionAction(JointPositionAction):
  """Apply delayed targets while retaining raw action history for rewards."""

  cfg: Go2DelayedJointPositionActionCfg

  def __init__(self, cfg: Go2DelayedJointPositionActionCfg, env) -> None:
    super().__init__(cfg, env)
    self._decimation = int(env.cfg.decimation)
    self._delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self._substep = 0
    self._previous_processed = torch.zeros_like(self._processed_actions)
    self._delay_enabled = bool(cfg.delay)
    self._delay_mode = cfg.delay_mode
    if self._delay_mode not in ("switch", "buffer"):
      raise ValueError(f"Unknown Go2 delay mode: {self._delay_mode!r}")
    if cfg.delay_steps_range is None:
      self._delay_range = (0, max(0, self._decimation - 1))
    else:
      self._delay_range = cfg.delay_steps_range
    low, high = self._delay_range
    if not 0 <= low <= high < self._decimation:
      raise ValueError(
        "delay_steps_range must satisfy 0 <= low <= high < env decimation"
      )
    self._action_buffer = torch.zeros(
      self.num_envs, self.action_dim, high + 1, device=self.device
    )
    self._default_processed = self._entity.data.default_joint_pos[:, self._target_ids].clone()
    self._previous_processed[:] = self._default_processed
    self._action_buffer[:] = self._default_processed.unsqueeze(-1)
    self._observation_latency = bool(cfg.observation_latency)
    self._obs_motor_latency_steps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._obs_imu_latency_steps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    motor_low, motor_high = cfg.observation_motor_latency_range
    imu_low, imu_high = cfg.observation_imu_latency_range
    if not (0 <= motor_low <= motor_high and 0 <= imu_low <= imu_high):
      raise ValueError("Observation latency ranges must satisfy 0 <= low <= high")
    if cfg.observation_imu_orientation not in ("euler", "gravity"):
      raise ValueError(
        "observation_imu_orientation must be 'euler' or 'gravity'"
      )
    self._obs_motor_latency_buffer = torch.zeros(
      self.num_envs, self.action_dim * 2, motor_high + 1, device=self.device
    )
    self._obs_imu_latency_buffer = torch.zeros(
      self.num_envs, 6, imu_high + 1, device=self.device
    )
    self._obs_motor_latency_range = (motor_low, motor_high)
    self._obs_imu_latency_range = (imu_low, imu_high)
    # Observation terms can retrieve this state through the action manager;
    # keeping it on the term avoids a second global per-environment manager.

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    if self._delay_enabled and self._delay_mode == "switch":
      self._delay_steps = torch.randint(
        self._delay_range[0], self._delay_range[1] + 1,
        (self.num_envs,), device=self.device
      )
    elif not self._delay_enabled:
      self._delay_steps.zero_()
    self._substep = 0

  def apply_actions(self) -> None:
    if self._delay_enabled and self._delay_mode == "buffer":
      self._action_buffer[:, :, 1:] = self._action_buffer[:, :, :-1].clone()
      self._action_buffer[:, :, 0] = self._processed_actions
      selected = self._action_buffer[
        torch.arange(self.num_envs, device=self.device),
        :,
        self._delay_steps,
      ]
    elif self._delay_enabled:
      use_current = self._substep >= self._delay_steps
      selected = torch.where(
        use_current.unsqueeze(-1), self._processed_actions, self._previous_processed
      )
    else:
      selected = self._processed_actions
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = selected - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
    self._previous_processed[:] = selected
    self._substep = (self._substep + 1) % self._decimation

  def post_physics_step(self) -> None:
    """Record source-style motor/IMU samples after one physics substep."""
    if not self._observation_latency:
      return
    q = self._entity.data.joint_pos[:, self._target_ids] - self._default_processed
    dq = self._entity.data.joint_vel[:, self._target_ids] * 0.05
    self._obs_motor_latency_buffer[:, :, 1:] = self._obs_motor_latency_buffer[:, :, :-1].clone()
    self._obs_motor_latency_buffer[:, :, 0] = torch.cat((q, dq), dim=-1)
    if self.cfg.observation_imu_orientation == "gravity":
      orientation = self._entity.data.projected_gravity_b
    else:
      roll, pitch, yaw = euler_xyz_from_quat(self._entity.data.root_link_quat_w)
      orientation = torch.stack((roll, pitch, yaw), dim=-1)
    imu = torch.cat((self._entity.data.root_link_ang_vel_b * 0.25, orientation), dim=-1)
    self._obs_imu_latency_buffer[:, :, 1:] = self._obs_imu_latency_buffer[:, :, :-1].clone()
    self._obs_imu_latency_buffer[:, :, 0] = imu

  def get_delayed_motor_observation(self) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return delayed ``(q_rel, dq_scaled)`` or ``None`` when disabled."""
    if not self._observation_latency:
      return None
    ids = torch.arange(self.num_envs, device=self.device)
    values = self._obs_motor_latency_buffer[ids, :, self._obs_motor_latency_steps]
    return values[:, : self.action_dim], values[:, self.action_dim :]

  def get_delayed_imu_observation(self) -> torch.Tensor | None:
    """Return delayed ``ang_vel*0.25 || orientation`` or ``None``."""
    if not self._observation_latency:
      return None
    ids = torch.arange(self.num_envs, device=self.device)
    return self._obs_imu_latency_buffer[ids, :, self._obs_imu_latency_steps]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    target_ids = slice(None) if env_ids is None else env_ids
    self._previous_processed[target_ids] = self._default_processed[target_ids]
    if self._delay_enabled and self._delay_mode == "buffer":
      count = self._delay_steps[target_ids].shape[0]
      self._delay_steps[target_ids] = torch.randint(
        self._delay_range[0], self._delay_range[1] + 1,
        (count,), device=self.device,
      )
    else:
      self._delay_steps[target_ids] = 0
    self._action_buffer[target_ids] = self._default_processed[target_ids].unsqueeze(-1)
    if self._observation_latency:
      self._obs_motor_latency_buffer[target_ids] = 0.0
      self._obs_imu_latency_buffer[target_ids] = 0.0
      count = self._obs_motor_latency_steps[target_ids].shape[0]
      motor_low, motor_high = self._obs_motor_latency_range
      imu_low, imu_high = self._obs_imu_latency_range
      self._obs_motor_latency_steps[target_ids] = torch.randint(
        motor_low, motor_high + 1, (count,), device=self.device
      )
      self._obs_imu_latency_steps[target_ids] = torch.randint(
        imu_low, imu_high + 1, (count,), device=self.device
      )
