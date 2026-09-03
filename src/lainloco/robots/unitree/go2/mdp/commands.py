"""Go2-specific command terms used by the special-action tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class Go2TriggeredCommandCfg(UniformVelocityCommandCfg):
  """Command configuration with a delayed one-shot jump trigger.

  The source Spring-Jump and BackFlip environments keep a three-component
  command ``(vx, vy, trigger)``.  On reset the trigger is zero and is changed
  to one at a randomly selected control step.  It is intentionally represented
  by a command term rather than an event so the state is reset per environment
  and is available to observations/rewards through the command manager.
  """

  trigger_steps: tuple[int, int] = (50, 60)
  initial_lin_vel_x: tuple[float, float] = (0.8, 1.2)
  shared_initial_lin_vel_x: bool = False
  contact_sensor_name: str = "feet_ground_contact"
  # Source ``check_jump`` filters vertical foot force at 1 N.
  contact_threshold: float = 1.0
  push_towards_goal: bool = False
  push_probability: float = 0.8
  push_probability_decay_interval_steps: int = 24 * 50
  push_probability_decay: float = 0.1
  upward_push_range: tuple[float, float] = (0.0, 0.0)
  pitch_push_range: tuple[float, float] = (0.0, 0.0)

  def build(self, env: ManagerBasedRlEnv) -> Go2TriggeredCommand:
    return Go2TriggeredCommand(self, env)


class Go2TriggeredCommand(UniformVelocityCommand):
  """Source-compatible delayed jump command and per-env flight state."""

  cfg: Go2TriggeredCommandCfg  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(self, cfg: Go2TriggeredCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    low, high = cfg.trigger_steps
    if low < 0 or high <= low:
      raise ValueError(f"Invalid trigger_steps={cfg.trigger_steps}")
    self.trigger_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.triggered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.was_in_flight = torch.zeros_like(self.triggered)
    self.has_jumped = torch.zeros_like(self.triggered)
    self.last_contact = torch.zeros(
      (self.num_envs, 4), dtype=torch.bool, device=self.device
    )
    self.start_pos = torch.zeros((self.num_envs, 2), device=self.device)
    self.landing_pos = torch.zeros((self.num_envs, 2), device=self.device)
    self.max_height = torch.zeros(self.num_envs, device=self.device)
    self.max_pitch_ang_vel = torch.zeros(self.num_envs, device=self.device)
    self._pending_start_capture = torch.zeros_like(self.triggered)
    self._push_applied = torch.zeros_like(self.triggered)
    self._postflight_push_applied = torch.zeros_like(self.triggered)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    low, high = self.cfg.trigger_steps
    self.trigger_step[env_ids] = torch.randint(
      low,
      # Match the source ``torch.randint`` call: the upper bound is
      # exclusive, so (50, 60) produces control steps 50 through 59.
      high,
      (len(env_ids),),
      device=self.device,
    )
    self.triggered[env_ids] = False
    self.was_in_flight[env_ids] = False
    self.has_jumped[env_ids] = False
    self.last_contact[env_ids] = False
    self.start_pos[env_ids] = self.robot.data.root_link_pos_w[env_ids, :2]
    self.landing_pos[env_ids] = self.start_pos[env_ids]
    self.max_height[env_ids] = 0.0
    self.max_pitch_ang_vel[env_ids] = 0.0
    self._pending_start_capture[env_ids] = True
    self._push_applied[env_ids] = False
    self._postflight_push_applied[env_ids] = False

    sample_count = 1 if self.cfg.shared_initial_lin_vel_x else len(env_ids)
    sampler = torch.empty(sample_count, device=self.device).uniform_(
      *self.cfg.initial_lin_vel_x
    )
    self.vel_command_b[env_ids, 0] = sampler[0] if sample_count == 1 else sampler
    self.vel_command_b[env_ids, 1] = 0.0
    self.vel_command_b[env_ids, 2] = 0.0
    self._apply_source_command_masks(env_ids)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    # Keep the ordinary command implementation's optional standing/world
    # handling, then apply the source trigger and update the flight state.
    super()._update_command(env_ids)
    if env_ids is None:
      ids = torch.arange(self.num_envs, device=self.device)
    else:
      ids = env_ids
    if len(ids) == 0:
      return

    # Reset events write qpos before the environment's reset-path forward().
    # Capture the actual post-reset position here, after that forward, rather
    # than relying on the stale kinematic cache seen by _resample_command().
    capture_ids = ids[self._pending_start_capture[ids]]
    if len(capture_ids) > 0:
      self.start_pos[capture_ids] = self.robot.data.root_link_pos_w[capture_ids, :2]
      self.landing_pos[capture_ids] = self.start_pos[capture_ids]
      self._pending_start_capture[capture_ids] = False

    trigger_now = (~self.triggered[ids]) & (
      self._env.episode_length_buf[ids] >= self.trigger_step[ids]
    )
    trigger_ids = ids[trigger_now]
    if len(trigger_ids) > 0:
      self.vel_command_b[trigger_ids, 2] = 1.0
      self.triggered[trigger_ids] = True
      self._apply_trigger_push(trigger_ids)

    sensor = self._env.scene[self.cfg.contact_sensor_name]
    force = sensor.data.force
    if force is not None:
      if force.ndim != 3 or force.shape[-1] != 3 or force.shape[-2] != 4:
        raise ValueError(
          f"{self.cfg.contact_sensor_name} force must be [B, 4, 3], got {tuple(force.shape)}"
        )
      contact = force[..., 2] > self.cfg.contact_threshold
    else:
      found = sensor.data.found
      if found is None or found.shape[-1] != 4:
        raise ValueError(
          f"{self.cfg.contact_sensor_name} must expose four foot contacts, got "
          f"{None if found is None else tuple(found.shape)}"
        )
      contact = found > 0
    contact_filt = contact | self.last_contact
    command_active = self.vel_command_b[:, 2] > 0.0
    ids_active = ids[command_active[ids]]
    if len(ids_active) > 0:
      airborne = (~contact_filt[ids_active]).all(dim=1)
      self.was_in_flight[ids_active[airborne]] = True
      landed = contact_filt[ids_active].any(dim=1) & self.was_in_flight[ids_active]
      new_landings = ids_active[landed & ~self.has_jumped[ids_active]]
      if len(new_landings) > 0:
        self.landing_pos[new_landings] = self.robot.data.root_link_pos_w[
          new_landings, :2
        ]
      self.has_jumped[ids_active[landed]] = True
      self.max_height[ids_active] = torch.maximum(
        self.max_height[ids_active],
        self.robot.data.root_link_pos_w[ids_active, 2]
        - self._env.scene.env_origins[ids_active, 2],
      )
      self.max_pitch_ang_vel[ids_active] = torch.maximum(
        self.max_pitch_ang_vel[ids_active],
        self.robot.data.root_link_ang_vel_b[ids_active, 1].abs(),
      )
      postflight = ids_active[
        self.was_in_flight[ids_active] & ~self._postflight_push_applied[ids_active]
      ]
      if len(postflight) > 0:
        self._apply_postflight_push(postflight)
    self.last_contact[ids] = contact[ids]

  def _apply_trigger_push(self, env_ids: torch.Tensor) -> None:
    """Apply the source one-shot take-off impulse, when enabled."""
    # The source marks the take-off push as attempted for every environment
    # selected at the trigger step, even when its per-environment 0.8 draw
    # skips the actual impulse.  Record that outer selection before sampling.
    self._push_applied[env_ids] = True
    if not self.cfg.push_towards_goal or len(env_ids) == 0:
      return
    probability = self._current_push_probability()
    candidates = env_ids[torch.rand(len(env_ids), device=self.device) < probability]
    if len(candidates) == 0:
      return
    velocity = self.robot.data.root_link_vel_w[candidates].clone()
    low, high = self.cfg.upward_push_range
    if high > 0.0:
      velocity[:, 2] += torch.empty(len(candidates), device=self.device).uniform_(
        low, high
      )
    self.robot.write_root_link_velocity_to_sim(velocity, env_ids=candidates)

  def _apply_postflight_push(self, env_ids: torch.Tensor) -> None:
    """Apply the source's one-shot pitch impulse after take-off.

    BackFlip applies its upward impulse when the trigger fires, then applies a
    separate angular-y impulse once the feet first leave the ground.  Spring-
    Jump leaves ``pitch_push_range`` at zero, so this path is a no-op there.
    """
    self._postflight_push_applied[env_ids] = True
    if not self.cfg.push_towards_goal or len(env_ids) == 0:
      return
    low, high = self.cfg.pitch_push_range
    if high <= 0.0:
      return
    probability = self._current_push_probability()
    candidates = env_ids[torch.rand(len(env_ids), device=self.device) < probability]
    if len(candidates) == 0:
      return
    velocity = self.robot.data.root_link_vel_w[candidates].clone()
    velocity[:, 4] += torch.empty(len(candidates), device=self.device).uniform_(
      low, high
    )
    self.robot.write_root_link_velocity_to_sim(velocity, env_ids=candidates)

  def _current_push_probability(self) -> float:
    """Return the source flip-task impulse curriculum probability."""
    interval = self.cfg.push_probability_decay_interval_steps
    if interval <= 0:
      return self.cfg.push_probability
    stages = self._env.common_step_counter // interval
    return max(
      self.cfg.push_probability - stages * self.cfg.push_probability_decay,
      0.0,
    )
