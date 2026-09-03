"""Hardware-independent Go2 control FSM with an explicit safety fallback.

This module deliberately stops at joint command generation. SDK transport,
operator authorization, hardware joint mapping, and physical safety validation
remain separate acceptance work before commands may reach a real robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..robot import GO2


class Go2ControlState(str, Enum):
  """Operator-visible states in the deployment controller."""

  PASSIVE = "passive"
  STAND = "stand"
  POLICY = "policy"
  SAFETY_FALLBACK = "safety_fallback"


@dataclass(frozen=True, slots=True)
class Go2RobotState:
  """Joint state and freshness facts supplied by a simulator or transport."""

  joint_position: np.ndarray
  joint_velocity: np.ndarray
  observation_age_s: float = 0.0
  emergency_stop: bool = False


@dataclass(frozen=True, slots=True)
class Go2SafetyLimits:
  """Conservative software limits; not a substitute for hardware limits."""

  stand_duration_s: float = 1.0
  max_observation_age_s: float = 0.1
  max_abs_joint_position: float = 4.0
  max_abs_joint_velocity: float = 30.0
  policy_action_clip: float = 1.0
  policy_action_trip: float = 2.0
  stand_kp: float = 40.0
  stand_kd: float = 1.0
  policy_kp: float = 40.0
  policy_kd: float = 1.0

  def __post_init__(self) -> None:
    positive = (
      self.stand_duration_s,
      self.max_observation_age_s,
      self.max_abs_joint_position,
      self.max_abs_joint_velocity,
      self.policy_action_clip,
      self.policy_action_trip,
      self.stand_kp,
      self.stand_kd,
      self.policy_kp,
      self.policy_kd,
    )
    if not all(math.isfinite(value) and value > 0 for value in positive):
      raise ValueError("Go2 safety limits must be finite and positive")
    if self.policy_action_trip < self.policy_action_clip:
      raise ValueError("policy_action_trip cannot be below policy_action_clip")


@dataclass(frozen=True, slots=True)
class Go2ControlCommand:
  """One ordered joint command, with actuation explicitly enabled or disabled."""

  state: Go2ControlState
  joint_order: tuple[str, ...]
  position_target: np.ndarray
  kp: np.ndarray
  kd: np.ndarray
  feedforward_torque: np.ndarray
  enabled: bool
  fault_reason: str | None = None


class Go2ControllerFsm:
  """Passive → Stand → Policy controller with latched safe fallback."""

  def __init__(self, limits: Go2SafetyLimits | None = None) -> None:
    self.limits = limits or Go2SafetyLimits()
    self.state = Go2ControlState.PASSIVE
    self.fault_reason: str | None = None
    self._stand_start: np.ndarray | None = None
    self._stand_elapsed_s = 0.0
    self._default_position = np.asarray(
      [value for _, value in GO2.default_pose], dtype=np.float32
    )
    self._action_scale = np.asarray(GO2.action_scale, dtype=np.float32)

  def request(self, state: Go2ControlState, robot_state: Go2RobotState) -> None:
    """Request a legal operator transition; fallback remains latched."""
    if self.state == Go2ControlState.SAFETY_FALLBACK:
      raise RuntimeError("Safety fallback is latched; reset it before transitioning")
    if state == Go2ControlState.SAFETY_FALLBACK:
      self._trip("operator requested safety fallback")
      return
    if state == Go2ControlState.PASSIVE:
      self.state = state
      self._stand_start = None
      self._stand_elapsed_s = 0.0
      return
    if state == Go2ControlState.STAND:
      if self.state not in {Go2ControlState.PASSIVE, Go2ControlState.STAND}:
        raise ValueError(f"Cannot enter stand from {self.state.value}")
      position, _ = self._validated_robot_state(robot_state)
      self.state = state
      self._stand_start = position.copy()
      self._stand_elapsed_s = 0.0
      return
    if state == Go2ControlState.POLICY:
      if self.state != Go2ControlState.STAND:
        raise ValueError("Policy can only be entered from stand")
      if self._stand_elapsed_s < self.limits.stand_duration_s:
        raise ValueError("Stand interpolation must complete before policy")
      self.state = state
      return
    raise AssertionError(f"Unhandled Go2 control state: {state}")

  def reset_fallback(self, robot_state: Go2RobotState) -> None:
    """Clear a latched fallback only after the supplied state passes validation."""
    if self.state != Go2ControlState.SAFETY_FALLBACK:
      raise RuntimeError("Controller is not in safety fallback")
    reason = self._safety_reason(robot_state)
    if reason is not None:
      raise RuntimeError(f"Cannot reset fallback: {reason}")
    self.state = Go2ControlState.PASSIVE
    self.fault_reason = None
    self._stand_start = None
    self._stand_elapsed_s = 0.0

  def _validated_robot_state(
    self, robot_state: Go2RobotState
  ) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(robot_state.joint_position, dtype=np.float32)
    velocity = np.asarray(robot_state.joint_velocity, dtype=np.float32)
    expected = (GO2.action_dim,)
    if position.shape != expected or velocity.shape != expected:
      raise ValueError(
        f"Go2 joint position/velocity must have shape {expected}, got "
        f"{position.shape} and {velocity.shape}"
      )
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
      raise ValueError("Go2 joint state contains non-finite values")
    if robot_state.observation_age_s < 0 or not math.isfinite(
      robot_state.observation_age_s
    ):
      raise ValueError("observation_age_s must be finite and non-negative")
    return position, velocity

  def _safety_reason(self, robot_state: Go2RobotState) -> str | None:
    try:
      position, velocity = self._validated_robot_state(robot_state)
    except ValueError as exc:
      return str(exc)
    if robot_state.emergency_stop:
      return "emergency stop is active"
    if robot_state.observation_age_s > self.limits.max_observation_age_s:
      return (
        f"observation is stale ({robot_state.observation_age_s:g}s > "
        f"{self.limits.max_observation_age_s:g}s)"
      )
    if np.any(np.abs(position) > self.limits.max_abs_joint_position):
      return "joint position exceeded the software safety limit"
    if np.any(np.abs(velocity) > self.limits.max_abs_joint_velocity):
      return "joint velocity exceeded the software safety limit"
    return None

  def _trip(self, reason: str) -> None:
    self.state = Go2ControlState.SAFETY_FALLBACK
    self.fault_reason = reason

  def _passive_command(self) -> Go2ControlCommand:
    zeros = np.zeros(GO2.action_dim, dtype=np.float32)
    return Go2ControlCommand(
      state=self.state,
      joint_order=GO2.joint_order,
      position_target=self._default_position.copy(),
      kp=zeros.copy(),
      kd=zeros.copy(),
      feedforward_torque=zeros.copy(),
      enabled=False,
      fault_reason=self.fault_reason,
    )

  def step(
    self,
    robot_state: Go2RobotState,
    *,
    dt: float = GO2.control_dt,
    policy_action: np.ndarray | None = None,
  ) -> Go2ControlCommand:
    """Generate one command or atomically fall back to disabled actuation."""
    if not math.isfinite(dt) or dt <= 0:
      raise ValueError("dt must be finite and positive")
    if not math.isclose(dt, GO2.control_dt, rel_tol=0.0, abs_tol=1e-12):
      self._trip(f"control dt mismatch ({dt!r} != {GO2.control_dt!r})")
      return self._passive_command()
    reason = self._safety_reason(robot_state)
    if reason is not None:
      self._trip(reason)
      return self._passive_command()
    if self.state in {Go2ControlState.PASSIVE, Go2ControlState.SAFETY_FALLBACK}:
      return self._passive_command()
    if self.state == Go2ControlState.STAND:
      if self._stand_start is None:
        self._trip("stand state has no captured start position")
        return self._passive_command()
      self._stand_elapsed_s = min(
        self._stand_elapsed_s + dt, self.limits.stand_duration_s
      )
      phase = self._stand_elapsed_s / self.limits.stand_duration_s
      blend = phase * phase * (3.0 - 2.0 * phase)
      target = self._stand_start + blend * (self._default_position - self._stand_start)
      return self._enabled_command(target, self.limits.stand_kp, self.limits.stand_kd)
    if policy_action is None:
      self._trip("policy state requires an action")
      return self._passive_command()
    action = np.asarray(policy_action, dtype=np.float32)
    if action.shape != (GO2.action_dim,):
      self._trip(
        f"policy action must have shape {(GO2.action_dim,)}, got {action.shape}"
      )
      return self._passive_command()
    if not np.all(np.isfinite(action)):
      self._trip("policy action contains non-finite values")
      return self._passive_command()
    if np.any(np.abs(action) > self.limits.policy_action_trip):
      self._trip("policy action exceeded the hard safety limit")
      return self._passive_command()
    clipped = np.clip(
      action, -self.limits.policy_action_clip, self.limits.policy_action_clip
    )
    target = self._default_position + self._action_scale * clipped
    return self._enabled_command(target, self.limits.policy_kp, self.limits.policy_kd)

  def _enabled_command(
    self, target: np.ndarray, kp: float, kd: float
  ) -> Go2ControlCommand:
    return Go2ControlCommand(
      state=self.state,
      joint_order=GO2.joint_order,
      position_target=np.asarray(target, dtype=np.float32),
      kp=np.full(GO2.action_dim, kp, dtype=np.float32),
      kd=np.full(GO2.action_dim, kd, dtype=np.float32),
      feedforward_torque=np.zeros(GO2.action_dim, dtype=np.float32),
      enabled=True,
      fault_reason=None,
    )
