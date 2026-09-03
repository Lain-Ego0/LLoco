"""Go2 Passive/Stand/Policy and safety-fallback tests."""

from __future__ import annotations

import numpy as np
import pytest

from lainloco.robots.unitree.go2 import GO2
from lainloco.robots.unitree.go2.deploy import (
  Go2ControllerFsm,
  Go2ControlState,
  Go2RobotState,
)


def _robot_state(
  *,
  joint_position: np.ndarray | None = None,
  joint_velocity: np.ndarray | None = None,
  observation_age_s: float = 0.0,
  emergency_stop: bool = False,
) -> Go2RobotState:
  return Go2RobotState(
    joint_position=(
      np.zeros(12, dtype=np.float32) if joint_position is None else joint_position
    ),
    joint_velocity=(
      np.zeros(12, dtype=np.float32) if joint_velocity is None else joint_velocity
    ),
    observation_age_s=observation_age_s,
    emergency_stop=emergency_stop,
  )


def _finish_stand(fsm: Go2ControllerFsm, state: Go2RobotState) -> None:
  fsm.request(Go2ControlState.STAND, state)
  for _ in range(50):
    command = fsm.step(state)
    assert command.enabled


def test_go2_fsm_runs_passive_stand_policy_sequence() -> None:
  state = _robot_state()
  fsm = Go2ControllerFsm()
  passive = fsm.step(state)
  assert passive.state == Go2ControlState.PASSIVE
  assert not passive.enabled
  np.testing.assert_array_equal(passive.kp, 0.0)

  _finish_stand(fsm, state)
  fsm.request(Go2ControlState.POLICY, state)
  action = np.linspace(-1.5, 1.5, GO2.action_dim, dtype=np.float32)
  policy = fsm.step(state, policy_action=action)

  expected = np.asarray([value for _, value in GO2.default_pose]) + np.asarray(
    GO2.action_scale
  ) * np.clip(action, -1.0, 1.0)
  assert policy.state == Go2ControlState.POLICY
  assert policy.enabled
  assert policy.joint_order == GO2.joint_order
  np.testing.assert_allclose(policy.position_target, expected)


def test_go2_fsm_rejects_policy_before_stand_completes() -> None:
  state = _robot_state()
  fsm = Go2ControllerFsm()
  with pytest.raises(ValueError, match="only be entered from stand"):
    fsm.request(Go2ControlState.POLICY, state)
  fsm.request(Go2ControlState.STAND, state)
  with pytest.raises(ValueError, match="must complete"):
    fsm.request(Go2ControlState.POLICY, state)


@pytest.mark.parametrize(
  ("state", "action", "reason"),
  (
    (_robot_state(emergency_stop=True), np.zeros(12), "emergency stop"),
    (_robot_state(observation_age_s=0.2), np.zeros(12), "stale"),
    (_robot_state(), np.full(12, np.nan), "non-finite"),
    (_robot_state(), np.full(12, 3.0), "hard safety limit"),
  ),
)
def test_go2_fsm_latches_safety_fallback(
  state: Go2RobotState, action: np.ndarray, reason: str
) -> None:
  fsm = Go2ControllerFsm()
  if not state.emergency_stop and state.observation_age_s <= 0.1:
    healthy = _robot_state()
    _finish_stand(fsm, healthy)
    fsm.request(Go2ControlState.POLICY, healthy)
  command = fsm.step(state, policy_action=action)

  assert command.state == Go2ControlState.SAFETY_FALLBACK
  assert not command.enabled
  assert command.fault_reason is not None and reason in command.fault_reason
  with pytest.raises(RuntimeError, match="latched"):
    fsm.request(Go2ControlState.PASSIVE, _robot_state())


def test_go2_fsm_fallback_requires_explicit_healthy_reset() -> None:
  fsm = Go2ControllerFsm()
  command = fsm.step(_robot_state(), dt=0.01)
  assert command.state == Go2ControlState.SAFETY_FALLBACK
  with pytest.raises(RuntimeError, match="stale"):
    fsm.reset_fallback(_robot_state(observation_age_s=0.2))

  fsm.reset_fallback(_robot_state())
  assert fsm.state == Go2ControlState.PASSIVE
  assert fsm.fault_reason is None
