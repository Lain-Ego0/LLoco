"""Versioned Go2 policy contracts shared by training and deployment."""

from dataclasses import dataclass

from lainloco.core import (
  ObservationField,
  PolicyContract,
  RecurrentStateSpec,
)

from .robot import GO2

GO2_POLICY_CONTRACT_VERSION = "1"


@dataclass(frozen=True)
class Go2PolicyInputSpec:
  """Authoritative dimensions for source-compatible Go2 policy interfaces."""

  actor_dim: int = 45
  history_length: int = 5
  terrain_dim: int = 187
  cts_privileged_dim: int = 233
  ts_privileged_dim: int = 74
  action_dim: int = 12
  ts_student_lstm_layers: int = 3
  ts_student_hidden_dim: int = 256

  @property
  def history_dim(self) -> int:
    return self.actor_dim * self.history_length


GO2_POLICY_INPUTS = Go2PolicyInputSpec()


def make_go2_policy_contract(
  task_id: str,
  actor_dim: int,
  *,
  history_length: int = 1,
  recurrent: bool = False,
  conditional_fields: tuple[ObservationField, ...] = (),
) -> PolicyContract:
  """Build a policy contract from the immutable Go2 robot facts."""
  return PolicyContract(
    contract_version=GO2_POLICY_CONTRACT_VERSION,
    robot_id=GO2.robot_id,
    task_id=task_id,
    joint_order=GO2.joint_order,
    action_dim=GO2.action_dim,
    action_scale=GO2.action_scale,
    observation_fields=(ObservationField("actor", actor_dim),),
    history_length=history_length,
    history_order="oldest_to_newest",
    history_reset="zero",
    normalization="profile-defined",
    recurrent_state=(
      RecurrentStateSpec(
        GO2_POLICY_INPUTS.ts_student_lstm_layers,
        GO2_POLICY_INPUTS.ts_student_hidden_dim,
      )
      if recurrent
      else None
    ),
    control_dt=GO2.control_dt,
    conditional_fields=conditional_fields,
  )
