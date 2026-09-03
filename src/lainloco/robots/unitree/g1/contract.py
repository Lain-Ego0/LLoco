"""Versioned G1 velocity policy contracts."""

from lainloco.core import ObservationField, PolicyContract

from .robot import G1

G1_POLICY_CONTRACT_VERSION = "1"


def make_g1_policy_contract(task_id: str, actor_dim: int) -> PolicyContract:
  return PolicyContract(
    contract_version=G1_POLICY_CONTRACT_VERSION,
    robot_id=G1.robot_id,
    task_id=task_id,
    joint_order=G1.joint_order,
    action_dim=G1.action_dim,
    action_scale=G1.action_scale,
    observation_fields=(ObservationField("actor", actor_dim),),
    history_length=1,
    history_order="oldest_to_newest",
    history_reset="zero",
    normalization="profile-defined",
    recurrent_state=None,
    control_dt=G1.control_dt,
  )
