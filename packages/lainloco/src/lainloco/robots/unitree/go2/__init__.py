"""Unitree Go2 specifications and experiment catalog."""

from .contract import (
  GO2_POLICY_CONTRACT_VERSION,
  GO2_POLICY_INPUTS,
  Go2PolicyInputSpec,
  make_go2_policy_contract,
)
from .robot import GO2, GO2_JOINT_ORDER
from .tasks import GO2_TASKS
from .training import GO2_TRAINING_PROFILES

__all__ = [
  "GO2",
  "GO2_JOINT_ORDER",
  "GO2_POLICY_CONTRACT_VERSION",
  "GO2_POLICY_INPUTS",
  "GO2_TASKS",
  "GO2_TRAINING_PROFILES",
  "Go2PolicyInputSpec",
  "make_go2_policy_contract",
]
