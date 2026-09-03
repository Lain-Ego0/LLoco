"""Compatibility exports for the pre-split core specification module."""

from .experiment_spec import ExperimentSpec
from .policy_contract import ObservationField, PolicyContract, RecurrentStateSpec
from .robot_spec import RobotSpec
from .task_spec import TaskSpec
from .training_spec import TrainingSpec

__all__ = [
  "ExperimentSpec",
  "ObservationField",
  "PolicyContract",
  "RecurrentStateSpec",
  "RobotSpec",
  "TaskSpec",
  "TrainingSpec",
]
