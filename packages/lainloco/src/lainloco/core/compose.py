"""Explicit experiment composition entry point."""

from .experiment_spec import ExperimentSpec
from .policy_contract import PolicyContract
from .robot_spec import RobotSpec
from .task_spec import TaskSpec
from .training_spec import TrainingSpec


def compose_experiment(
  *,
  robot: RobotSpec,
  task: TaskSpec,
  training: TrainingSpec,
  contract: PolicyContract,
) -> ExperimentSpec:
  """Compose and validate a launchable experiment from its four domains."""
  return ExperimentSpec(
    robot=robot,
    task=task,
    training=training,
    contract=contract,
  )
