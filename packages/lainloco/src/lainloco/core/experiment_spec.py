"""Launchable robot, task, training and policy composition."""

from dataclasses import dataclass

from .policy_contract import PolicyContract
from .robot_spec import RobotSpec
from .task_spec import TaskSpec
from .training_spec import TrainingSpec


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
  """A launchable composition of robot, task, training, and policy contract."""

  robot: RobotSpec
  task: TaskSpec
  training: TrainingSpec
  contract: PolicyContract

  def __post_init__(self) -> None:
    if self.robot.robot_id != self.task.robot_id:
      raise ValueError("task robot_id does not match RobotSpec")
    if self.contract.robot_id != self.robot.robot_id:
      raise ValueError("policy contract robot_id does not match RobotSpec")
    if self.contract.task_id != self.task.task_id:
      raise ValueError("policy contract task_id does not match TaskSpec")
    if self.contract.joint_order != self.robot.joint_order:
      raise ValueError("policy contract joint order does not match RobotSpec")
