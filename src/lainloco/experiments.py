"""Repository-wide robot and experiment discovery."""

from __future__ import annotations

from functools import cache

from lainloco.core import Catalog, RobotSpec, TaskSpec, TrainingSpec
from lainloco.integrations.mjlab import MjlabExperimentBinding


@cache
def robot_catalog() -> Catalog[RobotSpec]:
  from lainloco.robots.unitree.g1 import G1
  from lainloco.robots.unitree.go2 import GO2

  return Catalog((G1, GO2), id_of=lambda robot: robot.robot_id)


@cache
def experiment_catalog() -> Catalog[MjlabExperimentBinding]:
  from lainloco.robots.unitree.g1.experiments import G1_EXPERIMENTS
  from lainloco.robots.unitree.go2.experiments import GO2_EXPERIMENTS

  return Catalog(
    (*G1_EXPERIMENTS.values(), *GO2_EXPERIMENTS.values()),
    id_of=lambda binding: binding.binding_id,
  )


def resolve_robot(robot_id: str) -> RobotSpec:
  aliases = {
    "g1": "unitree/g1",
    "go2": "unitree/go2",
    "unitree/g1": "unitree/g1",
    "unitree/go2": "unitree/go2",
  }
  try:
    canonical_id = aliases[robot_id.lower()]
  except KeyError as exc:
    choices = ", ".join(robot_catalog().ids())
    raise KeyError(f"Unknown robot {robot_id!r}; choose one of: {choices}") from exc
  return robot_catalog().get(canonical_id)


def task_catalog(robot_id: str) -> Catalog[TaskSpec]:
  robot = resolve_robot(robot_id)
  if robot.robot_id == "unitree/g1":
    from lainloco.robots.unitree.g1.tasks import G1_TASKS

    return G1_TASKS
  from lainloco.robots.unitree.go2.tasks import GO2_TASKS

  return GO2_TASKS


def training_profile_catalog(robot_id: str) -> Catalog[TrainingSpec]:
  robot = resolve_robot(robot_id)
  if robot.robot_id == "unitree/g1":
    from lainloco.robots.unitree.g1.training import G1_TRAINING_PROFILES

    return G1_TRAINING_PROFILES
  from lainloco.robots.unitree.go2.training import GO2_TRAINING_PROFILES

  return GO2_TRAINING_PROFILES


def resolve_experiment(task_id: str, profile_id: str) -> MjlabExperimentBinding:
  return experiment_catalog().get(f"{task_id}::{profile_id}")
