"""Explicit teacher-to-student distillation workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from mjlab.scripts.train import TrainConfig, launch_training

from lainloco.robots.unitree.go2.experiments import resolve_experiment
from lainloco.robots.unitree.go2.training.runner import VelocityDistillationRunner


@dataclass(frozen=True, slots=True)
class DistillationPlan:
  """Validated task/profile composition for a student training run."""

  task_id: str
  profile_id: str
  registry_task_id: str
  teacher_checkpoint: Path
  iterations: int
  num_envs: int
  log_root: Path
  gpu_ids: list[int] | None


def build_distillation_plan(
  teacher_checkpoint: str | Path,
  *,
  task_id: str = "go2/velocity-rough",
  profile_id: str = "ts-student",
  iterations: int = 20_000,
  num_envs: int = 1024,
  log_root: str | Path = "logs/rsl_rl",
  gpu_ids: list[int] | None = None,
) -> DistillationPlan:
  """Validate inputs and resolve the explicit distillation experiment."""
  checkpoint = Path(teacher_checkpoint).expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint}")
  if iterations <= 0:
    raise ValueError("iterations must be positive")
  if num_envs <= 0:
    raise ValueError("num_envs must be positive")
  binding = resolve_experiment(task_id, profile_id)
  if not issubclass(binding.runner_cls, VelocityDistillationRunner):
    raise ValueError(f"{task_id}::{profile_id} is not a distillation experiment")
  return DistillationPlan(
    task_id=task_id,
    profile_id=profile_id,
    registry_task_id=binding.legacy_task_id,
    teacher_checkpoint=checkpoint,
    iterations=iterations,
    num_envs=num_envs,
    log_root=Path(log_root).expanduser().resolve(),
    gpu_ids=gpu_ids,
  )


def launch_distillation(plan: DistillationPlan) -> None:
  """Run a validated plan while scoping the legacy checkpoint bridge."""
  cfg = TrainConfig.from_task(plan.registry_task_id)
  cfg.env.scene.num_envs = plan.num_envs
  cfg.agent.max_iterations = plan.iterations
  cfg.agent.logger = "tensorboard"
  cfg.agent.upload_model = False
  cfg = replace(cfg, log_root=str(plan.log_root), gpu_ids=plan.gpu_ids)

  previous = os.environ.get("GO2_TS_TEACHER_CHECKPOINT")
  os.environ["GO2_TS_TEACHER_CHECKPOINT"] = str(plan.teacher_checkpoint)
  try:
    launch_training(plan.registry_task_id, cfg)
  finally:
    if previous is None:
      os.environ.pop("GO2_TS_TEACHER_CHECKPOINT", None)
    else:
      os.environ["GO2_TS_TEACHER_CHECKPOINT"] = previous
