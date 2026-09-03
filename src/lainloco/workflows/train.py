"""Explicit task/profile training workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from mjlab.scripts.train import TrainConfig, launch_training

from lainloco.robots.unitree.go2.experiments import resolve_experiment
from lainloco.robots.unitree.go2.training.runner import VelocityDistillationRunner


@dataclass(frozen=True, slots=True)
class TrainingPlan:
  """Validated composition for a normal on-policy training run."""

  task_id: str
  profile_id: str
  registry_task_id: str
  iterations: int
  num_envs: int
  log_root: Path
  gpu_ids: list[int] | None


def build_training_plan(
  *,
  task_id: str,
  profile_id: str,
  iterations: int | None = None,
  num_envs: int = 4096,
  log_root: str | Path = "runs/rsl_rl",
  gpu_ids: list[int] | None = None,
) -> TrainingPlan:
  """Resolve a non-distillation experiment and validate launch dimensions."""
  if num_envs <= 0:
    raise ValueError("num_envs must be positive")
  binding = resolve_experiment(task_id, profile_id)
  if issubclass(binding.runner_cls, VelocityDistillationRunner):
    raise ValueError(
      f"{task_id}::{profile_id} is a distillation experiment; use lainloco distill"
    )
  default_iterations = binding.rl_factory().max_iterations
  selected_iterations = default_iterations if iterations is None else iterations
  if selected_iterations <= 0:
    raise ValueError("iterations must be positive")
  return TrainingPlan(
    task_id=task_id,
    profile_id=profile_id,
    registry_task_id=binding.legacy_task_id,
    iterations=selected_iterations,
    num_envs=num_envs,
    log_root=Path(log_root).expanduser().resolve(),
    gpu_ids=gpu_ids,
  )


def launch_experiment_training(plan: TrainingPlan) -> None:
  """Launch a validated experiment through mjlab's maintained trainer."""
  cfg = TrainConfig.from_task(plan.registry_task_id)
  cfg.env.scene.num_envs = plan.num_envs
  cfg.agent.max_iterations = plan.iterations
  cfg = replace(cfg, log_root=str(plan.log_root), gpu_ids=plan.gpu_ids)
  launch_training(plan.registry_task_id, cfg)
