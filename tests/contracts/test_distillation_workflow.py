"""Contract tests for the explicit distillation workflow."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lainloco.workflows import build_distillation_plan, launch_distillation


def _teacher_checkpoint(tmp_path: Path) -> Path:
  checkpoint = tmp_path / "teacher.pt"
  checkpoint.touch()
  return checkpoint


def test_distillation_plan_resolves_task_and_profile(tmp_path: Path) -> None:
  checkpoint = _teacher_checkpoint(tmp_path)
  plan = build_distillation_plan(
    checkpoint,
    iterations=3,
    num_envs=7,
    gpu_ids=None,
  )
  assert plan.task_id == "go2/velocity-rough"
  assert plan.profile_id == "ts-student"
  assert plan.registry_task_id == "Mjlab-TS-Student-Rough-Unitree-Go2"
  assert plan.teacher_checkpoint == checkpoint.resolve()
  assert plan.iterations == 3
  assert plan.num_envs == 7
  assert plan.gpu_ids is None


def test_distillation_plan_rejects_invalid_inputs(tmp_path: Path) -> None:
  checkpoint = _teacher_checkpoint(tmp_path)
  with pytest.raises(FileNotFoundError):
    build_distillation_plan(tmp_path / "missing.pt")
  with pytest.raises(ValueError, match="iterations"):
    build_distillation_plan(checkpoint, iterations=0)
  with pytest.raises(ValueError, match="not a distillation experiment"):
    build_distillation_plan(checkpoint, profile_id="ppo")


def test_launch_distillation_scopes_teacher_checkpoint(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  from lainloco.workflows import distill

  checkpoint = _teacher_checkpoint(tmp_path)
  plan = build_distillation_plan(
    checkpoint,
    iterations=2,
    num_envs=4,
    log_root=tmp_path / "logs",
    gpu_ids=None,
  )
  observed: dict[str, object] = {}

  def fake_launch(task_id, cfg) -> None:
    observed["task_id"] = task_id
    observed["teacher"] = os.environ["GO2_TS_TEACHER_CHECKPOINT"]
    observed["iterations"] = cfg.agent.max_iterations
    observed["num_envs"] = cfg.env.scene.num_envs
    observed["logger"] = cfg.agent.logger
    observed["upload_model"] = cfg.agent.upload_model

  monkeypatch.setattr(distill, "launch_training", fake_launch)
  monkeypatch.setenv("GO2_TS_TEACHER_CHECKPOINT", "previous.pt")
  launch_distillation(plan)

  assert observed == {
    "task_id": "Mjlab-TS-Student-Rough-Unitree-Go2",
    "teacher": str(checkpoint.resolve()),
    "iterations": 2,
    "num_envs": 4,
    "logger": "tensorboard",
    "upload_model": False,
  }
  assert os.environ["GO2_TS_TEACHER_CHECKPOINT"] == "previous.pt"
