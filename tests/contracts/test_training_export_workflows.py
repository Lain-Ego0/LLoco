"""Explicit task/profile training and checkpoint export plan tests."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from rsl_rl.modules import EmpiricalNormalization
from torch import nn

from lainloco.robots.unitree.go2.deploy.policy import go2_policy_contract_metadata
from lainloco.robots.unitree.go2.training.runner import VelocityOnPolicyRunner
from lainloco.workflows import (
  build_playback_plan,
  build_policy_export_plan,
  build_training_plan,
)


def test_training_plan_resolves_task_profile_and_cpu(tmp_path: Path) -> None:
  plan = build_training_plan(
    task_id="go2/backflip",
    profile_id="ppo",
    iterations=3,
    num_envs=2,
    log_root=tmp_path,
    gpu_ids=None,
  )

  assert plan.registry_task_id == "Mjlab-Backflip-Flat-Unitree-Go2"
  assert plan.iterations == 3
  assert plan.num_envs == 2
  assert plan.gpu_ids is None


def test_training_plan_uses_profile_default_iterations() -> None:
  plan = build_training_plan(
    task_id="go2/velocity-flat",
    profile_id="ppo",
    num_envs=1,
  )
  assert plan.iterations > 0


def test_training_plan_rejects_distillation_profile() -> None:
  with pytest.raises(ValueError, match="use lainloco distill"):
    build_training_plan(
      task_id="go2/velocity-rough",
      profile_id="ts-student",
      num_envs=1,
    )


def test_playback_plan_requires_checkpoint_only_for_trained(tmp_path: Path) -> None:
  random_plan = build_playback_plan(
    task_id="go2/trot",
    profile_id="ppo",
    agent="random",
    num_envs=2,
  )
  assert random_plan.registry_task_id == "Mjlab-Trot-Flat-Unitree-Go2"

  with pytest.raises(ValueError, match="requires --checkpoint"):
    build_playback_plan(
      task_id="go2/trot",
      profile_id="ppo",
      agent="trained",
    )

  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"plan-only")
  trained_plan = build_playback_plan(
    task_id="go2/trot",
    profile_id="ppo",
    checkpoint=checkpoint,
  )
  assert trained_plan.checkpoint == checkpoint.resolve()


def test_export_plan_requires_checkpoint_and_new_destination(tmp_path: Path) -> None:
  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"checkpoint-plan-only")
  plan = build_policy_export_plan(
    checkpoint,
    tmp_path / "bundle",
    task_id="go2/velocity-flat",
    profile_id="ppo",
  )
  assert plan.checkpoint == checkpoint.resolve()
  assert plan.registry_task_id == "Mjlab-Velocity-Flat-Unitree-Go2"

  (tmp_path / "bundle").mkdir()
  with pytest.raises(FileExistsError, match="already exists"):
    build_policy_export_plan(
      checkpoint,
      tmp_path / "bundle",
      task_id="go2/velocity-flat",
      profile_id="ppo",
    )


def test_recurrent_metadata_does_not_declare_conditional_input() -> None:
  actor = SimpleNamespace(latent_kind="ts", use_student=True)
  metadata = go2_policy_contract_metadata(actor, recurrent=True)

  assert metadata["go2_policy_mode"] == "ts_student_recurrent"
  assert "go2_conditional_dim" not in metadata


def test_runner_restores_checkpoint_owned_actor_normalizer(tmp_path: Path) -> None:
  source_normalizer = EmpiricalNormalization(45)
  checkpoint = tmp_path / "student.pt"
  torch.save(
    {
      "actor_state_dict": {
        f"obs_normalizer.{key}": value
        for key, value in source_normalizer.state_dict().items()
      }
    },
    checkpoint,
  )

  class Actor(nn.Module):
    _go2_actor_dim = 45

    def __init__(self) -> None:
      super().__init__()
      self.obs_normalizer = nn.Identity()

  actor = Actor()
  runner = cast(Any, object.__new__(VelocityOnPolicyRunner))
  runner.alg = SimpleNamespace(get_policy=lambda: actor)
  runner.device = "cpu"
  loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
  runner._prepare_checkpoint_for_load(loaded, checkpoint)

  assert isinstance(actor.obs_normalizer, EmpiricalNormalization)
  restored = actor.obs_normalizer.state_dict()
  for key, expected in source_normalizer.state_dict().items():
    torch.testing.assert_close(restored[key], expected)
