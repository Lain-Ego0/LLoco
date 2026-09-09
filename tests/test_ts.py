"""Contract and CPU smoke tests for the Gym Go2 TS teacher migration."""

from typing import Any, cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import lloco.tasks  # noqa: F401
from lloco.tasks.go2_skills.ts.rl import TsTeacherPolicy


def test_ts_source_contract() -> None:
  cfg = load_env_cfg("Unitree-Go2-TS-Teacher-Rough")
  assert cfg.scene.num_envs == 4096
  assert cfg.decimation == 4
  assert cfg.rewards["tracking_lin_vel"].weight == 1.0
  assert cfg.rewards["base_height"].weight == -1.0
  assert set(cfg.rewards) == {
    "tracking_lin_vel",
    "tracking_ang_vel",
    "lin_vel_z",
    "ang_vel_xy",
    "orientation",
    "base_height",
    "torques",
    "dof_acc",
    "collision",
    "action_rate",
    "stumble",
  }
  runner = cast(Any, load_rl_cfg("Unitree-Go2-TS-Teacher-Rough"))
  assert runner.algorithm.class_name.endswith(":TsPPO")


def test_ts_observation_and_policy_smoke() -> None:
  env = ManagerBasedRlEnv(
    load_env_cfg("Unitree-Go2-TS-Teacher-Rough", play=True), device="cpu"
  )
  try:
    observations, _ = env.reset()
    values = cast(dict[str, torch.Tensor], observations)
    assert {k: tuple(v.shape) for k, v in values.items()} == {
      "actor": (1, 45),
      "terrain": (1, 187),
      "domain": (1, 74),
      "critic": (1, 309),
    }
    assert all(torch.isfinite(value).all() for value in values.values())
    policy = TsTeacherPolicy()
    action = policy.inference(values)
    assert action.shape == (1, 12)
    assert torch.isfinite(action).all()
  finally:
    env.close()
