"""FireDog 2.2 R2 CPU load/reset/action/observation regression."""

import io
from contextlib import redirect_stderr, redirect_stdout

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.firedog2_2.firedog2_2_env_cfg import firedog2_2_velocity_env_cfg


def test_firedog2_2_mjlab_cpu_loop() -> None:
  with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    env = ManagerBasedRlEnv(firedog2_2_velocity_env_cfg(play=True), device="cpu")
    try:
      observations, _ = env.reset()
      assert env.action_manager.total_action_dim == 16
      assert observations["actor"].shape == (1, 32)
      action = torch.zeros((1, 16), device="cpu")
      action[0, 0] = 0.05
      next_observations, _, _, _, _ = env.step(action)
      assert next_observations["actor"].shape == (1, 32)
      assert torch.isfinite(next_observations["actor"]).all()
    finally:
      env.close()
