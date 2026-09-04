"""MuJoCo Warp reset/step smoke test for completed Go2 skill tasks."""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import lloco.tasks  # noqa: F401


def test_trot_single_environment_reset_step() -> None:
  cfg = load_env_cfg("Unitree-Go2-Trot-Flat")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    assert observations["actor"].shape == (1, 470)
    assert observations["critic"].shape == (1, 204)
    result = env.step(torch.zeros((1, 12)))
    observations, reward = result[0], result[1]
    assert observations["actor"].shape == (1, 470)
    assert observations["critic"].shape == (1, 204)
    assert torch.isfinite(observations["actor"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
  finally:
    env.close()
