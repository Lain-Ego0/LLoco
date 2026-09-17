"""Reset/step smoke test for the migrated OpenDoge flat velocity task."""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import lloco.tasks  # noqa: F401


def test_opendoge_flat_reset_step() -> None:
  cfg = load_env_cfg("LainLab-OpenDoge-Flat")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    assert observations["actor"].shape == (1, 48)
    assert observations["critic"].shape == (1, 72)
    observations, reward, *_ = env.step(torch.zeros((1, 12)))
    assert torch.isfinite(observations["actor"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
  finally:
    env.close()
