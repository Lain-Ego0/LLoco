"""MuJoCo Warp reset/step smoke test for completed Go2 skill tasks."""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import lloco.tasks  # noqa: F401
from lloco.tasks.go2_skills.mdp.observations import source_vertical_contact


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


def test_jump_single_environment_reset_step() -> None:
  cfg = load_env_cfg("Unitree-Go2-Jump-Flat")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    assert observations["actor"].shape == (1, 470)
    assert observations["critic"].shape == (1, 210)
    result = env.step(torch.zeros((1, 12)))
    observations, reward = result[0], result[1]
    assert observations["actor"].shape == (1, 470)
    assert observations["critic"].shape == (1, 210)
    assert torch.isfinite(observations["actor"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_jump_vertical_contact_sign_after_settling() -> None:
  cfg = load_env_cfg("Unitree-Go2-Jump-Flat", play=True)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    for _ in range(12):
      env.step(torch.zeros((1, 12)))
    sensor = env.scene["feet_ground_contact"]
    force = sensor.data.force
    assert force is not None
    assert torch.all(force[0, :, 2] < -5.0)
    assert torch.all(source_vertical_contact(sensor, 5.0))
  finally:
    env.close()
