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


def test_rear_stand_single_environment_reset_step() -> None:
  cfg = load_env_cfg("Unitree-Go2-Rear-Stand-Flat")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    assert observations["actor"].shape == (1, 45)
    assert observations["critic"].shape == (1, 86)
    result = env.step(torch.zeros((1, 12)))
    observations, reward = result[0], result[1]
    assert observations["actor"].shape == (1, 45)
    assert observations["critic"].shape == (1, 86)
    assert torch.isfinite(observations["actor"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_rear_stand_privileged_layout_and_randomization_ranges() -> None:
  cfg = load_env_cfg("Unitree-Go2-Rear-Stand-Flat")
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    actor = observations["actor"]
    critic = observations["critic"]
    assert torch.equal(critic[:, 3:48], actor)
    domain = critic[:, 48:82]
    assert torch.all((0.2 <= domain[:, 0]) & (domain[:, 0] <= 1.2))
    assert torch.all((-1.0 <= domain[:, 1]) & (domain[:, 1] <= 2.0))
    assert torch.all((-0.05 <= domain[:, 2:5]) & (domain[:, 2:5] <= 0.05))
    assert torch.all((0.9 <= domain[:, 5:17]) & (domain[:, 5:17] <= 1.1))
    assert torch.all((0.9 <= domain[:, 17:29]) & (domain[:, 17:29] <= 1.1))
    assert torch.all((0.003 <= domain[:, 29]) & (domain[:, 29] <= 0.08))
    assert torch.all((0.01 <= domain[:, 30]) & (domain[:, 30] <= 0.1))
    assert torch.all((0.0 <= domain[:, 31]) & (domain[:, 31] <= 0.1))
    assert torch.all((0.0 <= domain[:, 32]) & (domain[:, 32] <= 0.3))
    assert torch.equal(domain[:, 32], domain[:, 33])
    assert torch.all((critic[:, 82:86] == 0.0) | (critic[:, 82:86] == 1.0))
  finally:
    env.close()
