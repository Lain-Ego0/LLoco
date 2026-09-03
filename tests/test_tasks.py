"""Task registration and mjlab compatibility tests."""

from importlib.metadata import version

from mjlab.tasks.registry import list_tasks, load_env_cfg

import lloco.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnvCfg

EXPECTED_ROBOTS = {"A2", "As2", "Go2", "G1", "G1-23Dof", "H1_2", "H2", "R1"}


def test_pinned_mjlab_version() -> None:
  assert version("mjlab") == "1.6.0"


def test_velocity_tasks_are_registered() -> None:
  registered = set(list_tasks())
  expected = {
    f"Unitree-{robot}-{terrain}"
    for robot in EXPECTED_ROBOTS
    for terrain in ("Flat", "Rough")
  }
  assert expected <= registered


def test_registered_configs_are_independent() -> None:
  first = load_env_cfg("Unitree-Go2-Flat")
  second = load_env_cfg("Unitree-Go2-Flat")
  assert isinstance(first, ManagerBasedRlEnvCfg)
  assert first is not second
  first.scene.num_envs = 37
  assert second.scene.num_envs != 37


def test_flat_and_play_overrides() -> None:
  for robot in EXPECTED_ROBOTS:
    train_cfg = load_env_cfg(f"Unitree-{robot}-Flat")
    play_cfg = load_env_cfg(f"Unitree-{robot}-Flat", play=True)
    assert train_cfg.scene.terrain is not None
    assert train_cfg.scene.terrain.terrain_type == "plane"
    assert train_cfg.scene.terrain.terrain_generator is None
    assert train_cfg.observations["actor"].enable_corruption
    assert not play_cfg.observations["actor"].enable_corruption
    assert "push_robot" not in play_cfg.events


def test_tracking_tasks_are_registered() -> None:
  registered = set(list_tasks())
  assert "Unitree-G1-Tracking" in registered
  assert "Unitree-G1-Tracking-No-State-Estimation" in registered
  assert "Unitree-G1-23Dof-Tracking" in registered
  assert "Unitree-G1-23Dof-Tracking-No-State-Estimation" in registered
