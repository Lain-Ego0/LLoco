"""Task registration and mjlab compatibility tests."""

from importlib.metadata import version
from typing import Any, cast

import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

ROBOT_TERRAINS = {
  **{
    robot: ("Flat", "Rough")
    for robot in ("A2", "As2", "Go2", "G1", "G1-23Dof", "H1_2", "H2", "R1")
  },
  "OpenDoge": ("Flat", "Rough"),
}
ROBOT_TASK_GROUP = {
  "OpenDoge": "LainLab",
  **{
    robot: "Unitree"
    for robot in ("A2", "As2", "Go2", "G1", "G1-23Dof", "H1_2", "H2", "R1")
  },
}
EXPECTED_ROBOTS = set(ROBOT_TASK_GROUP)


def task_id(robot: str, terrain: str) -> str:
  return f"{ROBOT_TASK_GROUP[robot]}-{robot}-{terrain}"


def test_pinned_mjlab_version() -> None:
  assert version("mjlab") == "1.6.0"


def test_velocity_tasks_are_registered() -> None:
  registered = set(list_tasks())
  expected = {
    task_id(robot, terrain)
    for robot, terrains in ROBOT_TERRAINS.items()
    for terrain in terrains
  }
  assert expected <= registered
  assert "Unitree-OpenDoge-Flat" not in registered


def test_registered_configs_are_independent() -> None:
  first = load_env_cfg("Unitree-Go2-Flat")
  second = load_env_cfg("Unitree-Go2-Flat")
  assert isinstance(first, ManagerBasedRlEnvCfg)
  assert first is not second
  first.scene.num_envs = 37
  assert second.scene.num_envs != 37


def test_flat_and_play_overrides() -> None:
  for robot in EXPECTED_ROBOTS:
    train_cfg = load_env_cfg(task_id(robot, "Flat"))
    play_cfg = load_env_cfg(task_id(robot, "Flat"), play=True)
    assert train_cfg.scene.terrain is not None
    assert train_cfg.scene.terrain.terrain_type == "plane"
    assert train_cfg.scene.terrain.terrain_generator is None
    assert train_cfg.observations["actor"].enable_corruption
    assert not play_cfg.observations["actor"].enable_corruption
    assert "push_robot" not in play_cfg.events


def test_opendoge_rough_overrides() -> None:
  cfg = load_env_cfg("LainLab-OpenDoge-Rough")
  terrain = cfg.scene.terrain
  assert terrain is not None
  generator = terrain.terrain_generator
  assert generator is not None
  assert generator.size == (6.0, 6.0)
  assert generator.border_width == 10.0
  assert generator.num_rows == 10
  assert generator.num_cols == 20
  assert generator.curriculum is True
  assert terrain.max_init_terrain_level == 2
  random_rough = cast(Any, generator.sub_terrains["random_rough"])
  assert random_rough.noise_range == (0.005, 0.025)
  assert cfg.sim.nconmax == 64
  assert cfg.sim.njmax == 600
  assert cfg.sim.contact_sensor_maxmatch == 128
  assert cfg.sim.mujoco.ccd_iterations == 200


def test_tracking_tasks_are_registered() -> None:
  registered = set(list_tasks())
  assert "Unitree-G1-Tracking" in registered
  assert "Unitree-G1-Tracking-No-State-Estimation" in registered
  assert "Unitree-G1-23Dof-Tracking" in registered
  assert "Unitree-G1-23Dof-Tracking-No-State-Estimation" in registered
