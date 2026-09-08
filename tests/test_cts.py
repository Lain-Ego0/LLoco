"""Regression tests for the concurrent teacher-student CTS migration."""

from typing import Any, cast

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import lloco.tasks  # noqa: F401
from lloco.tasks.go2_skills.cts import mdp as cts_mdp


def test_cts_source_physics_and_reward_contract() -> None:
  cfg = load_env_cfg("Unitree-Go2-CTS-Rough")
  assert "alive" not in cfg.rewards
  assert cfg.rewards["termination"].weight == 0.0
  assert cfg.rewards["base_height"].weight == -2.0
  assert cfg.rewards["dof_acc"].func is cts_mdp.CtsDofAcceleration
  assert cfg.rewards["action_smoothness"].func is cts_mdp.CtsActionSmoothness
  assert cfg.events["base_com"].func is cts_mdp.randomize_base_com
  assert cfg.events["reset_robot_joints"].params["scale_range"] == (0.9, 1.1)
  assert all(a.armature == 0.00448 for a in cfg.scene.entities["robot"].articulation.actuators)
  assert cfg.scene.entities["robot"].collisions[0].conaffinity == 0
  base_sensor = next(s for s in cfg.scene.sensors if s.name == "base_ground_contact")
  assert base_sensor.primary.mode == "geom"
  assert base_sensor.primary.pattern == "base1_collision"
  barrier = cfg.rewards["low_base_height_barrier"]
  assert barrier.func is cts_mdp.low_base_height_barrier
  assert barrier.weight == -100.0
  assert barrier.params["minimum_height"] == 0.30
  spec = cfg.scene.entities["robot"].spec_fn()
  collision_geoms = {g.name: g for g in spec.geoms if g.name and "collision" in g.name}
  assert len(collision_geoms) == 27
  assert collision_geoms["FL_thigh_collision"].size[0] == 0.055
  assert collision_geoms["FL_calf3_collision"].type == mujoco.mjtGeom.mjGEOM_CAPSULE


def test_cts_observation_contract() -> None:
  cfg = load_env_cfg("Unitree-Go2-CTS-Rough", play=True)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    values = cast(dict[str, Any], observations)
    assert values["actor"].shape == (1, 45)
    assert values["teacher"].shape == (1, 233)
    assert values["critic"].shape == (1, 278)
    assert values["history"].shape == (1, 225)
    assert all(torch.isfinite(value).all() for value in values.values())
  finally:
    env.close()
