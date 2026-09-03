"""First-class G1 catalog and workflow contracts."""

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from lainloco.experiments import (
  experiment_catalog,
  resolve_experiment,
  robot_catalog,
  task_catalog,
  training_profile_catalog,
)
from lainloco.robots.unitree.g1 import G1, G1_TASKS, G1_TRAINING_PROFILES
from lainloco.robots.unitree.g1.experiments import G1_EXPERIMENTS
from lainloco.workflows import build_playback_plan, build_training_plan


def test_g1_domain_has_authoritative_29_dof_contract() -> None:
  assert G1.robot_id == "unitree/g1"
  assert G1.action_dim == 29
  assert len(G1.action_scale) == 29
  assert tuple(name for name, _ in G1.default_pose) == G1.joint_order
  assert G1.base_body == "pelvis"
  assert G1.foot_sites == ("left_foot", "right_foot")
  assert G1.hardware_joint_mapping == ()


def test_repository_catalogs_include_g1_and_go2() -> None:
  assert robot_catalog().ids() == ("unitree/g1", "unitree/go2")
  assert task_catalog("g1") is G1_TASKS
  assert training_profile_catalog("unitree/g1") is G1_TRAINING_PROFILES
  assert len(G1_TASKS) == 2
  assert len(G1_TRAINING_PROFILES) == 1
  assert len(G1_EXPERIMENTS) == 2
  assert len(experiment_catalog()) == 18


def test_global_resolver_selects_g1_experiment() -> None:
  binding = resolve_experiment("g1/velocity-flat", "ppo")
  assert binding is G1_EXPERIMENTS.get("g1/velocity-flat::ppo")
  assert binding.experiment.robot is G1
  assert binding.registry_task_id == "Mjlab-Velocity-Flat-Unitree-G1"
  assert binding.experiment.contract.observation_dim == 99
  assert binding.experiment.contract.action_dim == 29
  assert not binding.distillation


def test_g1_canonical_ids_match_mjlab_baselines() -> None:
  aliases = {
    "LainLoco-G1-Velocity-Flat-v0": "Mjlab-Velocity-Flat-Unitree-G1",
    "LainLoco-G1-Velocity-Rough-v0": "Mjlab-Velocity-Rough-Unitree-G1",
  }
  for canonical_id, registry_id in aliases.items():
    assert load_env_cfg(canonical_id) == load_env_cfg(registry_id)
    assert load_env_cfg(canonical_id, play=True) == load_env_cfg(registry_id, play=True)
    assert load_rl_cfg(canonical_id) == load_rl_cfg(registry_id)
    assert load_runner_cls(canonical_id) is load_runner_cls(registry_id)


def test_g1_factories_do_not_leak_mutations() -> None:
  binding = resolve_experiment("g1/velocity-flat", "ppo")
  first = binding.env_factory()
  second = binding.env_factory()
  first.scene.num_envs = 123
  assert second.scene.num_envs != 123


def test_g1_training_and_playback_plans_use_global_resolver() -> None:
  training = build_training_plan(
    task_id="g1/velocity-flat",
    profile_id="ppo",
    iterations=3,
    num_envs=2,
    gpu_ids=None,
  )
  playback = build_playback_plan(
    task_id="g1/velocity-rough",
    profile_id="ppo",
    agent="random",
  )

  assert training.registry_task_id == "Mjlab-Velocity-Flat-Unitree-G1"
  assert training.iterations == 3
  assert playback.registry_task_id == "Mjlab-Velocity-Rough-Unitree-G1"
  assert playback.checkpoint is None
