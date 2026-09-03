"""A1/A2 contract tests for the LainLoco extension boundary."""

import ast
import inspect
import subprocess
import sys
from importlib import import_module
from importlib.metadata import entry_points, requires
from pathlib import Path

import pytest
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from lainloco.core import Catalog
from lainloco.robots.unitree.go2 import (
  GO2,
  GO2_TASKS,
  GO2_TRAINING_PROFILES,
)
from lainloco.robots.unitree.go2.experiments import GO2_EXPERIMENTS
from lainloco.robots.unitree.go2.tasks.locomotion.velocity import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
)

LEGACY_GO2_IDS = {
  "Mjlab-Velocity-Rough-Unitree-Go2",
  "Mjlab-Velocity-Flat-Unitree-Go2",
  "Mjlab-Trot-Flat-Unitree-Go2",
  "Mjlab-Jump-Flat-Unitree-Go2",
  "Mjlab-Spring-Jump-Flat-Unitree-Go2",
  "Mjlab-Backflip-Flat-Unitree-Go2",
  "Mjlab-Handstand-Flat-Unitree-Go2",
  "Mjlab-Leggedstand-Flat-Unitree-Go2",
  "Mjlab-DreamWaQ-Rough-Unitree-Go2",
  "Mjlab-AMP-DreamWaQ-Rough-Unitree-Go2",
  "Mjlab-CTS-Rough-Unitree-Go2",
  "Mjlab-AMP-CTS-Rough-Unitree-Go2",
  "Mjlab-AMP-TS-Rough-Unitree-Go2",
  "Mjlab-AMP-TS-Student-Rough-Unitree-Go2",
  "Mjlab-TS-Rough-Unitree-Go2",
  "Mjlab-TS-Student-Rough-Unitree-Go2",
}

CANONICAL_ALIASES = {
  "LainLoco-Go2-Velocity-Flat-v0": "Mjlab-Velocity-Flat-Unitree-Go2",
  "LainLoco-Go2-Locomotion-Rough-v0": "Mjlab-Velocity-Rough-Unitree-Go2",
  "LainLoco-Go2-Trot-Flat-v0": "Mjlab-Trot-Flat-Unitree-Go2",
  "LainLoco-Go2-Jump-Stairs-v0": "Mjlab-Jump-Flat-Unitree-Go2",
  "LainLoco-Go2-Spring-Jump-Flat-v0": "Mjlab-Spring-Jump-Flat-Unitree-Go2",
  "LainLoco-Go2-Backflip-Flat-v0": "Mjlab-Backflip-Flat-Unitree-Go2",
  "LainLoco-Go2-Handstand-Flat-v0": "Mjlab-Handstand-Flat-Unitree-Go2",
  "LainLoco-Go2-Leggedstand-Flat-v0": "Mjlab-Leggedstand-Flat-Unitree-Go2",
}


def test_mjlab_discovers_lainloco_entry_point() -> None:
  discovered = entry_points().select(group="mjlab.tasks", name="lainloco")
  assert len(tuple(discovered)) == 1


def test_entry_point_bootstraps_in_a_fresh_interpreter() -> None:
  code = """
import lainloco.robots.unitree.go2
from mjlab.tasks.registry import list_tasks
required = {
  'Mjlab-Trot-Flat-Unitree-Go2',
  'Mjlab-TS-Student-Rough-Unitree-Go2',
  'LainLoco-Go2-Trot-Flat-v0',
  'LainLoco-G1-Velocity-Flat-v0',
}
missing = required.difference(list_tasks())
if missing:
  raise SystemExit(f'missing task ids: {sorted(missing)}')
"""
  result = subprocess.run(
    [sys.executable, "-c", code],
    capture_output=True,
    text=True,
    check=False,
    timeout=30,
  )
  assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
  "legacy_module",
  (
    "mjlab.tasks.velocity.config.go2.env_cfgs",
    "mjlab.tasks.velocity.mdp.go2_actions",
    "mjlab.tasks.velocity.mdp.go2_commands",
    "mjlab.tasks.velocity.mdp.go2_events",
    "mjlab.tasks.velocity.config.go2.rl_cfg",
    "mjlab.tasks.velocity.rl.go2_algorithms.algorithms",
    "mjlab.tasks.velocity.rl.go2_algorithms.deployment",
    "mjlab.tasks.velocity.rl.go2_algorithms.models",
    "mjlab.tasks.velocity.rl.go2_algorithms.storage",
  ),
)
def test_legacy_modules_resolve_to_lainloco(legacy_module: str) -> None:
  module = import_module(legacy_module)
  assert module.__name__.startswith("lainloco.")


def test_legacy_go2_algorithms_facade_preserves_package_exports() -> None:
  legacy = import_module("mjlab.tasks.velocity.rl.go2_algorithms")
  assert vars(legacy)["CtsPPO"].__module__ == "lainloco.learning.algorithms"
  assert vars(legacy)["Go2DeploymentAdapter"].__module__ == (
    "lainloco.robots.unitree.go2.deploy.policy"
  )


def test_legacy_mdp_package_exports_resolve_to_lainloco() -> None:
  legacy_mdp = import_module("mjlab.tasks.velocity.mdp")
  command_cfg = legacy_mdp.__dict__["Go2TriggeredCommandCfg"]
  assert command_cfg.__module__ == "lainloco.robots.unitree.go2.mdp.commands"
  assert legacy_mdp.__dict__["go2_source_trot_observation"].__module__ == (
    "lainloco.robots.unitree.go2.mdp.observations"
  )
  assert legacy_mdp.__dict__["go2_trot_phase_reward"].__module__ == (
    "lainloco.robots.unitree.go2.mdp.rewards"
  )


def test_dependency_direction_is_one_way() -> None:
  mjlab_requirements = requires("mjlab") or []
  assert all(
    not requirement.lower().startswith("lainloco") for requirement in mjlab_requirements
  )


def _imported_modules(path: Path) -> set[str]:
  tree = ast.parse(path.read_text(), filename=str(path))
  imported: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      imported.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module is not None:
      imported.add(node.module)
  return imported


def test_generic_packages_do_not_depend_on_robot_domains() -> None:
  package_root = Path(__file__).parents[2] / "src/lainloco"
  offenders: list[str] = []
  for package in ("core", "learning", "runtime"):
    for path in (package_root / package).rglob("*.py"):
      if any(
        module == "lainloco.robots" or module.startswith("lainloco.robots.")
        for module in _imported_modules(path)
      ):
        offenders.append(str(path.relative_to(package_root)))
  assert offenders == []


def test_go2_deploy_does_not_depend_on_training_or_task_modules() -> None:
  package_root = Path(__file__).parents[2] / "src/lainloco"
  deploy_root = package_root / "robots/unitree/go2/deploy"
  forbidden = (
    "lainloco.learning",
    "lainloco.robots.unitree.go2.tasks",
    "lainloco.robots.unitree.go2.training",
    "mjlab.envs",
    "mjlab.tasks",
  )
  offenders: list[tuple[str, str]] = []
  for path in deploy_root.rglob("*.py"):
    for module in _imported_modules(path):
      if module.startswith(forbidden):
        offenders.append((str(path.relative_to(package_root)), module))
  assert offenders == []


def test_go2_catalogs_have_unique_expected_entries() -> None:
  assert GO2.robot_id == "unitree/go2"
  assert GO2.action_dim == 12
  assert tuple(name for name, _ in GO2.default_pose) == GO2.joint_order
  assert len(GO2.collision_geoms) == 23
  assert len(GO2_TASKS) == 8
  assert len(GO2_TRAINING_PROFILES) == 8
  assert len(GO2_EXPERIMENTS) == 16


def test_go2_policy_contracts_describe_conditional_inputs() -> None:
  dreamwaq = GO2_EXPERIMENTS.get("go2/velocity-rough::dreamwaq").experiment.contract
  teacher = GO2_EXPERIMENTS.get("go2/velocity-rough::ts-teacher").experiment.contract
  student = GO2_EXPERIMENTS.get("go2/velocity-rough::ts-student").experiment.contract

  assert tuple((field.name, field.width) for field in dreamwaq.conditional_fields) == (
    ("history", 225),
  )
  assert tuple((field.name, field.width) for field in teacher.conditional_fields) == (
    ("terrain", 187),
    ("privileged", 74),
  )
  assert student.conditional_fields == ()
  assert student.recurrent_state is not None


def test_training_profile_references_are_importable() -> None:
  fields = ("algorithm", "actor_model", "critic_model", "storage", "runner")
  for profile in GO2_TRAINING_PROFILES.values():
    for field in fields:
      module_name, qualified_name = getattr(profile, field).split(":", 1)
      value = import_module(module_name)
      for part in qualified_name.split("."):
        value = getattr(value, part)
      assert value is not None


def test_catalog_rejects_duplicate_ids() -> None:
  with pytest.raises(ValueError, match="Duplicate catalog ID"):
    Catalog(("same", "same"), id_of=lambda value: value)


def test_all_legacy_ids_remain_registered() -> None:
  assert LEGACY_GO2_IDS <= set(list_tasks())


@pytest.mark.parametrize(("canonical_id", "legacy_id"), CANONICAL_ALIASES.items())
def test_canonical_aliases_are_config_equivalent(
  canonical_id: str, legacy_id: str
) -> None:
  assert load_env_cfg(canonical_id) == load_env_cfg(legacy_id)
  assert load_env_cfg(canonical_id, play=True) == load_env_cfg(legacy_id, play=True)
  assert load_rl_cfg(canonical_id) == load_rl_cfg(legacy_id)
  assert load_runner_cls(canonical_id) is load_runner_cls(legacy_id)


def test_environment_factories_do_not_leak_mutations() -> None:
  binding = GO2_EXPERIMENTS.get("go2/trot::ppo")
  first = binding.env_factory()
  second = binding.env_factory()
  first.scene.num_envs = 123
  assert second.scene.num_envs != 123


def test_flat_and_rough_factories_do_not_inherit_from_each_other() -> None:
  flat_source = inspect.getsource(unitree_go2_flat_env_cfg)
  rough_source = inspect.getsource(unitree_go2_rough_env_cfg)
  assert "unitree_go2_rough_env_cfg" not in flat_source
  assert "unitree_go2_flat_env_cfg" not in rough_source


def test_locomotion_factories_are_owned_by_skill_modules() -> None:
  expected_modules = {
    "go2/velocity-flat::ppo": ("lainloco.robots.unitree.go2.tasks.locomotion.velocity"),
    "go2/velocity-rough::ppo": (
      "lainloco.robots.unitree.go2.tasks.locomotion.velocity"
    ),
    "go2/trot::ppo": "lainloco.robots.unitree.go2.tasks.locomotion.trot",
    "go2/jump::ppo": "lainloco.robots.unitree.go2.tasks.aerial.config",
    "go2/spring-jump::ppo": "lainloco.robots.unitree.go2.tasks.aerial.config",
    "go2/backflip::ppo": "lainloco.robots.unitree.go2.tasks.aerial.config",
    "go2/handstand::ppo": "lainloco.robots.unitree.go2.tasks.balance.config",
    "go2/leggedstand::ppo": "lainloco.robots.unitree.go2.tasks.balance.config",
  }
  for binding_id, module_name in expected_modules.items():
    assert GO2_EXPERIMENTS.get(binding_id).env_factory.__module__ == module_name
