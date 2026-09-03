"""mjlab task-entry-point adapter for LainLoco experiments."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType

from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
  register_mjlab_task,
)

_EXPERIMENT_MODULES = {
  "lainloco.robots.unitree.g1.experiments": "G1_EXPERIMENTS",
  "lainloco.robots.unitree.go2.experiments": "GO2_EXPERIMENTS",
}
_REGISTERING = False


def _catalog_is_being_built() -> bool:
  return any(
    module_name in sys.modules and not hasattr(sys.modules[module_name], catalog_name)
    for module_name, catalog_name in _EXPERIMENT_MODULES.items()
  )


def register_tasks() -> None:
  """Register every LainLoco experiment and compatibility alias."""
  global _REGISTERING
  if _REGISTERING or _catalog_is_being_built():
    # Importing an experiment module can itself be what first imports mjlab.
    # The module footer retries after its bindings have been constructed.
    return
  _REGISTERING = True
  try:
    from lainloco.experiments import experiment_catalog

    registered = set(list_tasks())
    for binding in experiment_catalog().values():
      if binding.registry_task_id in registered:
        env_cfg = load_env_cfg(binding.registry_task_id)
        play_env_cfg = load_env_cfg(binding.registry_task_id, play=True)
        rl_cfg = load_rl_cfg(binding.registry_task_id)
        runner_cls = load_runner_cls(binding.registry_task_id) or binding.runner_cls
      else:
        env_cfg = binding.env_factory()
        play_env_cfg = binding.env_factory(play=True)
        rl_cfg = binding.rl_factory()
        runner_cls = binding.runner_cls
      for task_id in binding.registered_ids:
        if task_id in registered:
          continue
        register_mjlab_task(
          task_id=task_id,
          env_cfg=env_cfg,
          play_env_cfg=play_env_cfg,
          rl_cfg=rl_cfg,
          runner_cls=runner_cls,
        )
        registered.add(task_id)
    _install_legacy_module_aliases()
  finally:
    _REGISTERING = False


def _install_legacy_module_aliases() -> None:
  """Keep old import paths loadable without a mjlab-to-LainLoco dependency."""
  legacy_package_name = "mjlab.tasks.velocity.rl.go2_algorithms"
  if legacy_package_name not in sys.modules:
    legacy_package = ModuleType(
      legacy_package_name,
      "Compatibility facade for Go2 learning and deployment symbols.",
    )
    legacy_package.__package__ = legacy_package_name
    legacy_package.__path__ = []  # type: ignore[attr-defined]
    public_names: list[str] = []
    for canonical_name in (
      "lainloco.learning",
      "lainloco.robots.unitree.go2.deploy",
    ):
      canonical = import_module(canonical_name)
      for symbol in canonical.__all__:
        setattr(legacy_package, symbol, getattr(canonical, symbol))
        public_names.append(symbol)
    setattr(legacy_package, "__all__", tuple(dict.fromkeys(public_names)))  # noqa: B010
    sys.modules[legacy_package_name] = legacy_package

  aliases = {
    "mjlab.tasks.velocity.config.go2.env_cfgs": (
      "lainloco.robots.unitree.go2.tasks.legacy"
    ),
    "mjlab.tasks.velocity.config.go2.rl_cfg": (
      "lainloco.robots.unitree.go2.training.config"
    ),
    "mjlab.tasks.velocity.mdp.go2_actions": ("lainloco.robots.unitree.go2.mdp.actions"),
    "mjlab.tasks.velocity.mdp.go2_commands": (
      "lainloco.robots.unitree.go2.mdp.commands"
    ),
    "mjlab.tasks.velocity.mdp.go2_events": ("lainloco.robots.unitree.go2.mdp.events"),
    "mjlab.tasks.velocity.rl.go2_algorithms.algorithms": (
      "lainloco.learning.algorithms"
    ),
    "mjlab.tasks.velocity.rl.go2_algorithms.deployment": (
      "lainloco.robots.unitree.go2.deploy.policy"
    ),
    "mjlab.tasks.velocity.rl.go2_algorithms.models": "lainloco.learning.models",
    "mjlab.tasks.velocity.rl.go2_algorithms.motion": "lainloco.learning.motion",
    "mjlab.tasks.velocity.rl.go2_algorithms.runner": "lainloco.learning.rollout",
    "mjlab.tasks.velocity.rl.go2_algorithms.storage": "lainloco.learning.storage",
    "mjlab.tasks.velocity.rl.go2_algorithms.symmetry": "lainloco.learning.symmetry",
  }
  for legacy_name, canonical_name in aliases.items():
    sys.modules.setdefault(legacy_name, import_module(canonical_name))

  legacy_runner = import_module("mjlab.tasks.velocity.rl.runner")
  canonical_runner = import_module("lainloco.robots.unitree.go2.training.runner")
  setattr(  # noqa: B010
    legacy_runner,
    "VelocityDistillationRunner",
    canonical_runner.VelocityDistillationRunner,
  )

  legacy_mdp = import_module("mjlab.tasks.velocity.mdp")
  public_mdp_symbols = {
    "Go2DelayedJointPositionAction": ("lainloco.robots.unitree.go2.mdp.actions"),
    "Go2DelayedJointPositionActionCfg": ("lainloco.robots.unitree.go2.mdp.actions"),
    "Go2TriggeredCommand": "lainloco.robots.unitree.go2.mdp.commands",
    "Go2TriggeredCommandCfg": "lainloco.robots.unitree.go2.mdp.commands",
    "go2_torque_multiplier": "lainloco.robots.unitree.go2.mdp.events",
    "reset_joints_by_scale": "lainloco.robots.unitree.go2.mdp.events",
  }
  for symbol, module_name in public_mdp_symbols.items():
    setattr(legacy_mdp, symbol, getattr(import_module(module_name), symbol))

  for category in ("observations", "rewards"):
    canonical_module = import_module(f"lainloco.robots.unitree.go2.mdp.{category}")
    legacy_module = import_module(f"mjlab.tasks.velocity.mdp.{category}")
    for symbol, value in vars(canonical_module).items():
      if symbol.startswith("go2_"):
        setattr(legacy_module, symbol, value)
        setattr(legacy_mdp, symbol, value)


# mjlab currently discovers extensions by loading the entry-point module.
# ``register_tasks`` defers aliases when the Experiment Catalog is still being
# constructed, then installs them on the catalog module's footer retry.
register_tasks()
