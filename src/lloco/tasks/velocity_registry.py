"""Registration for velocity tasks built from robot profiles."""

from collections.abc import Callable

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from lloco.tasks.rl import make_ppo_runner_cfg

from .velocity_env import make_flat_env_cfg, make_rough_env_cfg
from .velocity_profiles import PROFILES
from .velocity_types import TerrainName, VelocityRobotProfile

_ENV_FACTORIES: dict[TerrainName, Callable[..., ManagerBasedRlEnvCfg]] = {
  "Flat": make_flat_env_cfg,
  "Rough": make_rough_env_cfg,
}
_REGISTERED = False


def register_velocity_profile(profile: VelocityRobotProfile) -> None:
  """Register all configured terrains for one robot profile."""
  for terrain in profile.terrains:
    experiment_name = (
      f"{profile.task_group.lower().replace('-', '_')}_"
      f"{profile.task_name.lower().replace('-', '_')}_"
      f"{terrain.lower()}_velocity"
    )
    runner_cfg = make_ppo_runner_cfg(
      experiment_name,
      max_iterations=profile.max_iterations,
    )
    env_factory = _ENV_FACTORIES[terrain]
    register_mjlab_task(
      task_id=f"{profile.task_group}-{profile.task_name}-{terrain}",
      env_cfg=env_factory(profile),
      play_env_cfg=env_factory(profile, play=True),
      rl_cfg=runner_cfg,
      runner_cls=VelocityOnPolicyRunner,
    )


def register_velocity_tasks() -> None:
  """Register every built-in velocity profile exactly once."""
  global _REGISTERED
  if _REGISTERED:
    return
  for profile in PROFILES:
    register_velocity_profile(profile)
  _REGISTERED = True
