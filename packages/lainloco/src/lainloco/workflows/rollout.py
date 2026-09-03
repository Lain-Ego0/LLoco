"""Go2 experiment orchestration for the generic sim-to-sim control loop."""

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from lainloco.robots.unitree.go2.experiments import resolve_experiment
from lainloco.runtime import (
  BundlePolicyRuntime,
  MjlabSimulationBackend,
  SimToSimRuntime,
  SimToSimStats,
  load_policy_bundle,
)


def run_mjlab_bundle(
  bundle_path: str,
  *,
  steps: int,
  num_envs: int = 1,
  device: str = "cpu",
) -> SimToSimStats:
  """Resolve a Go2 bundle and run it in a fresh headless mjlab environment."""
  if num_envs <= 0:
    raise ValueError("num_envs must be positive")
  initial = load_policy_bundle(bundle_path)
  binding = resolve_experiment(
    initial.manifest.task_id, initial.manifest.training_profile_id
  )
  bundle = load_policy_bundle(bundle_path, expected_experiment=binding.experiment)
  cfg = load_env_cfg(binding.legacy_task_id, play=True)
  cfg.scene.num_envs = num_envs
  backend = MjlabSimulationBackend(ManagerBasedRlEnv(cfg, device=device))
  try:
    policy = BundlePolicyRuntime(bundle, batch_size=num_envs)
    return SimToSimRuntime(policy, backend).run(steps)
  finally:
    backend.close()
