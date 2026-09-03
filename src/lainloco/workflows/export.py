"""Checkpoint-to-Policy-Bundle export workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from lainloco.robots.unitree.go2.deploy.policy import go2_policy_contract_metadata
from lainloco.robots.unitree.go2.experiments import resolve_experiment
from lainloco.robots.unitree.go2.training.runner import VelocityDistillationRunner
from lainloco.runtime import (
  LoadedPolicyBundle,
  create_policy_bundle,
)


@dataclass(frozen=True, slots=True)
class PolicyExportPlan:
  """Validated checkpoint and experiment selection for bundle export."""

  checkpoint: Path
  destination: Path
  task_id: str
  profile_id: str
  registry_task_id: str
  device: str


def build_policy_export_plan(
  checkpoint: str | Path,
  destination: str | Path,
  *,
  task_id: str,
  profile_id: str,
  device: str = "cpu",
) -> PolicyExportPlan:
  """Resolve an experiment without loading an untrusted checkpoint yet."""
  checkpoint_path = Path(checkpoint).expanduser().resolve()
  destination_path = Path(destination).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
  if destination_path.exists():
    raise FileExistsError(
      f"Policy bundle destination already exists: {destination_path}"
    )
  if not device:
    raise ValueError("device must be non-empty")
  binding = resolve_experiment(task_id, profile_id)
  return PolicyExportPlan(
    checkpoint=checkpoint_path,
    destination=destination_path,
    task_id=task_id,
    profile_id=profile_id,
    registry_task_id=binding.legacy_task_id,
    device=device,
  )


def export_policy_bundle(plan: PolicyExportPlan) -> LoadedPolicyBundle:
  """Load a checkpoint, export its deterministic actor, and create a bundle."""
  binding = resolve_experiment(plan.task_id, plan.profile_id)
  env_cfg = load_env_cfg(plan.registry_task_id, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(plan.registry_task_id)
  env = ManagerBasedRlEnv(env_cfg, device=plan.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  try:
    runner = binding.runner_cls(
      wrapped, asdict(agent_cfg), log_dir=None, device=plan.device
    )
    runner.load(
      str(plan.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=plan.device,
    )
    with TemporaryDirectory(prefix="lainloco-export-") as temporary:
      export_dir = Path(temporary)
      filename = "policy.onnx"
      if binding.experiment.contract.recurrent_state is None:
        runner.export_policy_to_onnx(str(export_dir), filename)
      else:
        if not isinstance(runner, VelocityDistillationRunner):
          raise RuntimeError("Recurrent policy export requires a distillation runner")
        if not runner.export_recurrent_policy_to_onnx(str(export_dir), filename):
          raise RuntimeError("Selected runner did not provide a recurrent ONNX policy")
      policy_path = export_dir / filename
      metadata = get_base_metadata(env, "local-export")
      metadata.update(
        go2_policy_contract_metadata(
          runner.alg.get_policy(),
          recurrent=binding.experiment.contract.recurrent_state is not None,
        )
      )
      attach_metadata_to_onnx(str(policy_path), metadata)
      return create_policy_bundle(plan.destination, policy_path, binding.experiment)
  finally:
    env.close()
