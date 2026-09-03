"""G1 experiment compositions and mjlab bindings."""

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from lainloco.core import Catalog, compose_experiment
from lainloco.integrations.mjlab import MjlabExperimentBinding

from .contract import make_g1_policy_contract
from .robot import G1
from .tasks import G1_TASKS
from .tasks.locomotion import unitree_g1_flat_env_cfg, unitree_g1_rough_env_cfg
from .training import G1_TRAINING_PROFILES
from .training.config import unitree_g1_ppo_runner_cfg


def _binding(
  task_id: str,
  registry_task_id: str,
  canonical_task_id: str,
  env_factory,
  actor_dim: int,
) -> MjlabExperimentBinding:
  task = G1_TASKS.get(task_id)
  return MjlabExperimentBinding(
    binding_id=f"{task_id}::ppo",
    experiment=compose_experiment(
      robot=G1,
      task=task,
      training=G1_TRAINING_PROFILES.get("ppo"),
      contract=make_g1_policy_contract(task_id, actor_dim),
    ),
    registry_task_id=registry_task_id,
    canonical_task_id=canonical_task_id,
    env_factory=env_factory,
    rl_factory=unitree_g1_ppo_runner_cfg,
    runner_cls=VelocityOnPolicyRunner,
  )


G1_EXPERIMENTS = Catalog(
  (
    _binding(
      "g1/velocity-flat",
      "Mjlab-Velocity-Flat-Unitree-G1",
      "LainLoco-G1-Velocity-Flat-v0",
      unitree_g1_flat_env_cfg,
      99,
    ),
    _binding(
      "g1/velocity-rough",
      "Mjlab-Velocity-Rough-Unitree-G1",
      "LainLoco-G1-Velocity-Rough-v0",
      unitree_g1_rough_env_cfg,
      286,
    ),
  ),
  id_of=lambda binding: binding.binding_id,
)


def resolve_experiment(task_id: str, profile_id: str) -> MjlabExperimentBinding:
  return G1_EXPERIMENTS.get(f"{task_id}::{profile_id}")


from lainloco.bootstrap import register_tasks as _register_tasks  # noqa: E402

_register_tasks()
