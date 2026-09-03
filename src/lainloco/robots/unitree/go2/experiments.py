"""Go2 experiment compositions and mjlab compatibility bindings."""

from __future__ import annotations

from functools import partial

from lainloco.core import Catalog, ObservationField, compose_experiment
from lainloco.integrations.mjlab import ConfigFactory, MjlabExperimentBinding
from lainloco.robots.unitree.go2.training.runner import (
  VelocityDistillationRunner,
  VelocityOnPolicyRunner,
)

from .contract import GO2_POLICY_INPUTS, make_go2_policy_contract
from .deploy.policy import go2_policy_contract_metadata
from .robot import GO2
from .tasks import GO2_TASKS
from .tasks.aerial import (
  unitree_go2_backflip_env_cfg,
  unitree_go2_jump_env_cfg,
  unitree_go2_spring_jump_env_cfg,
)
from .tasks.balance import (
  unitree_go2_handstand_env_cfg,
  unitree_go2_leggedstand_env_cfg,
)
from .tasks.locomotion import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
  unitree_go2_trot_env_cfg,
)
from .tasks.locomotion.variants import (
  unitree_go2_amp_cts_env_cfg,
  unitree_go2_amp_dreamwaq_env_cfg,
  unitree_go2_amp_ts_env_cfg,
  unitree_go2_amp_ts_student_env_cfg,
  unitree_go2_cts_env_cfg,
  unitree_go2_dreamwaq_env_cfg,
  unitree_go2_ts_env_cfg,
  unitree_go2_ts_student_env_cfg,
)
from .training import GO2_TRAINING_PROFILES
from .training.config import (
  unitree_go2_custom_runner_cfg,
  unitree_go2_ppo_runner_cfg,
  unitree_go2_source_ppo_runner_cfg,
)


def _binding(
  *,
  binding_id: str,
  task_id: str,
  profile_id: str,
  legacy_task_id: str,
  env_factory: ConfigFactory,
  rl_factory: ConfigFactory,
  actor_dim: int,
  canonical_task_id: str | None = None,
  history_length: int = 1,
  recurrent: bool = False,
  conditional_fields: tuple[ObservationField, ...] = (),
  runner_cls: type = VelocityOnPolicyRunner,
  distillation: bool = False,
) -> MjlabExperimentBinding:
  task = GO2_TASKS.get(task_id)
  experiment = compose_experiment(
    robot=GO2,
    task=task,
    training=GO2_TRAINING_PROFILES.get(profile_id),
    contract=make_go2_policy_contract(
      task.task_id,
      actor_dim,
      history_length=history_length,
      recurrent=recurrent,
      conditional_fields=conditional_fields,
    ),
  )
  return MjlabExperimentBinding(
    binding_id=binding_id,
    experiment=experiment,
    registry_task_id=legacy_task_id,
    canonical_task_id=canonical_task_id,
    env_factory=env_factory,
    rl_factory=rl_factory,
    runner_cls=runner_cls,
    distillation=distillation,
    metadata_factory=go2_policy_contract_metadata,
  )


_SOURCE_SYMMETRY = "lainloco.learning.symmetry"

GO2_EXPERIMENTS = Catalog(
  (
    _binding(
      binding_id="go2/velocity-flat::ppo",
      task_id="go2/velocity-flat",
      profile_id="ppo",
      legacy_task_id="Mjlab-Velocity-Flat-Unitree-Go2",
      canonical_task_id="LainLoco-Go2-Velocity-Flat-v0",
      env_factory=unitree_go2_flat_env_cfg,
      rl_factory=unitree_go2_ppo_runner_cfg,
      actor_dim=48,
    ),
    _binding(
      binding_id="go2/velocity-rough::ppo",
      task_id="go2/velocity-rough",
      profile_id="ppo",
      legacy_task_id="Mjlab-Velocity-Rough-Unitree-Go2",
      canonical_task_id="LainLoco-Go2-Locomotion-Rough-v0",
      env_factory=unitree_go2_rough_env_cfg,
      rl_factory=unitree_go2_ppo_runner_cfg,
      actor_dim=235,
    ),
    _binding(
      binding_id="go2/trot::ppo",
      task_id="go2/trot",
      profile_id="ppo",
      legacy_task_id="Mjlab-Trot-Flat-Unitree-Go2",
      canonical_task_id="LainLoco-Go2-Trot-Flat-v0",
      env_factory=unitree_go2_trot_env_cfg,
      rl_factory=partial(
        unitree_go2_source_ppo_runner_cfg,
        learning_rate=1.0e-5,
        symmetry_func=f"{_SOURCE_SYMMETRY}.trot_jump_symmetry",
      ),
      actor_dim=470,
      history_length=10,
    ),
    _binding(
      binding_id="go2/jump::ppo",
      task_id="go2/jump",
      profile_id="ppo",
      legacy_task_id="Mjlab-Jump-Flat-Unitree-Go2",
      canonical_task_id="LainLoco-Go2-Jump-Stairs-v0",
      env_factory=unitree_go2_jump_env_cfg,
      rl_factory=partial(
        unitree_go2_source_ppo_runner_cfg,
        learning_rate=1.0e-4,
        max_iterations=15_000,
        symmetry_func=f"{_SOURCE_SYMMETRY}.trot_jump_symmetry",
      ),
      actor_dim=470,
      history_length=10,
    ),
    _binding(
      binding_id="go2/spring-jump::ppo",
      task_id="go2/spring-jump",
      profile_id="ppo",
      legacy_task_id="Mjlab-Spring-Jump-Flat-Unitree-Go2",
      canonical_task_id="LainLoco-Go2-Spring-Jump-Flat-v0",
      env_factory=unitree_go2_spring_jump_env_cfg,
      rl_factory=partial(
        unitree_go2_source_ppo_runner_cfg,
        learning_rate=1.0e-5,
        max_iterations=50_000,
        symmetry_func=f"{_SOURCE_SYMMETRY}.spring_jump_symmetry",
      ),
      actor_dim=470,
      history_length=10,
    ),
    _binding(
      binding_id="go2/backflip::ppo",
      task_id="go2/backflip",
      profile_id="ppo",
      legacy_task_id="Mjlab-Backflip-Flat-Unitree-Go2",
      canonical_task_id="LainLoco-Go2-Backflip-Flat-v0",
      env_factory=unitree_go2_backflip_env_cfg,
      rl_factory=partial(
        unitree_go2_source_ppo_runner_cfg,
        learning_rate=1.0e-5,
        max_iterations=50_000,
      ),
      actor_dim=470,
      history_length=10,
    ),
    _binding(
      binding_id="go2/handstand::ppo",
      task_id="go2/handstand",
      profile_id="ppo",
      legacy_task_id="Mjlab-Handstand-Flat-Unitree-Go2",
      canonical_task_id="LainLoco-Go2-Handstand-Flat-v0",
      env_factory=unitree_go2_handstand_env_cfg,
      rl_factory=partial(
        unitree_go2_source_ppo_runner_cfg,
        learning_rate=1.0e-3,
        max_iterations=15_000,
        symmetry_func=f"{_SOURCE_SYMMETRY}.handstand_symmetry",
      ),
      actor_dim=45,
    ),
    _binding(
      binding_id="go2/leggedstand::ppo",
      task_id="go2/leggedstand",
      profile_id="ppo",
      legacy_task_id="Mjlab-Leggedstand-Flat-Unitree-Go2",
      canonical_task_id="LainLoco-Go2-Leggedstand-Flat-v0",
      env_factory=unitree_go2_leggedstand_env_cfg,
      rl_factory=partial(
        unitree_go2_source_ppo_runner_cfg,
        learning_rate=1.0e-3,
        max_iterations=15_000,
      ),
      actor_dim=45,
    ),
    _binding(
      binding_id="go2/velocity-rough::dreamwaq",
      task_id="go2/velocity-rough",
      profile_id="dreamwaq",
      legacy_task_id="Mjlab-DreamWaQ-Rough-Unitree-Go2",
      env_factory=unitree_go2_dreamwaq_env_cfg,
      rl_factory=partial(unitree_go2_custom_runner_cfg, "dreamwaq"),
      actor_dim=GO2_POLICY_INPUTS.actor_dim,
      history_length=GO2_POLICY_INPUTS.history_length,
      conditional_fields=(ObservationField("history", GO2_POLICY_INPUTS.history_dim),),
    ),
    _binding(
      binding_id="go2/velocity-rough::amp-dreamwaq",
      task_id="go2/velocity-rough",
      profile_id="amp-dreamwaq",
      legacy_task_id="Mjlab-AMP-DreamWaQ-Rough-Unitree-Go2",
      env_factory=unitree_go2_amp_dreamwaq_env_cfg,
      rl_factory=partial(unitree_go2_custom_runner_cfg, "amp_dreamwaq"),
      actor_dim=GO2_POLICY_INPUTS.actor_dim,
      history_length=GO2_POLICY_INPUTS.history_length,
      conditional_fields=(ObservationField("history", GO2_POLICY_INPUTS.history_dim),),
    ),
    _binding(
      binding_id="go2/velocity-rough::cts",
      task_id="go2/velocity-rough",
      profile_id="cts",
      legacy_task_id="Mjlab-CTS-Rough-Unitree-Go2",
      env_factory=unitree_go2_cts_env_cfg,
      rl_factory=partial(unitree_go2_custom_runner_cfg, "cts"),
      actor_dim=GO2_POLICY_INPUTS.actor_dim,
      history_length=GO2_POLICY_INPUTS.history_length,
      conditional_fields=(ObservationField("history", GO2_POLICY_INPUTS.history_dim),),
    ),
    _binding(
      binding_id="go2/velocity-rough::amp-cts",
      task_id="go2/velocity-rough",
      profile_id="amp-cts",
      legacy_task_id="Mjlab-AMP-CTS-Rough-Unitree-Go2",
      env_factory=unitree_go2_amp_cts_env_cfg,
      rl_factory=partial(unitree_go2_custom_runner_cfg, "amp_cts"),
      actor_dim=GO2_POLICY_INPUTS.actor_dim,
      history_length=GO2_POLICY_INPUTS.history_length,
      conditional_fields=(ObservationField("history", GO2_POLICY_INPUTS.history_dim),),
    ),
    _binding(
      binding_id="go2/velocity-rough::amp-ts-teacher",
      task_id="go2/velocity-rough",
      profile_id="amp-ts-teacher",
      legacy_task_id="Mjlab-AMP-TS-Rough-Unitree-Go2",
      env_factory=unitree_go2_amp_ts_env_cfg,
      rl_factory=partial(unitree_go2_custom_runner_cfg, "amp_ts"),
      actor_dim=GO2_POLICY_INPUTS.actor_dim,
      conditional_fields=(
        ObservationField("terrain", GO2_POLICY_INPUTS.terrain_dim),
        ObservationField("privileged", GO2_POLICY_INPUTS.ts_privileged_dim),
      ),
    ),
    _binding(
      binding_id="go2/velocity-rough::amp-ts-student-legacy",
      task_id="go2/velocity-rough",
      profile_id="ts-student",
      legacy_task_id="Mjlab-AMP-TS-Student-Rough-Unitree-Go2",
      env_factory=unitree_go2_amp_ts_student_env_cfg,
      rl_factory=partial(unitree_go2_custom_runner_cfg, "amp_ts_student"),
      actor_dim=GO2_POLICY_INPUTS.actor_dim,
      history_length=GO2_POLICY_INPUTS.history_length,
      recurrent=True,
      runner_cls=VelocityDistillationRunner,
      distillation=True,
    ),
    _binding(
      binding_id="go2/velocity-rough::ts-teacher",
      task_id="go2/velocity-rough",
      profile_id="ts-teacher",
      legacy_task_id="Mjlab-TS-Rough-Unitree-Go2",
      env_factory=unitree_go2_ts_env_cfg,
      rl_factory=partial(unitree_go2_custom_runner_cfg, "ts"),
      actor_dim=GO2_POLICY_INPUTS.actor_dim,
      conditional_fields=(
        ObservationField("terrain", GO2_POLICY_INPUTS.terrain_dim),
        ObservationField("privileged", GO2_POLICY_INPUTS.ts_privileged_dim),
      ),
    ),
    _binding(
      binding_id="go2/velocity-rough::ts-student",
      task_id="go2/velocity-rough",
      profile_id="ts-student",
      legacy_task_id="Mjlab-TS-Student-Rough-Unitree-Go2",
      env_factory=unitree_go2_ts_student_env_cfg,
      rl_factory=partial(unitree_go2_custom_runner_cfg, "ts_student"),
      actor_dim=GO2_POLICY_INPUTS.actor_dim,
      history_length=GO2_POLICY_INPUTS.history_length,
      recurrent=True,
      runner_cls=VelocityDistillationRunner,
      distillation=True,
    ),
  ),
  id_of=lambda binding: binding.binding_id,
)


def resolve_experiment(task_id: str, profile_id: str) -> MjlabExperimentBinding:
  """Resolve one explicit task/profile composition.

  The AMP-student legacy job intentionally has no new profile identity. New
  distillation commands use ``ts-student`` with an explicit teacher checkpoint.
  """
  return GO2_EXPERIMENTS.get(f"{task_id}::{profile_id}")


# Handles the valid import order where this module is what first imports mjlab.
from lainloco.bootstrap import register_tasks as _register_tasks  # noqa: E402

_register_tasks()
