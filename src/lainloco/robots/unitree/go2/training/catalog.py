"""Go2 bindings to reusable learning profiles."""

from lainloco.core import Catalog, TrainingSpec


def _profile(
  profile_id: str,
  algorithm: str = "rsl_rl.algorithms.ppo:PPO",
  actor: str = "rsl_rl.models.mlp_model:MLPModel",
  critic: str = "rsl_rl.models.mlp_model:MLPModel",
  storage: str = "rsl_rl.storage.rollout_storage:RolloutStorage",
  runner: str = "lainloco.robots.unitree.go2.training.runner:VelocityOnPolicyRunner",
  auxiliary: tuple[str, ...] = (),
  groups: tuple[str, ...] = ("actor", "critic"),
) -> TrainingSpec:
  return TrainingSpec(
    profile_id=profile_id,
    algorithm=algorithm,
    actor_model=actor,
    critic_model=critic,
    storage=storage,
    runner=runner,
    optimizer="torch.optim:Adam",
    auxiliary_losses=auxiliary,
    required_observation_groups=groups,
    exporter="mjlab.rl.runner:MjlabOnPolicyRunner.export_policy_to_onnx",
  )


GO2_TRAINING_PROFILES = Catalog(
  (
    _profile("ppo"),
    _profile(
      "dreamwaq",
      algorithm=("lainloco.learning.algorithms:DreamWaQPPO"),
      actor="lainloco.learning.models:DreamWaQActorModel",
      storage="lainloco.learning.storage:Go2RolloutStorage",
      auxiliary=("velocity-supervision", "vae-reconstruction", "vae-kl"),
      groups=("actor", "critic", "history"),
    ),
    _profile(
      "amp-dreamwaq",
      algorithm=("lainloco.learning.algorithms:AmpDreamWaQPPO"),
      actor="lainloco.learning.models:DreamWaQActorModel",
      storage="lainloco.learning.storage:Go2RolloutStorage",
      auxiliary=("amp", "velocity-supervision", "vae-reconstruction", "vae-kl"),
      groups=("actor", "critic", "history", "amp"),
    ),
    _profile(
      "cts",
      algorithm="lainloco.learning.algorithms:CtsPPO",
      actor="lainloco.learning.models:CtsActorModel",
      critic="lainloco.learning.models:CtsCriticModel",
      storage="lainloco.learning.storage:Go2RolloutStorage",
      auxiliary=("teacher-student-latent",),
      groups=("actor", "critic", "history"),
    ),
    _profile(
      "amp-cts",
      algorithm="lainloco.learning.algorithms:AmpCtsPPO",
      actor="lainloco.learning.models:CtsActorModel",
      critic="lainloco.learning.models:CtsCriticModel",
      storage="lainloco.learning.storage:Go2RolloutStorage",
      auxiliary=("amp", "teacher-student-latent"),
      groups=("actor", "critic", "history", "amp"),
    ),
    _profile(
      "ts-teacher",
      algorithm=("lainloco.learning.algorithms:Go2AuxiliaryPPO"),
      actor="lainloco.learning.models:TeacherActorModel",
      storage="lainloco.learning.storage:Go2RolloutStorage",
      auxiliary=("terrain-encoder", "privileged-encoder"),
      groups=("actor", "critic", "terrain", "privileged"),
    ),
    _profile(
      "amp-ts-teacher",
      algorithm="lainloco.learning.algorithms:AmpPPO",
      actor="lainloco.learning.models:TeacherActorModel",
      storage="lainloco.learning.storage:Go2RolloutStorage",
      auxiliary=("amp", "terrain-encoder", "privileged-encoder"),
      groups=("actor", "critic", "terrain", "privileged", "amp"),
    ),
    _profile(
      "ts-student",
      algorithm=("lainloco.learning.algorithms:TeacherStudentPPO"),
      actor="lainloco.learning.models:StudentActorModel",
      storage="lainloco.learning.storage:Go2RolloutStorage",
      runner=("lainloco.robots.unitree.go2.training.runner:VelocityDistillationRunner"),
      auxiliary=("teacher-action-distillation",),
      groups=("actor", "critic", "history"),
    ),
  ),
  id_of=lambda profile: profile.profile_id,
)
