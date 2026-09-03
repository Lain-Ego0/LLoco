"""G1 bindings to reusable learning profiles."""

from lainloco.core import Catalog, TrainingSpec

G1_TRAINING_PROFILES = Catalog(
  (
    TrainingSpec(
      profile_id="ppo",
      algorithm="rsl_rl.algorithms.ppo:PPO",
      actor_model="rsl_rl.models.mlp_model:MLPModel",
      critic_model="rsl_rl.models.mlp_model:MLPModel",
      storage="rsl_rl.storage.rollout_storage:RolloutStorage",
      runner="mjlab.tasks.velocity.rl.runner:VelocityOnPolicyRunner",
      optimizer="torch.optim:Adam",
      auxiliary_losses=(),
      required_observation_groups=("actor", "critic"),
      exporter="mjlab.rl.runner:MjlabOnPolicyRunner.export_policy_to_onnx",
    ),
  ),
  id_of=lambda profile: profile.profile_id,
)
