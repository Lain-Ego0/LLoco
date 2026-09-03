"""RL configuration for Unitree Go2 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go2 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )


def unitree_go2_source_ppo_runner_cfg(
  *,
  learning_rate: float = 1.0e-3,
  max_iterations: int = 15_000,
  save_interval: int = 100,
  seed: int = 1,
  symmetry_func: str | None = None,
) -> RslRlOnPolicyRunnerCfg:
  """Create PPO config matching the source RSL-RL observation handling."""
  cfg = unitree_go2_ppo_runner_cfg()
  # The source policies consume the environment's already scaled/clipped
  # observations directly and do not apply a running observation normalizer.
  cfg.actor.obs_normalization = False
  cfg.critic.obs_normalization = False
  cfg.clip_actions = 100.0
  cfg.algorithm.learning_rate = learning_rate
  cfg.max_iterations = max_iterations
  cfg.save_interval = save_interval
  cfg.seed = seed
  if symmetry_func is not None:
    cfg.algorithm.symmetry_cfg = {
      "data_augmentation_func": symmetry_func,
      "use_data_augmentation": False,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 1.0,
    }
  return cfg


def unitree_go2_custom_runner_cfg(kind: str) -> RslRlOnPolicyRunnerCfg:
  """Return PPO config using the migrated Go2 auxiliary update hook."""
  is_ts_family = kind in ("amp_ts", "amp_ts_student", "ts", "ts_student")
  cfg = unitree_go2_source_ppo_runner_cfg(
    learning_rate=1.0e-5 if kind in ("amp_ts", "ts") else 1.0e-3,
    max_iterations=20_000,
    save_interval=500,
    seed=5 if is_ts_family else 1,
  )
  algorithm_classes = {
    "cts": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.CtsPPO",
    "amp_cts": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.AmpCtsPPO",
    "dreamwaq": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.DreamWaQPPO",
    "amp_dreamwaq": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.AmpDreamWaQPPO",
    "amp_ts": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.AmpPPO",
    # Source student jobs are pure recurrent behavior distillation.  Their
    # dedicated runner does not execute PPO or AMP discriminator updates.
    "amp_ts_student": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.TeacherStudentPPO",
    "ts": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.Go2AuxiliaryPPO",
    "ts_student": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.TeacherStudentPPO",
  }
  actor_classes = {
    "cts": "mjlab.tasks.velocity.rl.go2_algorithms.models.CtsActorModel",
    "amp_cts": "mjlab.tasks.velocity.rl.go2_algorithms.models.CtsActorModel",
    "dreamwaq": "mjlab.tasks.velocity.rl.go2_algorithms.models.DreamWaQActorModel",
    "amp_dreamwaq": "mjlab.tasks.velocity.rl.go2_algorithms.models.DreamWaQActorModel",
    "amp_ts": "mjlab.tasks.velocity.rl.go2_algorithms.models.TeacherActorModel",
    "amp_ts_student": "mjlab.tasks.velocity.rl.go2_algorithms.models.StudentActorModel",
    "ts": "mjlab.tasks.velocity.rl.go2_algorithms.models.TeacherActorModel",
    "ts_student": "mjlab.tasks.velocity.rl.go2_algorithms.models.StudentActorModel",
  }
  critic_classes = {
    "cts": "mjlab.tasks.velocity.rl.go2_algorithms.models.CtsCriticModel",
    "amp_cts": "mjlab.tasks.velocity.rl.go2_algorithms.models.CtsCriticModel",
  }
  try:
    cfg.algorithm.class_name = algorithm_classes[kind]
    cfg.actor.class_name = actor_classes[kind]
    if kind in critic_classes:
      cfg.critic.class_name = critic_classes[kind]
  except KeyError as exc:
    raise ValueError(f"Unknown Go2 custom algorithm: {kind}") from exc
  # The legacy TS distillation runners collect 50 frames per update (the
  # student LSTM is trained on that longer rollout) and clamp every action
  # standard deviation to 0.05.  Encode both constraints in the current
  # runner config instead of relying on a deployment-only workaround.
  if kind in ("amp_ts_student", "ts_student"):
    cfg.num_steps_per_env = 50
  if kind in (
    "amp_cts", "amp_dreamwaq", "amp_ts", "amp_ts_student", "ts", "ts_student"
  ):
    # Legacy min_normalized_std=0.05 is multiplied by each joint's complete
    # position range, yielding a distinct floor for hip/thigh/calf noise.
    cfg.actor.distribution_cfg["class_name"] = (
      "mjlab.tasks.velocity.rl.go2_algorithms.models."
      "Go2ClampedGaussianDistribution"
    )
    cfg.actor.distribution_cfg["min_std"] = (
      0.10472, 0.253075, 0.094247,
    ) * 4
  cfg.experiment_name = f"go2_{kind}"
  return cfg
