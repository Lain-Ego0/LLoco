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


def unitree_go2_custom_runner_cfg(kind: str) -> RslRlOnPolicyRunnerCfg:
  """Return PPO config using the migrated Go2 auxiliary update hook."""
  cfg = unitree_go2_ppo_runner_cfg()
  algorithm_classes = {
    "cts": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.CtsPPO",
    "amp_cts": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.AmpCtsPPO",
    "dreamwaq": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.DreamWaQPPO",
    "amp_dreamwaq": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.AmpDreamWaQPPO",
    "amp_ts": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.AmpTeacherStudentPPO",
    "amp_ts_student": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.AmpTeacherStudentPPO",
    "ts": "mjlab.tasks.velocity.rl.go2_algorithms.algorithms.TeacherStudentPPO",
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
  cfg.experiment_name = f"go2_{kind}"
  return cfg
