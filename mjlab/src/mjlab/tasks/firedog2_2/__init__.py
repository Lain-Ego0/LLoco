"""RoboLab FireDog 2.2 simulation-first task bindings."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.firedog2_2.firedog2_2_env_cfg import (
  firedog2_2_velocity_env_cfg,
  firedog2_2_velocity_flat_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


def firedog2_2_velocity_flat_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 128, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.005,
      desired_kl=0.01,
      value_loss_coef=1.0,
      max_grad_norm=1.0,
    ),
    experiment_name="firedog2_2_velocity_flat",
    run_name="r3-real-policy",
    logger="tensorboard",
    save_interval=50,
    num_steps_per_env=32,
    max_iterations=2500,
    clip_actions=1.0,
  )


register_mjlab_task(
  task_id="robolab.motion.velocity.flat",
  env_cfg=firedog2_2_velocity_flat_env_cfg(),
  play_env_cfg=firedog2_2_velocity_flat_env_cfg(play=True),
  rl_cfg=firedog2_2_velocity_flat_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

__all__ = [
  "firedog2_2_velocity_env_cfg",
  "firedog2_2_velocity_flat_env_cfg",
  "firedog2_2_velocity_flat_ppo_cfg",
]
