"""G1 training configuration entry points."""

from mjlab.tasks.velocity.config.g1.rl_cfg import (
  unitree_g1_ppo_runner_cfg as _ppo_runner_cfg,
)


def unitree_g1_ppo_runner_cfg():
  return _ppo_runner_cfg()
