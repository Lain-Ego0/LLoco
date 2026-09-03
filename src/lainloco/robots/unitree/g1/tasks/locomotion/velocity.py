"""LainLoco-owned entry points for maintained mjlab G1 velocity tasks."""

from mjlab.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg as _flat_env_cfg,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_rough_env_cfg as _rough_env_cfg,
)


def unitree_g1_flat_env_cfg(play: bool = False):
  return _flat_env_cfg(play=play)


def unitree_g1_rough_env_cfg(play: bool = False):
  return _rough_env_cfg(play=play)
