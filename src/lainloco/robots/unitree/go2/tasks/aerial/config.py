"""Go2 aerial skill environment factories."""

from mjlab.envs import ManagerBasedRlEnvCfg

from ..special import _go2_special_action_env_cfg


def unitree_go2_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_special_action_env_cfg("jump", play)


def unitree_go2_spring_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_special_action_env_cfg("spring_jump", play)


def unitree_go2_backflip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_special_action_env_cfg("backflip", play)
