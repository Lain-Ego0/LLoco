"""Go2 balance skill environment factories."""

from mjlab.envs import ManagerBasedRlEnvCfg

from ..special import _go2_special_action_env_cfg


def unitree_go2_handstand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_special_action_env_cfg("handstand", play)


def unitree_go2_leggedstand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_special_action_env_cfg("leggedstand", play)
