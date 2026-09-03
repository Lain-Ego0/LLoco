"""Go2 locomotion task factories."""

from .trot import unitree_go2_trot_env_cfg
from .velocity import unitree_go2_flat_env_cfg, unitree_go2_rough_env_cfg

__all__ = [
  "unitree_go2_flat_env_cfg",
  "unitree_go2_rough_env_cfg",
  "unitree_go2_trot_env_cfg",
]
