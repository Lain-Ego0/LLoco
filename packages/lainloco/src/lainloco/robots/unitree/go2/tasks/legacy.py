"""Compatibility exports for the former monolithic Go2 environment module."""

from .aerial import (
  unitree_go2_backflip_env_cfg,
  unitree_go2_jump_env_cfg,
  unitree_go2_spring_jump_env_cfg,
)
from .balance import unitree_go2_handstand_env_cfg, unitree_go2_leggedstand_env_cfg
from .locomotion import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
  unitree_go2_trot_env_cfg,
)
from .locomotion.variants import (
  unitree_go2_amp_cts_env_cfg,
  unitree_go2_amp_dreamwaq_env_cfg,
  unitree_go2_amp_ts_env_cfg,
  unitree_go2_amp_ts_student_env_cfg,
  unitree_go2_cts_env_cfg,
  unitree_go2_dreamwaq_env_cfg,
  unitree_go2_ts_env_cfg,
  unitree_go2_ts_student_env_cfg,
)

__all__ = [
  "unitree_go2_amp_cts_env_cfg",
  "unitree_go2_amp_dreamwaq_env_cfg",
  "unitree_go2_amp_ts_env_cfg",
  "unitree_go2_amp_ts_student_env_cfg",
  "unitree_go2_backflip_env_cfg",
  "unitree_go2_cts_env_cfg",
  "unitree_go2_dreamwaq_env_cfg",
  "unitree_go2_flat_env_cfg",
  "unitree_go2_handstand_env_cfg",
  "unitree_go2_jump_env_cfg",
  "unitree_go2_leggedstand_env_cfg",
  "unitree_go2_rough_env_cfg",
  "unitree_go2_spring_jump_env_cfg",
  "unitree_go2_trot_env_cfg",
  "unitree_go2_ts_env_cfg",
  "unitree_go2_ts_student_env_cfg",
]
