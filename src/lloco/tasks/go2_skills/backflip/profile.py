"""Immutable constants from registered Gym ``go2_backflip``."""
from dataclasses import dataclass

@dataclass(frozen=True)
class BackflipProfile:
  task_id: str = "Unitree-Go2-Backflip-Flat"
  experiment_name: str = "go2_backflip"
  num_envs: int = 4096
  episode_length_s: float = 4.0
  physics_dt: float = .005
  decimation: int = 4
  action_scale: float = .25

BACKFLIP = BackflipProfile()
