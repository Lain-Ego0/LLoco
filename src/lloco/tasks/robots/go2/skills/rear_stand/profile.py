"""Immutable constants for the Rear Stand task."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RearStandProfile:
  display_name: str = "Rear Stand"
  task_id: str = "Unitree-Go2-Rear-Stand-Flat"
  experiment_name: str = "go2_rear_stand"
  num_envs: int = 4096
  episode_length_s: float = 20.0
  physics_dt: float = 0.005
  decimation: int = 4
  action_scale: float = 0.25
  actor_frame_dim: int = 45
  actor_history: int = 1
  critic_frame_dim: int = 86
  critic_history: int = 1
  cycle_time: float = 1.6
  target_foot_height: float = 0.06
  base_height_target: float = 0.52


REAR_STAND = RearStandProfile()
