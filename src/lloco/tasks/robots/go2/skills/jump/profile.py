"""Immutable constants for the Jump task."""

from dataclasses import dataclass


@dataclass(frozen=True)
class JumpProfile:
  task_id: str = "Unitree-Go2-Jump-Flat"
  experiment_name: str = "go2_jump"
  num_envs: int = 4096
  episode_length_s: float = 24.0
  physics_dt: float = 0.005
  decimation: int = 4
  action_scale: float = 0.25
  actor_frame_dim: int = 47
  actor_history: int = 10
  critic_frame_dim: int = 70
  critic_history: int = 3
  cycle_time: float = 1.5
  target_foot_height: float = 0.05
  base_height_target: float = 0.3


JUMP = JumpProfile()
