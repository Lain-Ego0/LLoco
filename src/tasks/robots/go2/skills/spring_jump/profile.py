"""Immutable source constants for Gym ``go2_spring_jump``."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpringJumpProfile:
  task_id: str = "Unitree-Go2-Spring-Jump-Flat"
  experiment_name: str = "go2_spring_jump"
  num_envs: int = 4096
  episode_length_s: float = 5.0
  physics_dt: float = 0.005
  decimation: int = 4
  action_scale: float = 0.25
  actor_frame_dim: int = 47
  actor_history: int = 10
  critic_frame_dim: int = 65
  critic_history: int = 3


SPRING_JUMP = SpringJumpProfile()
