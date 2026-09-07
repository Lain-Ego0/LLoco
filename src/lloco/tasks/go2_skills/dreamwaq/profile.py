"""Immutable source constants for ``go2_dreamwaq``."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DreamWaQProfile:
  task_id: str = "Unitree-Go2-DreamWaQ-Rough"
  experiment_name: str = "go2_dreamwaq"
  num_envs: int = 4096
  episode_length_s: float = 20.0
  physics_dt: float = 0.005
  decimation: int = 4
  action_scale: float = 0.25
  actor_dim: int = 45
  actor_history: int = 5
  critic_dim: int = 261
  critic_history: int = 3
  latent_dim: int = 16
  explicit_dim: int = 3


DREAMWAQ = DreamWaQProfile()
