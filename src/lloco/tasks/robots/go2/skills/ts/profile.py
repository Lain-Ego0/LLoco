from dataclasses import dataclass


@dataclass(frozen=True)
class TSProfile:
  task_id: str = "Unitree-Go2-TS-Teacher-Rough"
  experiment_name: str = "go2_ts"
  num_envs: int = 4096
  episode_length_s: float = 20.0
  decimation: int = 4
  physics_dt: float = 0.005


TS = TSProfile()
