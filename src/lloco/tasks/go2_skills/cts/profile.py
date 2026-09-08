from dataclasses import dataclass


@dataclass(frozen=True)
class CtsProfile:
  task_id: str = "Unitree-Go2-CTS-Rough"
  experiment_name: str = "go2_cts"
  num_envs: int = 2048
  episode_length_s: float = 20.0
  decimation: int = 4
  physics_dt: float = 0.005


CTS = CtsProfile()
