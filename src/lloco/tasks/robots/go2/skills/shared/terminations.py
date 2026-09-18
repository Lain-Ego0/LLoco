"""Termination terms shared by Go2 skills."""

import torch
from mjlab.sensor import ContactSensor


def base_contact(env, sensor_name: str, force_threshold: float = 1.0) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  return torch.linalg.vector_norm(force, dim=-1).amax(dim=1) > force_threshold
