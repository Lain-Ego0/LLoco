"""Go2 command generators."""

from dataclasses import dataclass

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg


class TrotVelocityCommand(UniformVelocityCommand):
  """Source sampling: 5% all-zero, then an independent 5% XY-zero draw."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    all_zero = torch.rand(len(env_ids), device=self.device) < 0.05
    self.vel_command_b[env_ids[all_zero]] = 0.0
    xy_zero = torch.rand(len(env_ids), device=self.device) < 0.05
    self.vel_command_b[env_ids[xy_zero], :2] = 0.0
    moving_xy = torch.linalg.vector_norm(
      self.vel_command_b[env_ids, :2], dim=1
    ) > 0.1
    self.vel_command_b[env_ids, :2] *= moving_xy.unsqueeze(1)


@dataclass(kw_only=True)
class TrotVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env) -> TrotVelocityCommand:
    return TrotVelocityCommand(self, env)
