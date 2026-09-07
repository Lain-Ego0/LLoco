"""Command sampler differences specific to the Gym AMP-DreamWaQ task."""

from dataclasses import dataclass

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from ..dreamwaq.mdp.commands import DreamWaQVelocityCommand


class AmpDreamWaQVelocityCommand(DreamWaQVelocityCommand):
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # AMP-DreamWaQ advances the hand-written speed schedule earlier than the
    # non-AMP task and independently injects standing/turn-in-place commands.
    iteration = self._env.common_step_counter // 24
    max_x = 1.0 if iteration <= 15_000 else 1.2 if iteration < 20_000 else 1.4 if iteration < 30_000 else 1.5
    random = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = random.uniform_(-max_x, max_x)
    self.vel_command_b[env_ids, 1] = random.uniform_(*self.cfg.ranges.lin_vel_y)
    assert self.cfg.ranges.heading is not None
    self.heading_target[env_ids] = random.uniform_(*self.cfg.ranges.heading)
    self.is_heading_env[env_ids] = True
    self.is_standing_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False

    moving = torch.linalg.vector_norm(self.vel_command_b[env_ids, :2], dim=1) > .2
    self.vel_command_b[env_ids, :2] *= moving.unsqueeze(1)
    high = env_ids < int(self.num_envs * .2)
    high_ids = env_ids[high]
    self.vel_command_b[high_ids, 0] = random[:len(high_ids)].uniform_(
      *self.cfg.ranges.lin_vel_x
    )
    self.vel_command_b[high_ids, 1] *= (
      torch.abs(self.vel_command_b[high_ids, 0]) < 1.0
    )

    standing = torch.rand(len(env_ids), device=self.device) < .05
    self.vel_command_b[env_ids[standing]] = 0.0
    turning = torch.rand(len(env_ids), device=self.device) < .05
    self.vel_command_b[env_ids[turning], :2] = 0.0


@dataclass(kw_only=True)
class AmpDreamWaQVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env) -> AmpDreamWaQVelocityCommand:
    return AmpDreamWaQVelocityCommand(self, env)
