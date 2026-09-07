"""The Gym Backflip one-shot, otherwise-zero command."""
from dataclasses import dataclass
import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from ...jump.mdp.commands import RearStandVelocityCommand

class BackflipCommand(UniformVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._takeoff_frame = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  def _resample_command(self, env_ids):
    self.vel_command_b[env_ids] = 0.
    self._takeoff_frame[env_ids] = torch.randint(50, 60, (len(env_ids),), device=self.device)
  def _update_command(self, env_ids=None):
    del env_ids
    self.vel_command_b[:, 2] = torch.maximum(self.vel_command_b[:, 2], (self._env.episode_length_buf >= self._takeoff_frame).float())
  create_gui = RearStandVelocityCommand.create_gui

@dataclass(kw_only=True)
class BackflipCommandCfg(UniformVelocityCommandCfg):
  def build(self, env): return BackflipCommand(self, env)
