"""One-shot target and take-off command used by ``go2_spring_jump``."""

from dataclasses import dataclass

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

from ...jump.mdp.commands import RearStandVelocityCommand


class SpringJumpCommand(UniformVelocityCommand):
  """Keep the sampled landing X target and raise the jump flag once per reset.

  The Gym implementation does not actually resample its command every five
  seconds despite retaining that inherited configuration field.  It samples X
  on reset and flips command[2] at a randomly selected policy frame 50--59.
  """

  def __init__(self, cfg, env) -> None:
    super().__init__(cfg, env)
    self._takeoff_frame = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.vel_command_b[env_ids] = 0.0
    self.vel_command_b[env_ids, 0].uniform_(0.8, 1.2)
    self._takeoff_frame[env_ids] = torch.randint(50, 60, (len(env_ids),), device=self.device)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids
    # Gym changes the flag at equality and leaves it high through landing.
    self.vel_command_b[:, 2] = torch.maximum(
      self.vel_command_b[:, 2],
      (self._env.episode_length_buf >= self._takeoff_frame).float(),
    )

  # The stock velocity GUI creates an invalid max=0 slider for Spring Jump's
  # fixed Y and yaw components.  Reuse the source-compatible GUI that renders
  # zero-range axes as disabled controls.
  create_gui = RearStandVelocityCommand.create_gui


@dataclass(kw_only=True)
class SpringJumpCommandCfg(UniformVelocityCommandCfg):
  def build(self, env) -> SpringJumpCommand:
    return SpringJumpCommand(self, env)
