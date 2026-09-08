"""Command sampling and curriculum matching the Gym Go2 CTS task."""

from dataclasses import dataclass

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg


class CtsVelocityCommand(UniformVelocityCommand):
  """Gym CTS command sampler (including its heading-command convention)."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = True
    else:
      self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    moving = torch.linalg.vector_norm(self.vel_command_b[env_ids, :2], dim=1) > 0.1
    self.vel_command_b[env_ids, :2] *= moving.unsqueeze(1)
    self.is_standing_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids
    if self.cfg.heading_command:
      error = self.heading_target - self.robot.data.heading_w
      error = torch.atan2(torch.sin(error), torch.cos(error))
      self.vel_command_b[:, 2] = torch.clamp(0.5 * error, -2.0, 2.0)


@dataclass(kw_only=True)
class CtsVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env) -> CtsVelocityCommand:
    return CtsVelocityCommand(self, env)


def cts_command_curriculum(env, env_ids, command_name: str, max_curriculum: float):
  """Gym CTS range expansion based on tracking reward at reset."""
  max_steps = round(env.max_episode_length_s / env.step_dt)
  if env.common_step_counter and env.common_step_counter % max_steps == 0:
    sums = env.reward_manager._episode_sums["tracking_lin_vel"]  # noqa: SLF001
    weight = env.reward_manager.get_term_cfg("tracking_lin_vel").weight * env.step_dt
    if torch.mean(sums[env_ids]) / max_steps > 0.8 * weight:
      term = env.command_manager.get_term(command_name)
      lo, hi = term.cfg.ranges.lin_vel_x
      term.cfg.ranges.lin_vel_x = (
        max(lo - 0.2, -max_curriculum),
        min(hi + 0.2, max_curriculum),
      )
  term = env.command_manager.get_term(command_name)
  return {"lin_vel_x_max": torch.tensor(term.cfg.ranges.lin_vel_x[1], device=env.device)}
