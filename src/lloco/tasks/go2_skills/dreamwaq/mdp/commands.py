"""DreamWaQ's source command sampler and command-range curriculum."""

from dataclasses import dataclass

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg


class DreamWaQVelocityCommand(UniformVelocityCommand):
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Gym chooses max X from the global PPO iteration (common steps / 24),
    # rather than from the configurable command curriculum.
    iteration = self._env.common_step_counter // 24
    max_x = 1.0 if iteration <= 45_000 else 1.2 if iteration < 60_000 else 1.4 if iteration < 70_000 else 1.5
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(-max_x, max_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    assert self.cfg.ranges.heading is not None
    self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
    self.is_heading_env[env_ids] = True
    self.is_standing_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False
    moving = torch.linalg.vector_norm(self.vel_command_b[env_ids, :2], dim=1) > .2
    self.vel_command_b[env_ids, :2] *= moving.unsqueeze(1)

  def _update_command(self, env_ids=None) -> None:
    del env_ids
    error = torch.atan2(torch.sin(self.heading_target - self.robot.data.heading_w), torch.cos(self.heading_target - self.robot.data.heading_w))
    self.vel_command_b[:, 2] = torch.clamp(.5 * error, -2.0, 2.0)
    # Source treats the first 20% as high-speed environments at resampling.
    high = torch.arange(self.num_envs, device=self.device) < int(self.num_envs * .2)
    self.vel_command_b[high, 1] *= torch.abs(self.vel_command_b[high, 0]) < 1.0


@dataclass(kw_only=True)
class DreamWaQVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env) -> DreamWaQVelocityCommand:
    return DreamWaQVelocityCommand(self, env)


def source_command_curriculum(env, env_ids, command_name: str, max_curriculum: float):
  max_steps = round(env.max_episode_length_s / env.step_dt)
  if env.common_step_counter and env.common_step_counter % max_steps == 0:
    reward = env.reward_manager._episode_sums["tracking_lin_vel"]  # noqa: SLF001
    weight = env.reward_manager.get_term_cfg("tracking_lin_vel").weight * env.step_dt
    high = env_ids[env_ids < int(env.num_envs * .2)]
    low = env_ids[env_ids > int(env.num_envs * .2)]
    if len(high) and len(low) and torch.mean(reward[high]) / max_steps > .8 * weight and torch.mean(reward[low]) / max_steps > .8 * weight:
      term = env.command_manager.get_term(command_name)
      lo, hi = term.cfg.ranges.lin_vel_x
      term.cfg.ranges.lin_vel_x = (max(lo - .2, -max_curriculum), min(hi + .2, max_curriculum))
  term = env.command_manager.get_term(command_name)
  return {"lin_vel_x_max": torch.tensor(term.cfg.ranges.lin_vel_x[1], device=env.device)}
