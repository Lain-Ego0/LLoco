"""Go2 curricula."""

import torch


def source_trot_command_curriculum(
  env, env_ids, command_name: str, max_curriculum: float
) -> dict[str, torch.Tensor]:
  """Expand X velocity exactly when the source's global episode check fires."""
  term = env.command_manager.get_term(command_name)
  cfg = term.cfg
  max_steps = round(env.max_episode_length_s / env.step_dt)
  if env.common_step_counter > 0 and env.common_step_counter % max_steps == 0:
    sums = env.reward_manager._episode_sums["tracking_lin_vel"]  # noqa: SLF001
    weight = env.reward_manager.get_term_cfg("tracking_lin_vel").weight
    if torch.mean(sums[env_ids]) / max_steps > 0.8 * weight * env.step_dt:
      low, high = cfg.ranges.lin_vel_x
      cfg.ranges.lin_vel_x = (
        max(low - 0.5, -max_curriculum),
        min(high + 0.5, max_curriculum),
      )
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
  }
