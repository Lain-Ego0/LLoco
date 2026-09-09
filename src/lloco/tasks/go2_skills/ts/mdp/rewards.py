"""TS reward terms whose reduction differs from the DreamWaQ bridge."""

import torch


def base_height(env, target_height: float) -> torch.Tensor:
  """Gym averages all 187 measured heights, rather than a central crop."""
  scan = env.scene["terrain_scan"].data
  terrain_z = scan.hit_pos_w[..., 2].mean(1)
  return torch.square(
    env.scene["robot"].data.root_link_pos_w[:, 2] - terrain_z - target_height
  )


def collision(env, sensor_names: tuple[str, ...]) -> torch.Tensor:
  """Count contacts on the Gym TS penalized set: thigh, calf and base."""
  count = torch.zeros(env.num_envs, device=env.device)
  for name in sensor_names:
    force = env.scene[name].data.force
    assert force is not None
    count = count + (torch.linalg.vector_norm(force, dim=-1) > 0.1).float().sum(1)
  return count
