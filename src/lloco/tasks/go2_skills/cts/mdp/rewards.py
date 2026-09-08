"""CTS reward terms whose state/contact semantics differ from shared tasks."""

import torch
from mjlab.entity import Entity
from mjlab.managers import RewardTermCfg

from ...shared.contacts import JOINT_NAMES


def low_base_height_barrier(env, minimum_height: float) -> torch.Tensor:
  """Penalize only the MuJoCo folded-leg equilibrium below normal gait height."""
  scan = env.scene["terrain_scan"].data
  hits = scan.hit_pos_w[..., 2].reshape(env.num_envs, 17, 11)
  terrain_z = hits[:, 7:10, 4:7].mean((1, 2))
  height = env.scene["robot"].data.root_link_pos_w[:, 2] - terrain_z
  return torch.square(torch.clamp_min(minimum_height - height, 0.0))


def _joint_ids(robot: Entity) -> list[int]:
  ids, _ = robot.find_joints(JOINT_NAMES, preserve_order=True)
  return ids


class CtsDofAcceleration:
  """Source acceleration penalty with episode-local previous velocity."""

  def __init__(self, cfg: RewardTermCfg, env) -> None:
    del cfg
    self._last_velocity = torch.zeros(
      (env.num_envs, 12), device=env.device
    )

  def __call__(self, env) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    velocity = robot.data.joint_vel[:, _joint_ids(robot)]
    value = torch.square((self._last_velocity - velocity) / env.step_dt).sum(1)
    self._last_velocity.copy_(velocity)
    return value

  def reset(self, env_ids=None) -> None:
    self._last_velocity[env_ids] = 0.0


class CtsActionSmoothness:
  """Second action difference with Gym-compatible reset history."""

  def __init__(self, cfg: RewardTermCfg, env) -> None:
    del cfg
    self._before_previous = torch.zeros_like(env.action_manager.prev_action)

  def __call__(self, env) -> torch.Tensor:
    previous = env.action_manager.prev_action
    value = torch.square(
      env.action_manager.action - 2 * previous + self._before_previous
    ).sum(1)
    self._before_previous.copy_(previous)
    return value

  def reset(self, env_ids=None) -> None:
    self._before_previous[env_ids] = 0.0


def collision(env, sensor_names: tuple[str, ...]) -> torch.Tensor:
  """Count thigh, calf, and base contacts as configured by source CTS."""
  total = torch.zeros(env.num_envs, device=env.device)
  for sensor_name in sensor_names:
    force = env.scene[sensor_name].data.force
    assert force is not None
    total += (torch.linalg.vector_norm(force, dim=-1) > 0.1).float().sum(1)
  return total
