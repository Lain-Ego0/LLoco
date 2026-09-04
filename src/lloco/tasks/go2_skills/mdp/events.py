"""Go2 reset and perturbation events."""

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import requires_model_fields


def overwrite_root_velocity(
  env,
  env_ids,
  max_push_vel_xy: float,
  max_push_ang_vel: float,
) -> None:
  """Match the source push: overwrite XY and all angular velocity components."""
  ids = resolve_env_ids(env, env_ids)
  robot: Entity = env.scene["robot"]
  velocity = robot.data.root_link_vel_w[ids].clone()
  velocity[:, :2].uniform_(-max_push_vel_xy, max_push_vel_xy)
  velocity[:, 3:].uniform_(-max_push_ang_vel, max_push_ang_vel)
  robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)


@requires_model_fields("geom_friction")
def source_friction_buckets(
  env,
  env_ids,
  low: float,
  high: float,
  num_buckets: int,
  entity_name: str = "robot",
) -> None:
  """Match the source's 256-value friction bucket assignment."""
  ids = resolve_env_ids(env, env_ids)
  robot: Entity = env.scene[entity_name]
  buckets = torch.empty(num_buckets, device=env.device).uniform_(low, high)
  bucket_ids = torch.randint(num_buckets, (len(ids),), device=env.device)
  values = buckets[bucket_ids]
  geom_ids = robot.indexing.geom_ids
  env.sim.model.geom_friction[ids[:, None], geom_ids, 0] = values[:, None]


def reset_joints_by_scale(
  env,
  env_ids,
  scale_range: tuple[float, float],
  entity_name: str = "robot",
) -> None:
  """Reset positions to a uniform multiple of each source default angle."""
  ids = resolve_env_ids(env, env_ids)
  robot: Entity = env.scene[entity_name]
  default_pos = robot.data.default_joint_pos
  default_vel = robot.data.default_joint_vel
  limits = robot.data.soft_joint_pos_limits
  assert default_pos is not None
  assert default_vel is not None
  assert limits is not None
  scale = torch.empty_like(default_pos[ids]).uniform_(*scale_range)
  position = (default_pos[ids] * scale).clamp(limits[ids, :, 0], limits[ids, :, 1])
  robot.write_joint_state_to_sim(
    position, torch.zeros_like(default_vel[ids]), env_ids=ids
  )


def sample_restitution_label(
  env,
  env_ids,
  low: float,
  high: float,
) -> None:
  """Sample the source restitution label where MuJoCo has no direct coefficient."""
  ids = resolve_env_ids(env, env_ids)
  labels = getattr(env, "_rear_stand_restitution", None)
  if labels is None:
    labels = torch.zeros((env.num_envs, 1), device=env.device)
    env._rear_stand_restitution = labels
  labels[ids].uniform_(low, high)
