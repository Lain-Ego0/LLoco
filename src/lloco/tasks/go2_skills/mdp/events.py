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
