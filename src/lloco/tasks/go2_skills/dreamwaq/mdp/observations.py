"""Field-for-field DreamWaQ observations and source-style histories."""

import torch
from mjlab.entity import Entity
from mjlab.managers import ObservationTermCfg

from ...shared.contacts import JOINT_NAMES


def _joint_ids(robot: Entity) -> list[int]:
  ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
  if tuple(names) != JOINT_NAMES:
    raise RuntimeError(f"Go2 joint order mismatch: {names}")
  return ids


def _command(env, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command[:, :3]


def _actor_frame(env, command_name: str) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = _joint_ids(robot)
  return torch.cat((
    _command(env, command_name) * torch.tensor((2.0, 2.0, 0.25), device=env.device),
    robot.data.root_link_ang_vel_b * 0.25,
    robot.data.projected_gravity_b,
    robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids],
    robot.data.joint_vel[:, ids] * 0.05,
    env.action_manager.action,
  ), dim=1)


def _noise(frame: torch.Tensor, joint_position_noise: float = 0.02) -> torch.Tensor:
  # Source noise vector: cmd/action 0; ang vel .2*.25; gravity .05;
  # q is task-specific (.02 for DreamWaQ, .01 for AMP); dq 1.5*.05.
  amplitude = torch.tensor(
    [0.0] * 3 + [0.05] * 3 + [0.05] * 3 + [joint_position_noise] * 12
    + [0.075] * 12 + [0.0] * 12,
    device=frame.device,
    dtype=frame.dtype,
  )
  return frame + (2.0 * torch.rand_like(frame) - 1.0) * amplitude


class DreamActor:
  """Current 45-field actor input, while retaining it for the delayed VAE input."""

  def __init__(self, cfg: ObservationTermCfg, env) -> None:
    del cfg
    self._current = torch.zeros((env.num_envs, 45), device=env.device)
    # Observation terms are constructed before ObservationManager is attached.
    # Share the source's previous actor frame through the environment instead.
    env._dreamwaq_actor_term = self

  def __call__(
    self,
    env,
    command_name: str,
    add_noise: bool,
    joint_position_noise: float = 0.02,
  ) -> torch.Tensor:
    frame = _actor_frame(env, command_name)
    self._current.copy_(
      _noise(frame, joint_position_noise) if add_noise else frame
    )
    return self._current

  def reset(self, env_ids=None) -> None:
    self._current[env_ids] = 0.0


class DreamActorHistory:
  """Five source frames immediately preceding the actor frame, oldest first."""

  def __init__(self, cfg: ObservationTermCfg, env) -> None:
    del cfg
    self._history = torch.zeros((env.num_envs, 5, 45), device=env.device)

  def __call__(self, env) -> torch.Tensor:
    # The Gym code shifts ``obs_hist_buf`` before constructing this step's
    # ``obs_buf``; use the actor term from the prior manager evaluation.
    output = self._history.reshape(env.num_envs, -1).clone()
    actor = env._dreamwaq_actor_term
    assert isinstance(actor, DreamActor)
    self._history = torch.roll(self._history, shifts=-1, dims=1)
    self._history[:, -1] = actor._current
    return output

  def reset(self, env_ids=None) -> None:
    self._history[env_ids] = 0.0


class DreamCriticHistory:
  """Three 261-field privileged frames, matching source frame-major order."""

  def __init__(self, cfg: ObservationTermCfg, env) -> None:
    del cfg
    self._history = torch.zeros((env.num_envs, 3, 261), device=env.device)

  def __call__(self, env, command_name: str, sensor_name: str) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    _joint_ids(robot)
    scan = env.scene[sensor_name].data
    # ``height_scan`` is base Z minus terrain height.  The Gym value is
    # clip(base_z - .5 - terrain_height, -1, 1) * 5.
    heights = torch.clamp(scan.frame_pos_w[:, 0, 2:3] - 0.5 - scan.hit_pos_w[..., 2], -1.0, 1.0) * 5.0
    friction = robot.data.model.geom_friction[:, robot.indexing.geom_ids[0], 0:1]
    restitution = getattr(env, "_dreamwaq_restitution", torch.zeros_like(friction))
    p_gain = getattr(env, "_dreamwaq_p_gain", torch.ones((env.num_envs, 12), device=env.device))
    d_gain = getattr(env, "_dreamwaq_d_gain", torch.ones((env.num_envs, 12), device=env.device))
    frame = torch.cat((
      heights,
      robot.data.root_link_lin_vel_b * 2.0,
      friction,
      restitution,
      p_gain,
      d_gain,
      _actor_frame(env, command_name),
    ), dim=1)
    assert frame.shape[1] == 261, frame.shape
    self._history = torch.roll(self._history, shifts=-1, dims=1)
    self._history[:, -1] = frame
    return self._history.reshape(env.num_envs, -1)

  def reset(self, env_ids=None) -> None:
    self._history[env_ids] = 0.0


def base_linear_velocity(env) -> torch.Tensor:
  return env.scene["robot"].data.root_link_lin_vel_b
