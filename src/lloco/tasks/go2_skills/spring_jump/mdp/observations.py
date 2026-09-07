"""Exact 47/65 source frames for the spring-jump task."""

import torch
from mjlab.entity import Entity
from mjlab.managers import ObservationTermCfg

from ...jump.mdp.observations import (
  _SourceHistory,
  contact_observation,
  joint_ids,
  root_euler,
)


def _actor_frame(env, command_name: str) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = joint_ids(robot)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.cat((
    torch.zeros((env.num_envs, 2), device=env.device), command,
    robot.data.root_link_ang_vel_b * 0.25, root_euler(robot),
    robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids],
    robot.data.joint_vel[:, ids] * 0.05, env.action_manager.action,
  ), dim=1)


def _critic_frame(env, command_name: str, sensor_name: str) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = joint_ids(robot)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  state = _state(env)
  # The source comment says two contacts, but concatenates all four feet.
  return torch.cat((
    command,
    robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids],
    robot.data.joint_pos[:, ids], robot.data.joint_vel[:, ids] * 0.05,
    env.action_manager.action, robot.data.root_link_lin_vel_b * 2.0,
    robot.data.root_link_ang_vel_b * 0.25, root_euler(robot),
    contact_observation(env, sensor_name), state.has_jumped.float().unsqueeze(1),
  ), dim=1)


class SpringActorHistory(_SourceHistory):
  frame_dim = 47
  history_length = 10

  def __call__(self, env, command_name: str, add_noise: bool) -> torch.Tensor:
    frame = _actor_frame(env, command_name)
    if add_noise:
      amplitude = torch.tensor((0.0,) * 5 + (0.05,) * 3 + (0.1,) * 3 + (0.01,) * 12 + (0.075,) * 12 + (0.0,) * 12, device=env.device)
      frame = frame + (2.0 * torch.rand_like(frame) - 1.0) * amplitude
    return self._append(frame)


class SpringCriticHistory(_SourceHistory):
  frame_dim = 65
  history_length = 3

  def __call__(self, env, command_name: str, sensor_name: str) -> torch.Tensor:
    return self._append(_critic_frame(env, command_name, sensor_name))


def _state(env):
  state = getattr(env, "_spring_jump_state", None)
  if state is None:
    from .rewards import SpringJumpState
    state = SpringJumpState(env)
    setattr(env, "_spring_jump_state", state)
  return state
