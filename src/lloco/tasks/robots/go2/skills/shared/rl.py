"""Shared RL runner construction and Go2 reward primitives."""

import torch
from mjlab.entity import Entity
from mjlab.managers import RewardTermCfg

from lloco.tasks.rl import make_ppo_runner_cfg as _make_ppo_runner_cfg

from .contacts import joint_ids, source_contact


def make_ppo_runner_cfg(*args, **kwargs):
  return _make_ppo_runner_cfg(*args, **kwargs)


def command(env, command_name: str) -> torch.Tensor:
  value = env.command_manager.get_command(command_name)
  assert value is not None
  return value


def moving(env, command_name: str) -> torch.Tensor:
  return torch.linalg.vector_norm(command(env, command_name)[:, :3], dim=1) > 0.1


def lin_vel_z_squared(env) -> torch.Tensor:
  return torch.square(env.scene["robot"].data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_squared(env) -> torch.Tensor:
  return torch.square(env.scene["robot"].data.root_link_ang_vel_b[:, :2]).sum(dim=1)


def orientation_squared(env) -> torch.Tensor:
  return torch.square(env.scene["robot"].data.projected_gravity_b[:, :2]).sum(dim=1)


def torques_squared(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.square(robot.data.qfrc_actuator[:, joint_ids(robot)]).sum(dim=1)


def absolute_torques(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return torch.abs(robot.data.qfrc_actuator[:, joint_ids(robot)]).sum(dim=1)


class DofAcceleration:
  def __init__(self, cfg: RewardTermCfg, env) -> None:
    del cfg
    self._last_velocity = torch.zeros(
      env.num_envs, len(joint_ids(env.scene["robot"])), device=env.device
    )

  def __call__(self, env) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    velocity = robot.data.joint_vel[:, joint_ids(robot)]
    value = torch.square((self._last_velocity - velocity) / env.step_dt).sum(dim=1)
    self._last_velocity.copy_(velocity)
    return value

  def reset(self, env_ids=None) -> None:
    self._last_velocity[env_ids] = 0.0


def collision(env, sensor_name: str) -> torch.Tensor:
  force = env.scene[sensor_name].data.force
  assert force is not None
  return (torch.linalg.vector_norm(force, dim=-1) > 0.1).float().sum(dim=1)


def action_rate(env) -> torch.Tensor:
  return torch.square(env.action_manager.prev_action - env.action_manager.action).sum(1)


def stand_still(env, command_name: str) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = joint_ids(robot)
  return torch.abs(
    robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
  ).sum(dim=1) * ~moving(env, command_name)


def default_pos(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids = joint_ids(robot)
  return torch.abs(
    robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
  ).sum(dim=1)


def contact_without_command(env, sensor_name: str, command_name: str) -> torch.Tensor:
  return (source_contact(env.scene[sensor_name], 0.1).sum(dim=1) == 4) * ~moving(
    env, command_name
  )


def terminal_cost(env) -> torch.Tensor:
  return env.termination_manager.terminated.float() / env.step_dt
