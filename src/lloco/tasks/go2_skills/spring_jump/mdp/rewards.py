"""Stateful reward equations translated from ``Go2_Spring_Jump.py``."""

import torch
from mjlab.entity import Entity
from mjlab.managers import RewardTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from ...shared.contacts import joint_ids, source_vertical_contact
from .observations import _state, root_euler


class SpringJumpState:
  def __init__(self, env) -> None:
    self.was_in_flight = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.has_jumped = torch.zeros_like(self.was_in_flight)
    self.last_contacts = torch.zeros((env.num_envs, 4), dtype=torch.bool, device=env.device)
    self.landing_pos = torch.zeros((env.num_envs, 2), device=env.device)
    self.start_pos = torch.zeros_like(self.landing_pos)
    self.max_height = torch.zeros(env.num_envs, device=env.device)
    self.pushed_up = torch.zeros_like(self.was_in_flight)
    self.steps = 0
    self.last_step = -1

  def update(self, env, command_name: str, sensor_name: str) -> None:
    step = int(env.common_step_counter)
    if step == self.last_step:
      return
    self.last_step = step
    self.steps += 1
    robot: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    contact = source_vertical_contact(env.scene[sensor_name], 1.0)
    filtered = contact | self.last_contacts
    self.last_contacts.copy_(contact)
    flight = ~filtered.any(dim=1) & (command[:, 2] > 0)
    self.was_in_flight |= flight
    landed = filtered.any(dim=1) & self.was_in_flight
    new_landing = landed & ~self.has_jumped
    self.landing_pos[new_landing] = robot.data.root_link_pos_w[new_landing, :2]
    self.has_jumped |= landed
    self.max_height = torch.maximum(self.max_height, robot.data.root_link_pos_w[:, 2])
    # Source's training-only helper: a one-time upward root-velocity overwrite
    # at take-off with p=max(8-floor(step/1200),0)/10.
    should_push = (~self.pushed_up) & (~self.has_jumped) & (command[:, 2] > 0)
    probability = max(8 - self.steps // 1200, 0) / 10.0
    ids = torch.nonzero(should_push & (torch.rand(env.num_envs, device=env.device) < probability)).squeeze(1)
    if len(ids):
      velocity = robot.data.root_link_vel_w[ids].clone()
      velocity[:, 2].uniform_(1.5, 2.2)
      robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
    self.pushed_up |= should_push

  def reset(self, env, env_ids=None) -> None:
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    robot: Entity = env.scene["robot"]
    self.was_in_flight[env_ids] = False
    self.has_jumped[env_ids] = False
    self.last_contacts[env_ids] = False
    self.landing_pos[env_ids] = robot.data.root_link_pos_w[env_ids, :2]
    self.start_pos[env_ids] = robot.data.root_link_pos_w[env_ids, :2]
    self.max_height[env_ids] = 0.0
    self.pushed_up[env_ids] = False


class _SpringReward:
  def __init__(self, cfg: RewardTermCfg, env) -> None:
    del cfg
    self.state = _state(env)

  def reset(self, env_ids=None) -> None:
    self.state.reset(self.env, env_ids)


class BeforeSetting(_SpringReward):
  def __init__(self, cfg, env) -> None:
    super().__init__(cfg, env); self.env = env
  def __call__(self, env, command_name: str, sensor_name: str) -> torch.Tensor:
    self.state.update(env, command_name, sensor_name)
    robot = env.scene["robot"]; ids = joint_ids(robot); cmd = env.command_manager.get_command(command_name)
    return torch.exp(-torch.abs(robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]).sum(1) / 2.0) * (cmd[:, 2] == 0)


def line_z(env, command_name, sensor_name):
  s = _state(env); s.update(env, command_name, sensor_name); r = env.scene["robot"]; c = env.command_manager.get_command(command_name)
  return (r.data.root_link_vel_w[:, 2] > 0) * r.data.root_link_vel_w[:, 2] * ~s.has_jumped * (c[:, 2] == 1)
def flight(env, command_name, sensor_name):
  s = _state(env); s.update(env, command_name, sensor_name); return s.was_in_flight.float()
def base_height_flight(env, command_name, sensor_name):
  s = _state(env); s.update(env, command_name, sensor_name); z = env.scene["robot"].data.root_link_pos_w[:, 2]
  return torch.exp(-torch.abs(z - .47) * 5) * s.was_in_flight * ~s.has_jumped * 6
def base_height_stance(env, command_name, sensor_name):
  s = _state(env); s.update(env, command_name, sensor_name); z = env.scene["robot"].data.root_link_pos_w[:, 2]
  return torch.exp(-torch.abs(z - .35) * 5) * s.has_jumped * (z > .2)
def land_pos(env, command_name, sensor_name):
  s = _state(env); s.update(env, command_name, sensor_name); c = env.command_manager.get_command(command_name); r = env.scene["robot"]
  return torch.exp(-torch.abs(s.start_pos + c[:, :2] - s.landing_pos).sum(1)) * s.has_jumped * (root_euler(r).sum(1) < .6) * (s.max_height > .42)
def dof_pos(env):
  r = env.scene["robot"]; ids = joint_ids(r); return torch.abs(r.data.joint_pos[:, ids] - r.data.default_joint_pos[:, ids]).sum(1)
def hip_pos(env):
  r = env.scene["robot"]; ids = joint_ids(r); return torch.abs(r.data.joint_pos[:, ids] - r.data.default_joint_pos[:, ids])[:, (0,3,6,9)].sum(1)
def orientation(env): return torch.exp(-torch.abs(root_euler(env.scene["robot"])).sum(1))
def ang_vel_xy(env): return torch.abs(env.scene["robot"].data.root_link_ang_vel_b).sum(1)
def torques(env): return torch.abs(env.scene["robot"].data.qfrc_actuator[:, joint_ids(env.scene["robot"])]).sum(1)
def action_rate(env): return torch.square(env.action_manager.action - env.action_manager.prev_action).sum(1)
def collision(env, sensor_name): return (torch.linalg.vector_norm(env.scene[sensor_name].data.force, dim=-1) > .1).sum(1)
def dof_vel(env): return torch.square(env.scene["robot"].data.joint_vel[:, joint_ids(env.scene["robot"])]).sum(1)
def tracking_lin_vel(env, command_name, sensor_name):
  s = _state(env); s.update(env, command_name, sensor_name); c = env.command_manager.get_command(command_name); v = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
  return torch.exp(-torch.square(c[:, 0] * 1.6 - v)) * s.was_in_flight * ~s.has_jumped * 5
def line_vel_stance(env, command_name, sensor_name):
  s = _state(env); s.update(env, command_name, sensor_name); return torch.abs(env.scene["robot"].data.root_link_lin_vel_b[:, :2]).sum(1) * s.has_jumped
def dof_pos_limits(env):
  r = env.scene["robot"]; ids = joint_ids(r); limits = r.data.soft_joint_pos_limits
  assert limits is not None
  pos = r.data.joint_pos[:, ids]
  return (-(pos - limits[:, ids, 0]).clamp(max=0.0) + (pos - limits[:, ids, 1]).clamp(min=0.0)).sum(1)
def feet_contact_forces(env, sensor_name, max_contact_force: float):
  force = env.scene[sensor_name].data.force
  assert force is not None
  return (torch.linalg.vector_norm(force, dim=-1) - max_contact_force).clamp(min=0.0).sum(1)
def foot_clearance(env, command_name, sensor_name):
  s = _state(env); s.update(env, command_name, sensor_name); r = env.scene["robot"]
  ids, _ = r.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)
  root_to_foot = r.data.site_pos_w[:, ids] - r.data.root_link_pos_w[:, None, :]
  quat = r.data.root_link_quat_w[:, None, :].expand(-1, len(ids), -1)
  height = quat_apply_inverse(quat.reshape(-1, 4), root_to_foot.reshape(-1, 3)).reshape(-1, len(ids), 3)[..., 2]
  return torch.abs(height + .20).sum(1) * s.was_in_flight * ~s.has_jumped * 6
