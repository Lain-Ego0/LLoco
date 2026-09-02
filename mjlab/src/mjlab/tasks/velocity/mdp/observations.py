from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def go2_source_trot_observation(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Build the 47-D single-frame observation used by the source Go2 trot task.

  The source Isaac Gym task concatenates phase (sin/cos), scaled commands, IMU
  angular velocity and Euler angles, relative joint position/velocity, and the
  previous policy action.  History is intentionally configured on the
  ``ObservationTermCfg`` so mjlab stores the frames in chronological order.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  phase = (
    env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
  ) % 0.5 / 0.5
  phase_features = torch.stack((torch.sin(2.0 * torch.pi * phase),
                                torch.cos(2.0 * torch.pi * phase)), dim=1)

  command_scale = torch.tensor((2.0, 2.0, 0.25), device=env.device)
  command_features = command[:, :3] * command_scale
  imu = torch.cat(
    (
      asset.data.root_link_ang_vel_b * 0.25,
      euler_xyz_from_quat(asset.data.root_link_quat_w),
    ),
    dim=-1,
  )
  joint_pos = asset.data.joint_pos - asset.data.default_joint_pos
  joint_vel = asset.data.joint_vel * 0.05
  actions = env.action_manager.action
  return torch.cat(
    (phase_features, command_features, imu, joint_pos, joint_vel, actions),
    dim=-1,
  )


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
