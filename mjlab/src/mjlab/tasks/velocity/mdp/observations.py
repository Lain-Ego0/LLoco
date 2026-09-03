from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.raycast_sensor import RayCastSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _go2_contact_mask(sensor: ContactSensor, threshold: float) -> torch.Tensor:
  """Return source-style vertical-force contact bits for a feet sensor."""
  force = sensor.data.force
  if force is not None:
    return (force[..., 2] > threshold).to(torch.float32)
  found = sensor.data.found
  assert found is not None
  return (found > 0).to(torch.float32)


def _go2_source_contact_mask(sensor: ContactSensor, threshold: float) -> torch.Tensor:
  """Return four-foot contacts in source ``FL, FR, RL, RR`` order."""
  contact = _go2_contact_mask(sensor, threshold)
  if contact.shape[-1] != 4:
    raise ValueError(f"Expected four foot contacts, got shape {tuple(contact.shape)}")
  # Shared mjlab sensor order is FR, FL, RR, RL; source buffers use FL, FR, RL, RR.
  return contact[:, (1, 0, 3, 2)]


def _go2_delayed_policy_state(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
  """Read optional source-style delayed motor and IMU observations.

  The buffers live on the Go2 action term and are updated after every MuJoCo
  substep.  Keeping this helper optional preserves ordinary mjlab behavior for
  tasks that do not enable latency randomization.
  """
  try:
    action_term = env.action_manager.get_term("joint_pos")
    motor = action_term.get_delayed_motor_observation()
    imu = action_term.get_delayed_imu_observation()
  except (AttributeError, KeyError, RuntimeError):
    return None
  if motor is None or imu is None:
    return None
  return motor[0], motor[1], imu


def go2_cts_teacher_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return the source CTS teacher/student split for each environment.

  The legacy runner reserves three environments for the privileged teacher and
  every fourth environment for the history-only student.  Keeping this as an
  observation group makes the split part of the rollout (and therefore stable
  during PPO minibatch shuffling) without changing the actor input dimension.
  A non-multiple-of-four environment count is accepted for small smoke runs;
  it simply follows the same repeating pattern.
  """
  env_ids = torch.arange(env.num_envs, device=env.device)
  return (env_ids.remainder(4) != 0).to(dtype=torch.float32).unsqueeze(-1)


def _go2_terrain_heights(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: RayCastSensor = env.scene[sensor_name]
  data = sensor.data
  frame_count, ray_count = sensor.num_frames, sensor.num_rays_per_frame
  batch = data.distances.shape[0]
  frame_z = data.frame_pos_w[:, :, 2:3]
  hit_z = data.hit_pos_w[..., 2].view(batch, frame_count, ray_count)
  heights = (frame_z - hit_z).view(batch, frame_count * ray_count)
  heights = torch.where(
    data.distances < 0,
    torch.full_like(heights, sensor.cfg.max_distance),
    heights,
  )
  # Isaac Gym's source privileged observations use
  # clip(base_height - measured_height - 0.5, -1, 1) * 5.  With a raycast
  # frame attached to base_link, ``heights`` is already base_height minus the
  # hit height, so apply the same offset/scale here.
  return (heights - 0.5).clamp(-1.0, 1.0) * 5.0


def go2_source_terrain_heights(
  env: ManagerBasedRlEnv,
  sensor_name: str = "terrain_scan",
) -> torch.Tensor:
  """Expose the source TS teacher's clipped and scaled 187-D height scan."""
  return _go2_terrain_heights(env, sensor_name)


def _go2_domain_randomization_fields(
  env: ManagerBasedRlEnv,
  *,
  include_mass_com: bool,
) -> torch.Tensor:
  """Expose the MuJoCo fields randomized by the Go2 source configurations."""
  asset: Entity = env.scene["robot"]
  size = 42 if include_mass_com else 26
  zeros = torch.zeros((env.num_envs, size), device=env.device)
  try:
    model = env.sim.model
    default_gain = env.sim.get_default_field("actuator_gainprm")
    default_bias = env.sim.get_default_field("actuator_biasprm")
    ctrl_ids = asset.indexing.ctrl_ids
    kp = getattr(env, "_go2_pd_kp_multiplier", None)
    kd = getattr(env, "_go2_pd_kd_multiplier", None)
    if kp is None or kd is None:
      kp = model.actuator_gainprm[:, ctrl_ids, 0] / default_gain[ctrl_ids, 0].clamp_min(1e-6)
      kd = (-model.actuator_biasprm[:, ctrl_ids, 2]) / (-default_bias[ctrl_ids, 2]).clamp_min(1e-6)
    if kp.shape[-1] != 12 or kd.shape[-1] != 12:
      return zeros
    # The source buffers store the sampled coefficient itself (not a ratio to
    # the XML default), so keep the MuJoCo value in the same units.
    friction = model.geom_friction[:, asset.indexing.geom_ids, 0].mean(dim=-1, keepdim=True)
    restitution = torch.zeros_like(friction)
    if include_mass_com:
      body_id = asset.indexing.body_ids[0]
      default_mass = env.sim.get_default_field("body_mass")[body_id].clamp_min(1e-6)
      # ``added_base_masses`` in the source is an additive kilogram offset.
      mass = model.body_mass[:, body_id:body_id + 1] - default_mass
      default_ipos = env.sim.get_default_field("body_ipos")[body_id]
      com = model.body_ipos[:, body_id] - default_ipos
      torque = getattr(env, "_go2_torque_multiplier", torch.ones_like(kp))
      return torch.cat((friction, restitution, mass, com, kp, kd, torque), dim=-1)
    return torch.cat((friction, restitution, kp, kd), dim=-1)
  except (AttributeError, IndexError, RuntimeError, TypeError):
    return zeros


def _go2_stand_randomization_fields(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Build the 34-D stand-task randomization block from MuJoCo model data."""
  asset: Entity = env.scene["robot"]
  zeros = torch.zeros((env.num_envs, 34), device=env.device)
  try:
    model = env.sim.model
    default = env.sim.get_default_field
    body_id = asset.indexing.body_ids[0]
    default_mass = default("body_mass")[body_id]
    mass = model.body_mass[:, body_id:body_id + 1] - default_mass
    com = model.body_ipos[:, body_id] - default("body_ipos")[body_id]
    default_gain = default("actuator_gainprm")
    default_bias = default("actuator_biasprm")
    ctrl_ids = asset.indexing.ctrl_ids
    kp = model.actuator_gainprm[:, ctrl_ids, 0] / default_gain[ctrl_ids, 0].clamp_min(1e-6)
    kd = (-model.actuator_biasprm[:, ctrl_ids, 2]) / (-default_bias[ctrl_ids, 2]).clamp_min(1e-6)
    dof_ids = asset.indexing.joint_v_adr
    armature = model.dof_armature[:, dof_ids].mean(dim=-1, keepdim=True)
    friction_loss = model.dof_frictionloss[:, dof_ids].mean(dim=-1, keepdim=True)
    damping = model.dof_damping[:, dof_ids].mean(dim=-1, keepdim=True)
    geom = model.geom_friction[:, asset.indexing.geom_ids, 0].mean(dim=-1, keepdim=True)
    restitution = torch.zeros_like(geom)
    if kp.shape[-1] != 12 or kd.shape[-1] != 12:
      return zeros
    return torch.cat((geom, mass, com, kp, kd, armature, friction_loss, damping,
                      restitution, restitution), dim=-1)
  except (AttributeError, IndexError, RuntimeError, TypeError):
    return zeros


def go2_source_trot_observation(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  cycle_time: float = 0.5,
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
  ) % cycle_time / cycle_time
  phase_features = torch.stack((torch.sin(2.0 * torch.pi * phase),
                                torch.cos(2.0 * torch.pi * phase)), dim=1)

  command_scale = torch.tensor((2.0, 2.0, 0.25), device=env.device)
  command_features = command[:, :3] * command_scale
  roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
  euler_xyz = torch.stack((roll, pitch, yaw), dim=-1)
  delayed = _go2_delayed_policy_state(env)
  if delayed is None:
    imu = torch.cat((asset.data.root_link_ang_vel_b * 0.25, euler_xyz), dim=-1)
    joint_pos = asset.data.joint_pos - asset.data.default_joint_pos
    joint_vel = asset.data.joint_vel * 0.05
  else:
    joint_pos, joint_vel, imu = delayed
  actions = env.action_manager.action
  return torch.cat(
    (phase_features, command_features, imu, joint_pos, joint_vel, actions),
    dim=-1,
  )


def go2_source_trot_privileged_observation(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  cycle_time: float = 0.5,
  stance_threshold: float = 0.5,
) -> torch.Tensor:
  """Build the 68-D privileged frame used by the source Go2 trot critic."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  phase = (
    env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
  ) % cycle_time / cycle_time
  phase_features = torch.stack(
    (torch.sin(2.0 * torch.pi * phase), torch.cos(2.0 * torch.pi * phase)),
    dim=1,
  )
  command_input = torch.cat(
    (phase_features, command[:, :3] * torch.tensor((2.0, 2.0, 0.25), device=env.device)),
    dim=1,
  )
  q_rel = asset.data.joint_pos - asset.data.default_joint_pos
  q = asset.data.joint_pos
  dq = asset.data.joint_vel * 0.05
  roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
  euler_xyz = torch.stack((roll, pitch, yaw), dim=-1)
  stance_mask = torch.stack((phase < stance_threshold, phase > stance_threshold), dim=1).to(torch.float32)
  sensor: ContactSensor = env.scene[sensor_name]
  contact = _go2_source_contact_mask(sensor, threshold=5.0)
  return torch.cat(
    (
      command_input,
      q_rel,
      q,
      dq,
      env.action_manager.action,
      # Source ``obs_scales.lin_vel`` is 2.0 for the privileged critic frame;
      # actor IMU velocity remains represented only through the 0.25 angular
      # scale above.
      asset.data.root_link_lin_vel_b * 2.0,
      asset.data.root_link_ang_vel_b * 0.25,
      euler_xyz,
      stance_mask,
      contact,
    ),
    dim=-1,
  )


def go2_source_jump_privileged_observation(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Build the source jump critic frame (68-D state plus two randomization fields)."""
  base = go2_source_trot_privileged_observation(
    env,
    command_name=command_name,
    sensor_name=sensor_name,
    asset_cfg=asset_cfg,
    cycle_time=1.5,
    # The source ``_get_gait_phase`` splits the cycle at one half for Jump;
    # the 1.5 s cycle length does not change that threshold.
    stance_threshold=0.5,
  )
  # The Isaac Gym jump critic appends the sampled friction coefficient and the
  # base mass normalized by 10.  Read the corresponding MuJoCo batch fields;
  # retain a finite zero fallback for minimal test doubles without model data.
  friction = torch.zeros((base.shape[0], 1), device=base.device, dtype=base.dtype)
  mass = torch.zeros_like(friction)
  try:
    asset: Entity = env.scene[asset_cfg.name]
    model = env.sim.model
    friction = model.geom_friction[:, asset.indexing.geom_ids, 0].mean(dim=-1, keepdim=True)
    body_id = asset.indexing.body_ids[0]
    mass = model.body_mass[:, body_id : body_id + 1] / 10.0
    friction = friction.to(dtype=base.dtype)
    mass = mass.to(dtype=base.dtype)
  except (AttributeError, IndexError, RuntimeError, TypeError):
    pass
  extras = torch.cat((friction, mass), dim=-1)
  return torch.cat((base, extras), dim=-1)


def go2_source_backflip_observation(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Build the 47-D backflip frame (zero phase channels, as in the source)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  zeros = torch.zeros((command.shape[0], 2), device=command.device, dtype=command.dtype)
  # The source BackFlip observer uses projected gravity (not Euler angles) in
  # its six-dimensional IMU block.  Keep this distinct from Spring-Jump/Trot,
  # whose source observers use Euler XYZ.
  projected_gravity = asset.data.projected_gravity_b
  delayed = _go2_delayed_policy_state(env)
  if delayed is None:
    imu = torch.cat((asset.data.root_link_ang_vel_b * 0.25, projected_gravity), dim=-1)
    q_rel = asset.data.joint_pos - asset.data.default_joint_pos
    joint_vel = asset.data.joint_vel * 0.05
  else:
    q_rel, joint_vel, imu = delayed
  return torch.cat((zeros, command[:, :3], imu, q_rel, joint_vel,
                    env.action_manager.action), dim=-1)


def go2_source_backflip_privileged_observation(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Build the 50-D backflip privileged frame."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  zeros = torch.zeros((command.shape[0], 2), device=command.device, dtype=command.dtype)
  command_input = torch.cat((zeros, command[:, :3]), dim=-1)
  q_rel = asset.data.joint_pos - asset.data.default_joint_pos
  projected_gravity = asset.data.projected_gravity_b
  return torch.cat((command_input, q_rel, asset.data.joint_vel * 0.05,
                    env.action_manager.action, asset.data.root_link_lin_vel_b * 2.0,
                    asset.data.root_link_ang_vel_b * 0.25, projected_gravity), dim=-1)


def go2_source_stand_observation(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_first: bool = False,
) -> torch.Tensor:
  """Build a 45-D blind observation.

  The handstand/leggedstand source puts IMU and gravity before the command,
  while CTS/DreamWaQ/TS put the command first.  The switch keeps both legacy
  contracts explicit without duplicating the state assembly.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  projected_gravity = asset.data.projected_gravity_b
  command_obs = command[:, :3] * torch.tensor((2.0, 2.0, 0.25), device=env.device)
  delayed = _go2_delayed_policy_state(env)
  if delayed is None:
    imu_obs = asset.data.root_link_ang_vel_b * 0.25
    joint_pos = asset.data.joint_pos - asset.data.default_joint_pos
    joint_vel = asset.data.joint_vel * 0.05
  else:
    joint_pos, joint_vel, delayed_imu = delayed
    imu_obs = delayed_imu[:, :3]
  state = (command_obs, imu_obs, projected_gravity) if command_first else (imu_obs, projected_gravity, command_obs)
  return torch.cat((*state, joint_pos, joint_vel, env.action_manager.action), dim=-1)


def go2_source_stand_privileged_observation(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Build the 86-D stand critic frame, reserving randomization fields."""
  asset: Entity = env.scene[asset_cfg.name]
  actor = go2_source_stand_observation(env, command_name, asset_cfg)
  sensor: ContactSensor = env.scene[sensor_name]
  # Source stores 34 domain-randomization scalars followed by four contacts.
  reserved = _go2_stand_randomization_fields(env).to(actor.dtype)
  contact = _go2_source_contact_mask(sensor, threshold=1.0).to(actor.dtype)
  return torch.cat((asset.data.root_link_lin_vel_b, actor, reserved, contact), dim=-1)


def go2_amp_observation(
  env: ManagerBasedRlEnv,
  sensor_name: str = "terrain_scan",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Build the 31-D AMP discriminator input used by the source project.

  AMP deliberately uses unscaled absolute joint positions and the terrain-
  relative base height, unlike the policy's 45-D normalized observation.
  """
  asset: Entity = env.scene[asset_cfg.name]
  # Source ``_get_base_heights`` averages root-height minus nearby terrain
  # samples.  Recover the equivalent scalar from the raycast heights when the
  # sensor is available; flat scenes naturally reduce to the root height.
  try:
    terrain = _go2_terrain_heights(env, sensor_name)
    base_height = (terrain / 5.0 + 0.5).mean(dim=-1, keepdim=True)
  except (KeyError, AttributeError, RuntimeError, TypeError):
    base_height = asset.data.root_link_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]
  return torch.cat(
    (
      asset.data.joint_pos,
      asset.data.root_link_lin_vel_b,
      asset.data.root_link_ang_vel_b,
      asset.data.joint_vel,
      base_height,
    ),
    dim=-1,
  )


def go2_dreamwaq_velocity_target(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return the 3-D body-frame velocity label used by DreamWaQ's VAE."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b


def go2_source_spring_jump_observation(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Build the 47-D spring-jump frame (zero phase channels)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  zeros = torch.zeros((command.shape[0], 2), device=command.device, dtype=command.dtype)
  roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
  euler_xyz = torch.stack((roll, pitch, yaw), dim=-1)
  delayed = _go2_delayed_policy_state(env)
  if delayed is None:
    imu = torch.cat((asset.data.root_link_ang_vel_b * 0.25, euler_xyz), dim=-1)
    q_rel = asset.data.joint_pos - asset.data.default_joint_pos
    joint_vel = asset.data.joint_vel * 0.05
  else:
    q_rel, joint_vel, imu = delayed
  return torch.cat(
    (
      zeros,
      command[:, :3],
      imu,
      q_rel,
      joint_vel,
      env.action_manager.action,
    ),
    dim=-1,
  )


def go2_source_spring_jump_privileged_observation(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Build the 65-D spring-jump privileged frame."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  sensor: ContactSensor = env.scene[sensor_name]
  contact = _go2_source_contact_mask(sensor, threshold=5.0).to(asset.data.joint_pos.dtype)
  try:
    command_term = env.command_manager.get_term(command_name)
  except (KeyError, AttributeError):
    command_term = None
  has_jumped = getattr(command_term, "has_jumped", None)
  if has_jumped is None:
    jumped = torch.zeros((command.shape[0], 1), device=command.device, dtype=command.dtype)
  else:
    jumped = has_jumped.to(device=command.device, dtype=command.dtype).unsqueeze(-1)
  return torch.cat(
    (
      command[:, :3],
      asset.data.joint_pos - asset.data.default_joint_pos,
      asset.data.joint_pos,
      asset.data.joint_vel * 0.05,
      env.action_manager.action,
      asset.data.root_link_lin_vel_b,
      asset.data.root_link_ang_vel_b * 0.25,
      asset.data.projected_gravity_b,
      contact,
      jumped,
    ),
    dim=-1,
  )


def go2_source_dreamwaq_privileged_observation(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  sensor_name: str = "terrain_scan",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Build the 261-D DreamWaQ privileged frame."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  terrain = _go2_terrain_heights(env, sensor_name)
  zeros = _go2_domain_randomization_fields(env, include_mass_com=False).to(terrain.dtype)
  return torch.cat(
    (
      terrain,
      asset.data.root_link_lin_vel_b,
      zeros,
      command[:, :3] * torch.tensor((2.0, 2.0, 0.25), device=env.device),
      asset.data.root_link_ang_vel_b * 0.25,
      asset.data.projected_gravity_b,
      asset.data.joint_pos - asset.data.default_joint_pos,
      asset.data.joint_vel * 0.05,
      env.action_manager.action,
    ),
    dim=-1,
  )


def go2_source_cts_privileged_observation(
  env: ManagerBasedRlEnv,
  sensor_name: str = "terrain_scan",
  contact_sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Build the 233-D CTS encoder input (42 randomization + contacts + terrain)."""
  terrain = _go2_terrain_heights(env, sensor_name)
  sensor: ContactSensor = env.scene[contact_sensor_name]
  reserved = _go2_domain_randomization_fields(env, include_mass_com=True).to(terrain.dtype)
  return torch.cat((reserved, _go2_source_contact_mask(sensor, threshold=1.0).to(terrain.dtype), terrain), dim=-1)


def go2_source_ts_privileged_observation(
  env: ManagerBasedRlEnv,
  contact_sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Build the 74-D terrain-independent TS teacher encoder input.

  The legacy TS environments expose ``privileged_buf`` separately from the
  309-D critic observation.  It contains the 70 source domain-randomization
  fields followed by a four-foot contact mask; terrain heights and base linear
  velocity stay in the critic-only path.  Keep this group separate in mjlab so
  the teacher encoder receives the same contract as the source runner.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[contact_sensor_name]

  # The shared helper provides the scalar friction/restitution, base mass and
  # COM, PD-gain and torque fields.  The source additionally stores 28 link
  # mass ratios between base mass and COM.  MuJoCo assets do not expose the
  # exact Isaac-Gym link ordering.  Fixed URDF children are collapsed into the
  # moving parents, so their unavailable ratios remain at the neutral value 1
  # and represented links are placed in their original 28-field slots.
  base_fields = _go2_domain_randomization_fields(env, include_mass_com=True)
  link_mass = torch.ones((env.num_envs, 28), device=env.device, dtype=base_fields.dtype)
  try:
    model = env.sim.model
    body_ids = asset.indexing.body_ids
    source_slots = {
      f"{leg}_{link}": 2 + leg_index * 6 + link_index
      for leg_index, leg in enumerate(("FL", "FR", "RL", "RR"))
      for link_index, link in enumerate(("hip", "thigh", "calf"))
    }
    default_mass = env.sim.get_default_field("body_mass")
    for local_index, body_name in enumerate(asset.body_names):
      source_index = source_slots.get(body_name)
      if source_index is None:
        continue
      body_id = body_ids[local_index]
      link_mass[:, source_index] = (
        model.body_mass[:, body_id] / default_mass[body_id].clamp_min(1.0e-6)
      )
  except (AttributeError, IndexError, RuntimeError, TypeError):
    pass

  # Reorder the 42-D shared block to match the source layout:
  # friction, restitution, base mass, link masses, COM, kp, kd, torque.
  base_mass = base_fields[:, 2:3]
  com = base_fields[:, 3:6]
  gains_and_torque = base_fields[:, 6:]
  domain = torch.cat((base_fields[:, :2], base_mass, link_mass, com, gains_and_torque), dim=-1)
  contact = _go2_source_contact_mask(sensor, threshold=1.0).to(domain.dtype)
  return torch.cat((domain, contact), dim=-1)


def go2_source_cts_critic_observation(
  env: ManagerBasedRlEnv,
  sensor_name: str = "terrain_scan",
  contact_sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  include_lin_vel: bool = False,
) -> torch.Tensor:
  """Build the 278/281-D CTS value input from actor and privileged fields."""
  asset: Entity = env.scene[asset_cfg.name]
  actor = go2_source_stand_observation(env, asset_cfg=asset_cfg, command_first=True)
  privileged = go2_source_cts_privileged_observation(env, sensor_name, contact_sensor_name)
  if include_lin_vel:
    return torch.cat((actor, privileged, asset.data.root_link_lin_vel_b), dim=-1)
  return torch.cat((actor, privileged), dim=-1)


def go2_source_ts_critic_observation(
  env: ManagerBasedRlEnv,
  sensor_name: str = "terrain_scan",
  contact_sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Build the 309-D TS value input in the source concatenation order.

  The legacy ``LeggedRobotAMP_TS`` critic buffer is
  ``base_lin_vel || actor_obs || domain(70) || contacts(4) || terrain(187)``.
  Keep the velocity prefix (unlike CTS, which appends it for AMP-CTS) because
  trained TS checkpoints and the value network depend on this ordering.
  """
  asset: Entity = env.scene[asset_cfg.name]
  actor = go2_source_stand_observation(env, asset_cfg=asset_cfg, command_first=True)
  ts_privileged = go2_source_ts_privileged_observation(
    env, contact_sensor_name=contact_sensor_name, asset_cfg=asset_cfg,
  )
  domain = ts_privileged[:, :70]
  contact = ts_privileged[:, 70:]
  terrain = _go2_terrain_heights(env, sensor_name).to(actor.dtype)
  return torch.cat(
    (asset.data.root_link_lin_vel_b, actor, domain, contact, terrain), dim=-1
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
