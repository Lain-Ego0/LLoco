"""Unitree Go2 velocity environment configurations."""

import math
from typing import Literal

from mjlab.asset_zoo.robots import (
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import Go2TriggeredCommandCfg, UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import (
  discrete_obstacles,
  hf_pyramid_slope,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise.noise_cfg import UniformNoiseCfg

TerrainType = Literal["rough", "obstacles"]


def _go2_source_geom_friction(
  cfg: ManagerBasedRlEnvCfg,
  ranges: tuple[float, float],
) -> None:
  """Match Isaac Gym's shared friction sample across every Go2 shape.

  The source ``_process_rigid_shape_props`` callback assigns one scalar
  coefficient to every rigid shape in an environment.  MuJoCo's first geom
  friction component is the closest equivalent.  Do not retain the generic
  torsional/rolling randomizers here: those are useful for the public mjlab
  baseline, but are not sampled by the Go2 source tasks.
  """
  event = cfg.events.get("foot_friction_slide")
  if event is None:
    return
  asset_cfg = event.params["asset_cfg"]
  asset_cfg.geom_names = (r".*",)
  event.params.update({
    "ranges": ranges,
    "axes": [0],
    "shared_random": True,
  })
  cfg.events.pop("foot_friction_spin", None)
  cfg.events.pop("foot_friction_roll", None)


def _go2_source_noise_cfg(
  *, command_first: bool, dof_pos_noise: float = 0.01, ang_vel_noise: float = 0.2
) -> UniformNoiseCfg:
  """Return the source uniform observation-noise vector for a 45-D frame."""
  # Source noise is sampled in [-scale, scale].  The stand tasks place IMU,
  # gravity, then command; CTS/DreamWaQ/TS place command first.
  imu = [ang_vel_noise * 0.25] * 3
  gravity = [0.05] * 3
  command = [0.0] * 3
  q = [dof_pos_noise] * 12
  dq = [1.5 * 0.05] * 12
  actions = [0.0] * 12
  values = command + imu + gravity + q + dq + actions if command_first else imu + gravity + command + q + dq + actions
  return UniformNoiseCfg(
    n_min=tuple(-value for value in values),
    n_max=tuple(values),
  )


def _go2_source_47_noise_cfg() -> UniformNoiseCfg:
  """Return the source uniform noise vector for phase-based 47-D frames."""
  values = [0.0] * 5 + [0.2 * 0.25] * 3 + [0.1] * 3 + [0.01] * 12 + [1.5 * 0.05] * 12 + [0.0] * 12
  return UniformNoiseCfg(n_min=tuple(-value for value in values), n_max=tuple(values))


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Set raycast sensor frame to the Go2 base link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  thigh_geom_names = tuple(
    f"{leg}_thigh_collision" for leg in foot_names
  )
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=thigh_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  calf_geom_names = tuple(
    f"{leg}_calf{i}_collision" for leg in foot_names for i in (1, 2)
  )
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=calf_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  trunk_head_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r"base.*_collision",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    thigh_ground_cfg,
    shank_ground_cfg,
    trunk_head_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_ACTION_SCALE

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Replace the base foot_friction with per-axis friction events for condim 6.
  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  # Per-body-group collision penalties.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": shank_ground_cfg.name},
  )
  cfg.rewards["trunk_head_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": trunk_head_ground_cfg.name},
  )

  # On rough terrain the quadruped tilts significantly; don't terminate on
  # orientation alone. Let out_of_terrain_bounds handle resets.
  cfg.terminations.pop("fell_over", None)

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensors and collision sensors not needed on flat.
  remove_sensors = {
    "terrain_scan",
    "self_collision",
    "thigh_ground_touch",
    "shank_ground_touch",
    "trunk_ground_touch",
  }
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  # Remove granular collision rewards (not useful on flat ground).
  for key in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards.pop(key, None)

  # On flat terrain fell_over is sufficient; thigh contact implies fallen.
  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def unitree_go2_trot_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the source-compatible Go2 trot task on flat terrain.

  This is the first special-action task in the migration.  It keeps the common
  mjlab velocity MDP for scene plumbing while adding the source task's 47-D
  single-frame observation, ten-frame history, phase gait reward, and standstill
  terms.  Latency buffers and symmetry augmentation are migrated separately.
  """
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_base"].params["pose_range"].update(
    {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
  )
  # Trot's source asset still penalizes thigh/calf/base contacts and resets
  # on base contact even though it trains on a plane.  The public flat
  # baseline removes these sensors, so restore only the three source body
  # groups here instead of weakening the task's collision contract.
  rough_sensor_cfgs = unitree_go2_rough_env_cfg(play=play).scene.sensors or ()
  collision_sensor_names = {"thigh_ground_touch", "shank_ground_touch", "trunk_ground_touch"}
  cfg.scene.sensors = (cfg.scene.sensors or ()) + tuple(
    sensor for sensor in rough_sensor_cfgs if sensor.name in collision_sensor_names
  )
  cfg.terminations.pop("fell_over", None)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "trunk_ground_touch"},
  )
  _go2_source_geom_friction(cfg, (0.2, 1.2))
  cfg.events["encoder_bias"].params["bias_range"] = (-0.035, 0.035)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.03, 0.03), 1: (-0.03, 0.03), 2: (-0.03, 0.03),
  }
  cfg.events["go2_base_mass"] = EventTermCfg(
    func=envs_mdp.dr.body_mass,
    mode="startup",
    params={
      "ranges": (-1.0, 2.0),
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "operation": "add",
    },
  )
  cfg.events["go2_link_mass"] = EventTermCfg(
    func=envs_mdp.dr.body_mass,
    mode="startup",
    params={
      "ranges": (0.9, 1.1),
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=(r"(?:FL|FR|RL|RR)_(?:hip|thigh|calf)",)
      ),
      "operation": "scale",
    },
  )
  cfg.events["go2_pd_gains"] = EventTermCfg(
    func=envs_mdp.dr.pd_gains,
    mode="startup",
    params={
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
      "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
      "operation": "scale",
    },
  )
  # The source ``go2_trot`` starts from a symmetric trot pose.  The shared Go2
  # asset defaults intentionally follow the jump/stand tasks (rear thigh 1.0
  # and signed hip offsets), so override only this task's reset pose here.
  cfg.scene.entities["robot"].init_state.joint_pos.update(
    {
      "FL_hip_joint": 0.0,
      "FR_hip_joint": 0.0,
      "RL_hip_joint": 0.0,
      "RR_hip_joint": 0.0,
      "FL_thigh_joint": 0.8,
      "FR_thigh_joint": 0.8,
      "RL_thigh_joint": 0.8,
      "RR_thigh_joint": 0.8,
    }
  )
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  cfg.actions["joint_pos"] = mdp.Go2DelayedJointPositionActionCfg(
    entity_name=joint_pos_action.entity_name,
    actuator_names=joint_pos_action.actuator_names,
    scale=joint_pos_action.scale,
    offset=joint_pos_action.offset,
    clip=joint_pos_action.clip,
    preserve_order=joint_pos_action.preserve_order,
    use_default_offset=joint_pos_action.use_default_offset,
    delay=True,
    delay_mode="buffer",
    delay_steps_range=(1, 3),
    observation_latency=True,
    observation_motor_latency_range=(1, 3),
    observation_imu_latency_range=(1, 3),
  )

  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      "source_trot": ObservationTermCfg(
        func=mdp.go2_source_trot_observation,
        params={"command_name": "twist"},
        noise=_go2_source_47_noise_cfg(),
        history_length=10,
        flatten_history_dim=True,
      )
    },
    concatenate_terms=True,
    enable_corruption=not play,
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms={
      "source_trot_privileged": ObservationTermCfg(
        func=mdp.go2_source_trot_privileged_observation,
        params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
        history_length=3,
        flatten_history_dim=True,
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.ranges.ang_vel_z = (-1.0, 1.0)
  twist.source_zero_command_prob = 0.05
  twist.source_zero_xy_prob = 0.05
  twist.source_min_lin_norm = 0.1
  twist.resampling_time_range = (5.0, 5.0)

  cfg.rewards["trot"] = RewardTermCfg(
    func=mdp.go2_trot_phase_reward,
    weight=0.8,
    params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
  )
  cfg.rewards["feet_clearance_source"] = RewardTermCfg(
    func=mdp.go2_trot_feet_clearance_reward,
    weight=0.1,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL"), preserve_order=True),
    },
  )
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.go2_source_tracking_linear_velocity,
    weight=2.0,
    params={"command_name": "twist", "sigma": 0.25, "trot_gate": True},
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.go2_source_tracking_angular_velocity,
    weight=2.0,
    params={"command_name": "twist", "sigma": 0.25, "trot_gate": True},
  )
  cfg.rewards["upright"] = RewardTermCfg(
    func=mdp.go2_orientation_penalty,
    weight=-2.0,
    params={},
  )
  cfg.rewards["body_ang_vel"] = RewardTermCfg(
    func=mdp.go2_angular_velocity_xy_penalty,
    weight=-0.05,
    params={},
  )
  cfg.rewards["lin_vel_z"] = RewardTermCfg(
    func=mdp.go2_linear_velocity_z_penalty,
    weight=-2.0,
    params={},
  )
  cfg.rewards["base_height"] = RewardTermCfg(
    func=mdp.go2_base_height_penalty,
    weight=-5.0,
    params={"target_height": 0.29, "sensor_name": "terrain_scan"},
  )
  cfg.rewards["dof_acc"] = RewardTermCfg(
    func=mdp.go2_joint_acceleration_penalty,
    weight=-2.5e-7,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["torques"] = RewardTermCfg(
    # Source Trot uses the base LeggedRobot squared-torque kernel.
    func=envs_mdp.joint_torques_l2,
    weight=-0.0001,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
  )
  cfg.rewards["action_smoothness"] = RewardTermCfg(
    func=mdp.go2_action_smoothness_penalty,
    # Source Trot only has first-order action-rate cost; the generic
    # second-order smoothness term is not part of its reward contract.
    weight=0.0,
    params={},
  )
  cfg.rewards["dof_pos_limits"].weight = 0.0
  cfg.rewards["action_rate_l2"].weight = -0.01
  cfg.rewards["pose"].weight = 0.0
  cfg.rewards["foot_clearance"].weight = 0.0
  cfg.rewards["foot_swing_height"].weight = 0.0
  cfg.rewards["foot_slip"].weight = 0.0
  cfg.rewards["soft_landing"].weight = 0.0
  cfg.rewards["default_hip_pos"] = RewardTermCfg(
    func=mdp.go2_default_hip_position_penalty,
    weight=-0.2,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["default_pos"] = RewardTermCfg(
    func=mdp.go2_default_position_penalty,
    weight=-0.1,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=mdp.go2_stand_still_penalty,
    weight=-1.0,
    params={"command_name": "twist"},
  )
  cfg.rewards["contact_without_command"] = RewardTermCfg(
    func=mdp.go2_contact_without_command,
    weight=1.0,
    params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
  )
  cfg.rewards["collision"] = RewardTermCfg(
    func=mdp.go2_special_collision_penalty,
    weight=-1.0,
    params={"sensor_names": tuple(sorted(collision_sensor_names))},
  )
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].interval_range_s = (4.0, 4.0)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.4, 0.4), "y": (-0.4, 0.4), "z": (0.0, 0.0),
      "roll": (-0.6, 0.6), "pitch": (-0.6, 0.6), "yaw": (-0.6, 0.6),
    }
  return cfg


def _go2_special_action_env_cfg(
  task: Literal["jump", "spring_jump", "backflip", "handstand", "leggedstand"],
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build a registered scaffold for a source Go2 special-action task.

  The common scene, actuator, reset and PPO plumbing is shared with the flat
  task.  Each task gets a distinct reward shaping term so its source-specific
  controller can be filled in without duplicating asset configuration.
  """
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_base"].params["pose_range"].update(
    {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
  )
  # Flat-task construction removes terrain collision sensors for the common
  # velocity baseline. Special-action source rewards still depend on body
  # collisions, so restore only the non-foot sensors here.
  rough_sensor_cfgs = unitree_go2_rough_env_cfg(play=play).scene.sensors or ()
  collision_sensor_names = {
    "self_collision", "thigh_ground_touch", "shank_ground_touch", "trunk_ground_touch",
  }
  cfg.scene.sensors = (cfg.scene.sensors or ()) + tuple(
    sensor for sensor in rough_sensor_cfgs if sensor.name in collision_sensor_names
  )
  # The source special tasks allow deliberate inversion/rotation.  The flat
  # baseline's 70-degree ``fell_over`` termination would stop a jump or
  # handstand before it can reach its target orientation, so replace it with
  # the source base-contact termination.
  cfg.terminations.pop("fell_over", None)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "trunk_ground_touch"},
  )
  collision_weight = {
    "jump": -1.0,
    "spring_jump": -50.0,
    "backflip": -10.0,
    "handstand": -2.0,
    "leggedstand": -1.0,
  }[task]
  cfg.rewards["collision"] = RewardTermCfg(
    func=mdp.go2_special_collision_penalty,
    weight=collision_weight,
    params={"sensor_names": tuple(sorted(collision_sensor_names))},
  )
  # Source special-task startup randomization.  The shared rough config owns
  # the event plumbing; specialize its ranges here instead of using the
  # public rough-task defaults.
  friction_range = {
    "jump": (0.2, 1.2),
    "spring_jump": (0.3, 1.0),
    "backflip": (0.2, 1.25),
    "handstand": (0.2, 1.2),
    "leggedstand": (0.2, 1.2),
  }[task]
  _go2_source_geom_friction(cfg, friction_range)
  cfg.events["encoder_bias"].params["bias_range"] = (-0.035, 0.035)
  com_extent = 0.05 if task in ("handstand", "leggedstand") else 0.03
  cfg.events["base_com"].params["ranges"] = {
    0: (-com_extent, com_extent),
    1: (-com_extent, com_extent),
    2: (-com_extent, com_extent),
  }
  base_mass_range = (-1.0, 2.0) if task in ("handstand", "leggedstand") else (-1.0, 1.0)
  cfg.events["go2_base_mass"] = EventTermCfg(
    func=envs_mdp.dr.body_mass,
    mode="startup",
    params={
      "ranges": base_mass_range,
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "operation": "add",
    },
  )
  cfg.events["go2_link_mass"] = EventTermCfg(
    func=envs_mdp.dr.body_mass,
    mode="startup",
    params={
      "ranges": (0.9, 1.1),
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=(r"(?:FL|FR|RL|RR)_(?:hip|thigh|calf)",)
      ),
      "operation": "scale",
    },
  )
  cfg.events["go2_pd_gains"] = EventTermCfg(
    func=envs_mdp.dr.pd_gains,
    mode="startup",
    params={
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
      "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
      "operation": "scale",
    },
  )
  if task in ("handstand", "leggedstand"):
    cfg.events["reset_robot_joints"].func = mdp.reset_joints_by_scale
    cfg.events["reset_robot_joints"].params.pop("position_range", None)
    cfg.events["reset_robot_joints"].params["scale_range"] = (0.5, 1.5)
    # The source stand tasks initialize all six root velocities uniformly in
    # [-0.5, 0.5], unlike the flat locomotion tasks which start at rest.
    cfg.events["reset_base"].params["velocity_range"] = {
      "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
      "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
    }
    joint_extent = (0.01, 0.1) if task == "handstand" else (0.01, 0.2)
    damping_extent = (0.0, 0.1) if task == "handstand" else (0.0, 0.2)
    armature_extent = (0.003, 0.08) if task == "handstand" else (0.005, 0.015)
    joint_cfg = SceneEntityCfg("robot", joint_names=(".*",))
    cfg.events["go2_joint_friction"] = EventTermCfg(
      func=envs_mdp.dr.joint_friction,
      mode="startup",
      params={"ranges": joint_extent, "asset_cfg": joint_cfg, "operation": "abs"},
    )
    cfg.events["go2_joint_damping"] = EventTermCfg(
      func=envs_mdp.dr.joint_damping,
      mode="startup",
      params={"ranges": damping_extent, "asset_cfg": joint_cfg, "operation": "abs"},
    )
    cfg.events["go2_joint_armature"] = EventTermCfg(
      func=envs_mdp.dr.joint_armature,
      mode="startup",
      params={"ranges": armature_extent, "asset_cfg": joint_cfg, "operation": "abs"},
    )
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  cfg.actions["joint_pos"] = mdp.Go2DelayedJointPositionActionCfg(
    entity_name=joint_pos_action.entity_name,
    actuator_names=joint_pos_action.actuator_names,
    scale=joint_pos_action.scale,
    offset=joint_pos_action.offset,
    clip=joint_pos_action.clip,
    preserve_order=joint_pos_action.preserve_order,
    use_default_offset=joint_pos_action.use_default_offset,
    delay=True,
    observation_latency=task in ("jump", "spring_jump", "backflip", "leggedstand"),
    observation_imu_orientation="gravity" if task == "backflip" else "euler",
    observation_motor_latency_range=(1, 3),
    observation_imu_latency_range=(1, 3),
  )
  if task == "jump":
    # Jump uses a reset-sampled 1–3 substep command buffer in the source.
    delayed_action = cfg.actions["joint_pos"]
    assert isinstance(delayed_action, mdp.Go2DelayedJointPositionActionCfg)
    delayed_action.delay_mode = "buffer"
    delayed_action.delay_steps_range = (1, 3)
  elif task == "spring_jump":
    # Source Spring-Jump starts exactly at its configured default pose.
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
  elif task == "backflip":
    # Source Backflip uses default_q * U(0.5, 1.25).
    cfg.events["reset_robot_joints"].func = mdp.reset_joints_by_scale
    cfg.events["reset_robot_joints"].params.pop("position_range", None)
    cfg.events["reset_robot_joints"].params["scale_range"] = (0.5, 1.25)
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  # All source special-action configs sample angular velocity directly; only
  # handstand enables heading tracking below.
  twist.heading_command = False
  twist.ranges.heading = None
  twist.resampling_time_range = (5.0, 5.0)
  # The source jump/spring/backflip configs sample yaw rate directly from
  # [-1, 1] rad/s (the common mjlab velocity template defaults to [-0.5, 0.5]).
  # Stand tasks override this below with their narrower source ranges.
  if task in ("jump", "spring_jump", "backflip"):
    twist.ranges.ang_vel_z = (-1.0, 1.0)
  if task == "jump":
    twist.source_zero_command_prob = 0.05
    twist.source_zero_xy_prob = 0.05
    twist.source_min_lin_norm = 0.1
  elif task in ("spring_jump", "backflip"):
    twist.source_zero_command_prob = 0.05
    twist.source_zero_xy_prob = 0.05
    twist.source_min_lin_norm = 0.1
  if task in ("spring_jump", "backflip", "handstand", "leggedstand"):
    # These source command configs explicitly disable command curriculum.
    cfg.curriculum.pop("command_vel", None)
  if "push_robot" in cfg.events:
    push_interval = 8.0 if task in ("handstand", "leggedstand") else 4.0
    push_linear = 1.0 if task == "leggedstand" else 0.4
    push_angular = 1.0 if task == "leggedstand" else 0.6
    cfg.events["push_robot"].interval_range_s = (push_interval, push_interval)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-push_linear, push_linear), "y": (-push_linear, push_linear),
      "z": (0.0, 0.0),
      "roll": (-push_angular, push_angular),
      "pitch": (-push_angular, push_angular),
      "yaw": (-push_angular, push_angular),
    }
  if task in ("spring_jump", "backflip"):
    # In the source flip environments command[2] is a delayed one-shot
    # trigger, not a yaw-rate target.  Keep the same three-value observation
    # contract while exposing the trigger/flight state through the term.
    if task == "spring_jump":
      initial_lin_vel_x = (0.8, 1.2)
      trigger_steps = (50, 60)
    else:
      initial_lin_vel_x = (0.0, 0.0)
      trigger_steps = (50, 100)
    twist.ranges.lin_vel_x = initial_lin_vel_x
    twist.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands["twist"] = Go2TriggeredCommandCfg(
      entity_name=twist.entity_name,
      resampling_time_range=twist.resampling_time_range,
      ranges=twist.ranges,
      heading_command=False,
      rel_standing_envs=0.0,
      rel_forward_envs=0.0,
      source_zero_command_prob=twist.source_zero_command_prob,
      source_zero_xy_prob=twist.source_zero_xy_prob,
      source_min_lin_norm=twist.source_min_lin_norm,
      trigger_steps=trigger_steps,
      initial_lin_vel_x=initial_lin_vel_x,
      contact_sensor_name="feet_ground_contact",
      push_towards_goal=True,
      upward_push_range=(1.5, 2.2) if task == "spring_jump" else (2.0, 3.5),
      pitch_push_range=(0.0, 0.0) if task == "spring_jump" else (2.0, 2.5),
    )
    twist = cfg.commands["twist"]
    assert isinstance(twist, Go2TriggeredCommandCfg)
  if task == "jump":
    # The source jump task is the one special-action task trained on a
    # curriculum of upward stairs (spring-jump and backflip use a plane).
    # Re-enable a generator after the common flat config removed it, while
    # retaining the source observation contract that does not expose heights.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
      size=(8.0, 8.0),
      border_width=25.0,
      num_rows=10,
      num_cols=20,
      curriculum=False,
      sub_terrains={
        "stairs": pyramid_stairs(
          proportion=1.0,
          step_height_range=(0.0, 0.1),
          step_width=0.31,
          platform_width=3.0,
          border_width=1.0,
        )
      },
      add_lights=True,
    )
    cfg.episode_length_s = 24.0
    cfg.observations["actor"] = ObservationGroupCfg(
      terms={
        "source_jump": ObservationTermCfg(
          func=mdp.go2_source_trot_observation,
          params={"command_name": "twist", "cycle_time": 1.5},
          noise=_go2_source_47_noise_cfg(),
          history_length=10,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=not play,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "source_jump_privileged": ObservationTermCfg(
          func=mdp.go2_source_jump_privileged_observation,
          params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
          history_length=3,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.rewards["jump"] = RewardTermCfg(
      func=mdp.go2_jump_contact_reward,
      weight=2.0,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )
    cfg.rewards["feet_clearance"] = RewardTermCfg(
      func=mdp.go2_jump_feet_clearance,
      weight=0.5,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL")),
        "command_name": "twist",
      },
    )
    cfg.rewards["foot_clearance"].weight = 0.0
    cfg.rewards["foot_swing_height"].weight = 0.0
  elif task == "spring_jump":
    cfg.episode_length_s = 5.0
    # The source spring-jump controller starts from a slightly crouched pose.
    robot = cfg.scene.entities["robot"]
    robot.init_state.pos = (0.0, 0.0, 0.39)
    robot.init_state.joint_pos.update(
      {
        "FL_hip_joint": 0.0,
        "FR_hip_joint": 0.0,
        "RL_hip_joint": 0.0,
        "RR_hip_joint": 0.0,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
      }
    )
    cfg.observations["actor"] = ObservationGroupCfg(
      terms={
        "source_spring_jump": ObservationTermCfg(
          func=mdp.go2_source_spring_jump_observation,
          params={"command_name": "twist"},
          noise=_go2_source_47_noise_cfg(),
          history_length=10,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=not play,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "source_spring_jump_privileged": ObservationTermCfg(
          func=mdp.go2_source_spring_jump_privileged_observation,
          params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
          history_length=3,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.rewards["flight"] = RewardTermCfg(
      # Source ``_reward_flight`` returns the command term's persistent
      # ``was_in_flight`` flag, rather than a periodic contact-phase kernel.
      # Passing the command name lets the mjlab term read that state and keeps
      # the reward active until the first landing.
      func=mdp.go2_all_feet_airborne,
      weight=2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
      },
    )
    cfg.terminations["reset_height"] = TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.15},
    )
  elif task in ("handstand", "leggedstand"):
    cfg.episode_length_s = 20.0
    # Stand tasks use the higher-gain controller from the source configs.
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
      actuator.stiffness = 40.0
      actuator.damping = 1.0
    if task == "handstand":
      twist.ranges.lin_vel_x = (-0.2, 0.6)
      twist.ranges.lin_vel_y = (0.0, 0.0)
      twist.ranges.ang_vel_z = (-0.4, 0.4)
      twist.heading_command = True
      twist.ranges.heading = (-math.pi, math.pi)
      twist.rel_heading_envs = 1.0
      twist.rel_standing_envs = 0.0
      twist.rel_forward_envs = 0.0
      twist.source_zero_command_prob = 0.20
      twist.source_zero_xy_prob = 0.10
      twist.source_min_lin_norm = 0.1
      twist.resampling_time_range = (10.0, 10.0)
    else:
      twist.ranges.lin_vel_x = (-0.4, 0.4)
      twist.ranges.lin_vel_y = (0.0, 0.0)
      twist.ranges.ang_vel_z = (-0.4, 0.4)
      twist.heading_command = False
      twist.source_zero_command_prob = 0.20
      twist.source_zero_xy_prob = 0.10
      twist.source_min_lin_norm = 0.1
      twist.resampling_time_range = (5.0, 5.0)
    # Neither stand variant uses the generic template's standing/forward
    # environment mixture.
    twist.rel_standing_envs = 0.0
    twist.rel_forward_envs = 0.0
    cfg.observations["actor"] = ObservationGroupCfg(
      terms={
        "source_stand": ObservationTermCfg(
          func=mdp.go2_source_stand_observation,
          params={"command_name": "twist"},
          noise=_go2_source_noise_cfg(command_first=False),
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=not play,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "source_stand_privileged": ObservationTermCfg(
          func=mdp.go2_source_stand_privileged_observation,
          params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.rewards["stand_still"] = RewardTermCfg(
      func=mdp.go2_stand_still_penalty,
      weight=-1.0,
      params={"command_name": "twist"},
    )
    if task == "handstand":
      target_gravity = (-1.0, 0.0, 0.0)
      target_height = 0.52
      airborne_indices = (0, 1)  # FL/FR in the source task.
      support_indices = (2, 3)  # RL/RR support pair.
      target_joint_pos = (
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5,
        0.0, 2.25, -1.75,
        0.0, 2.25, -1.75,
      )
      target_joint_slice = (0, 6)
    else:
      target_gravity = (1.0, 0.0, 0.0)
      target_height = 0.47
      airborne_indices = (2, 3)  # RL/RR in the source task.
      support_indices = (0, 1)  # FL/FR support pair.
      target_joint_pos = (
        0.0, -0.7, -1.75,
        0.0, -0.7, -1.75,
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5,
      )
      target_joint_slice = (6, 12)
    cfg.rewards["target_gravity"] = RewardTermCfg(
      func=mdp.go2_target_gravity_error,
      weight=-1.0,
      params={"target_gravity": target_gravity},
    )
    cfg.rewards["target_base_height"] = RewardTermCfg(
      func=mdp.go2_target_base_height_reward,
      weight=1.5 if task == "handstand" else 1.0,
      params={"target_height": target_height, "sharpness": 5.0},
    )
    cfg.rewards["handstand_feet_on_air"] = RewardTermCfg(
      func=mdp.go2_selected_feet_airborne,
      weight=0.4,
      params={"sensor_name": "feet_ground_contact", "foot_indices": airborne_indices},
    )
    cfg.rewards["single_support_contact"] = RewardTermCfg(
      func=mdp.go2_exactly_one_foot_contact,
      weight=0.3,
      params={"sensor_name": "feet_ground_contact", "foot_indices": support_indices},
    )
    cfg.rewards["desired_pose"] = RewardTermCfg(
      func=mdp.go2_target_joint_position_penalty,
      weight=-0.1 if task == "handstand" else -0.05,
      params={
        "target_joint_pos": target_joint_pos,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    )
    cfg.rewards["desired_pose_reward"] = RewardTermCfg(
      func=mdp.go2_target_joint_position_reward,
      weight=0.5,
      params={
        "target_joint_pos": target_joint_pos,
        "joint_slice": target_joint_slice,
        "sharpness": 1.0,
        "ready_target_height": target_height,
        "ready_threshold": 0.70 if task == "handstand" else 0.78,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    )
    cfg.rewards["default_pos"] = RewardTermCfg(
      func=mdp.go2_target_joint_position_penalty,
      weight=-0.1 if task == "handstand" else -0.05,
      params={
        "target_joint_pos": target_joint_pos,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    )
    cfg.rewards["default_hip_pos"] = RewardTermCfg(
      func=mdp.go2_hip_position_penalty,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["symmetric_joints"] = RewardTermCfg(
      func=mdp.go2_joint_symmetry_penalty,
      weight=-0.1,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        # Leggedstand's source reward compares only the rear/support pair;
        # Handstand keeps the default two-pair comparison.
        "leg_indices": (2, 3) if task == "leggedstand" else (0, 1, 2, 3),
      },
    )
    stand_joint_cfg = SceneEntityCfg("robot", joint_names=(".*",))
    stand_site_cfg = SceneEntityCfg(
      "robot",
      site_names=("FL", "FR", "RL", "RR"),
      preserve_order=True,
    )
    if task == "handstand":
      stand_forward_sign = -1.0
      stand_ready_height = 0.52
      swing_feet = (0, 1)
      support_feet = (2, 3)
      track_threshold = 0.70
    else:
      stand_forward_sign = 1.0
      stand_ready_height = 0.47
      swing_feet = (2, 3)
      support_feet = (0, 1)
      track_threshold = 0.78
    cfg.rewards["tracking_lin_vel_source"] = RewardTermCfg(
      func=mdp.go2_stand_linear_velocity_reward,
      weight=2.5,
      params={
        "command_name": "twist",
        "target_height": stand_ready_height,
        "ready_threshold": track_threshold,
        "forward_sign": stand_forward_sign,
        "sigma": 0.25,
      },
    )
    cfg.rewards["tracking_ang_vel_source"] = RewardTermCfg(
      func=mdp.go2_stand_angular_velocity_reward,
      weight=2.5,
      params={
        "command_name": "twist",
        "target_height": stand_ready_height,
        "ready_threshold": track_threshold,
        "forward_sign": stand_forward_sign,
        "sigma": 0.25,
      },
    )
    if task == "leggedstand":
      cfg.rewards["tracking_lin_vel_zero"] = RewardTermCfg(
        func=mdp.go2_stand_zero_linear_velocity_reward,
        weight=-0.2,
        params={
          "command_name": "twist",
          "target_height": stand_ready_height,
          "ready_threshold": track_threshold,
          "forward_sign": stand_forward_sign,
          "sigma": 0.25,
        },
      )
      cfg.rewards["tracking_ang_vel_zero"] = RewardTermCfg(
        func=mdp.go2_stand_zero_angular_velocity_penalty,
        weight=-0.2,
        params={
          "command_name": "twist",
          "target_height": stand_ready_height,
          "ready_threshold": track_threshold,
        },
      )
    cfg.rewards["lin_vel_z"] = RewardTermCfg(
      func=mdp.go2_stand_linear_z_velocity_reward,
      weight=0.2,
      params={},
    )
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(
      func=mdp.go2_stand_ang_vel_penalty,
      # Source stand configs use a positive ``exp(-||ω_yz||)`` reward.
      weight=0.2,
      params={},
    )
    cfg.rewards["torques"] = RewardTermCfg(
      func=mdp.go2_torque_penalty,
      weight=-0.0002,
      params={"asset_cfg": stand_joint_cfg},
    )
    cfg.rewards["dof_vel"] = RewardTermCfg(
      func=mdp.go2_joint_velocity_penalty,
      weight=0.0,
      params={"asset_cfg": stand_joint_cfg},
    )
    cfg.rewards["dof_acc"] = RewardTermCfg(
      # Both source stand configs retain the small acceleration regularizer
      # used by the original LeggedRobot implementation.
      func=mdp.go2_joint_acceleration_penalty,
      weight=-2.5e-7,
      params={"asset_cfg": stand_joint_cfg},
    )
    cfg.rewards["feet_contact_forces"] = RewardTermCfg(
      func=mdp.go2_foot_contact_force_penalty,
      weight=-0.1,
      params={"sensor_name": "feet_ground_contact", "max_contact_force": 200.0},
    )
    # Handstand explicitly enables the soft joint-limit term; Leggedstand's
    # source reward scale leaves it disabled.
    cfg.rewards["dof_pos_limits"].weight = -2.0 if task == "handstand" else 0.0
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["handstand_feet_on_air"] = RewardTermCfg(
      func=mdp.go2_selected_feet_airborne,
      weight=0.4,
      params={"sensor_name": "feet_ground_contact", "foot_indices": swing_feet},
    )
    cfg.rewards["contact"] = RewardTermCfg(
      func=mdp.go2_stand_contact_reward,
      weight=0.3,
      params={
        "sensor_name": "feet_ground_contact",
        "foot_indices": support_feet,
        "ready_target_height": stand_ready_height,
        "ready_threshold": track_threshold,
        "asset_cfg": stand_joint_cfg,
      },
    )
    cfg.rewards["ang_xz"] = RewardTermCfg(
      func=mdp.go2_stand_roll_penalty,
      weight=-0.5,
      params={
        "ready_target_height": stand_ready_height,
        "ready_threshold": track_threshold,
        "asset_cfg": stand_joint_cfg,
      },
    )
    cfg.rewards["feet_clearance"] = RewardTermCfg(
      func=mdp.go2_stand_foot_clearance_reward,
      weight=0.4,
      params={
        # The legacy reward reads ``contact_foot_indices`` for this term;
        # these are the two support limbs (the source variable is misleadingly
        # named ``feet_clearance`` even though it tracks that pair).
        "foot_indices": support_feet,
        "cycle_time": 1.6,
        "target_height": 0.06,
        "ready_target_height": stand_ready_height,
        "ready_threshold": track_threshold,
        "asset_cfg": stand_site_cfg,
      },
    )
    if task == "leggedstand":
      cfg.rewards["feet_air_time"] = RewardTermCfg(
        func=mdp.go2_stand_foot_air_time_reward,
        weight=2.0,
        params={
          "sensor_name": "feet_ground_contact",
          "foot_indices": support_feet,
          "ready_target_height": stand_ready_height,
          "ready_threshold": track_threshold,
          "asset_cfg": stand_joint_cfg,
        },
      )
    cfg.rewards["orientation_symmetry"] = RewardTermCfg(
      func=mdp.go2_stand_orientation_symmetry_penalty,
      weight=-0.5,
      params={},
    )
    cfg.rewards["feet_height_symmetry"] = RewardTermCfg(
      func=mdp.go2_stand_feet_height_symmetry_penalty,
      weight=-0.2,
      params={"foot_indices": swing_feet, "asset_cfg": stand_site_cfg},
    )
    cfg.rewards["handstand_feet_height_exp"] = RewardTermCfg(
      func=mdp.go2_stand_hand_height_reward,
      weight=5.0,
      params={
        "target_height": 0.67,
        "sharpness": 10.0,
        # Source ``feet_pos`` is built from ``feet_name_reward`` (the two
        # airborne limbs), not all four foot sites.
        "foot_indices": airborne_indices,
        "asset_cfg": stand_site_cfg,
      },
    )
    cfg.rewards["alive"] = RewardTermCfg(
      func=mdp.go2_alive_reward,
      weight=1.0,
      params={},
    )
    for name in ("track_linear_velocity", "track_angular_velocity", "pose", "upright", "foot_slip", "soft_landing"):
      cfg.rewards[name].weight = 0.0
  if task == "jump":
    # Source jump uses a positive idle-height kernel around 0.30 m.  The
    # generic velocity template's squared penalty at 0.42 m pulls the policy
    # toward the nominal standing height and is therefore inappropriate here.
    cfg.rewards["base_height"] = RewardTermCfg(
      func=mdp.go2_idle_base_height_reward,
      weight=1.0,
      params={"target_height": 0.30, "sharpness": 10.0, "command_name": "twist"},
    )
    cfg.rewards["vertical_velocity"] = RewardTermCfg(
      func=mdp.go2_jump_vertical_velocity_reward,
      weight=0.05,
      params={},
    )
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(
      func=mdp.go2_jump_angular_velocity_reward,
      weight=0.2,
      params={},
    )
    cfg.rewards["upright"] = RewardTermCfg(
      func=mdp.go2_jump_orientation_reward,
      weight=0.6,
      params={"sharpness": 10.0},
    )
    cfg.rewards["dof_acc"] = RewardTermCfg(
      func=mdp.go2_joint_acceleration_penalty,
      weight=-5.5e-4,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["default_pos"] = RewardTermCfg(
      func=mdp.go2_default_position_penalty,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["default_hip_pos"] = RewardTermCfg(
      func=mdp.go2_default_hip_position_penalty,
      weight=0.3,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["contact_without_command"] = RewardTermCfg(
      func=mdp.go2_contact_without_command,
      weight=1.0,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )
    cfg.rewards["track_linear_velocity"] = RewardTermCfg(
      func=mdp.go2_jump_tracking_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "sigma": 0.25, "command_threshold": 0.1},
    )
    cfg.rewards["track_angular_velocity"] = RewardTermCfg(
      func=mdp.go2_jump_tracking_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "sigma": 0.25, "command_threshold": 0.1},
    )
    cfg.rewards["air_time"] = RewardTermCfg(
      func=mdp.go2_jump_feet_air_time,
      weight=1.0,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )
    cfg.rewards["feet_contact_forces"] = RewardTermCfg(
      func=mdp.go2_foot_contact_force_penalty,
      weight=-0.01,
      params={"sensor_name": "feet_ground_contact", "max_contact_force": 100.0},
    )
    cfg.rewards["torques"] = RewardTermCfg(
      # The source Jump override uses an absolute-torque penalty rather than
      # the squared kernel used by the generic velocity task.
      func=mdp.go2_torque_penalty,
      weight=-0.0002,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    # The source Jump scales do not include a joint-position-limit term, and
    # use -0.01 (rather than mjlab's generic -0.1) for action-rate cost.
    cfg.rewards["dof_pos_limits"].weight = 0.0
    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.rewards["stand_still"] = RewardTermCfg(
      func=mdp.go2_stand_still_penalty,
      weight=-1.0,
      params={"command_name": "twist"},
    )
    for name in ("pose", "foot_slip", "soft_landing"):
      cfg.rewards[name].weight = 0.0
  elif task == "spring_jump":
    # Spring-Jump alternates a take-off/flight height target with a lower
    # post-landing stance target.  The contact-derived phase mask is stateless
    # but preserves the source shaping signal in mjlab's reward manager.
    cfg.rewards["line_z"] = RewardTermCfg(
      func=mdp.go2_upward_velocity_reward,
      weight=16.0,
      params={"command_name": "twist"},
    )
    cfg.rewards["flight"] = RewardTermCfg(
      func=mdp.go2_all_feet_airborne,
      weight=2.0,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )
    cfg.rewards["base_height_flight"] = RewardTermCfg(
      func=mdp.go2_base_height_phase_reward,
      weight=3.0,
      params={
        "target_height": 0.47,
        "phase": "airborne",
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "gain": 6.0,
      },
    )
    cfg.rewards["base_height_stance"] = RewardTermCfg(
      func=mdp.go2_base_height_phase_reward,
      weight=-10.0,
      params={
        "target_height": 0.35,
        "phase": "contact",
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
      },
    )
    cfg.rewards["land_pos"] = RewardTermCfg(
      func=mdp.go2_landing_position_reward,
      weight=25.0,
      params={
        "command_name": "twist",
        "target_offset_scale": 1.0,
        "min_jump_height": 0.42,
        "max_orientation_error": 0.6,
      },
    )
    cfg.rewards["before_setting"] = RewardTermCfg(
      func=mdp.go2_before_setting_reward,
      weight=5.0,
      params={"command_name": "twist", "denominator": 2.0},
    )
    cfg.rewards["orientation"] = RewardTermCfg(
      func=mdp.go2_orientation_reward,
      weight=2.0,
      params={},
    )
    cfg.rewards["dof_pos"] = RewardTermCfg(
      func=mdp.go2_joint_position_penalty,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["dof_hip_pos"] = RewardTermCfg(
      func=mdp.go2_hip_position_penalty,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(
      func=mdp.go2_ang_vel_xy_penalty,
      weight=-0.2,
      params={"include_pitch": True},
    )
    cfg.rewards["dof_vel"] = RewardTermCfg(
      func=mdp.go2_joint_velocity_penalty,
      weight=-0.001,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["dof_vel_limits"] = RewardTermCfg(
      func=mdp.go2_joint_velocity_limit_penalty,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["torques"] = RewardTermCfg(
      func=mdp.go2_torque_penalty,
      weight=-0.0001,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["feet_contact_forces"] = RewardTermCfg(
      func=mdp.go2_foot_contact_force_penalty,
      weight=-0.1,
      params={"sensor_name": "feet_ground_contact", "max_contact_force": 150.0},
    )
    cfg.rewards["line_vel_stance"] = RewardTermCfg(
      func=mdp.go2_line_velocity_stance_penalty,
      weight=-3.0,
      params={"command_name": "twist"},
    )
    cfg.rewards["foot_clearance"] = RewardTermCfg(
      func=mdp.go2_flip_foot_clearance_penalty,
      weight=-3.0,
      params={
        "command_name": "twist",
        "asset_cfg": SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL")),
      },
    )
    cfg.rewards["tracking_lin_vel_source"] = RewardTermCfg(
      func=mdp.go2_flight_linear_velocity_reward,
      weight=5.0,
      params={"command_name": "twist", "gain": 1.6},
    )
    # Terms from the generic velocity task that have no source equivalent are
    # disabled for the flip-specific reward contract.
    for name in ("track_linear_velocity", "track_angular_velocity", "pose", "upright", "foot_slip", "soft_landing"):
      cfg.rewards[name].weight = 0.0
    cfg.rewards["dof_pos_limits"].weight = -10.0
    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.rewards.pop("base_height", None)
    cfg.rewards.pop("vertical_velocity", None)
  elif task == "backflip":
    cfg.episode_length_s = 4.0
    # Backflip starts from the neutral hip pose used by the source task.
    cfg.scene.entities["robot"].init_state.joint_pos.update(
      {
        "FL_hip_joint": 0.0,
        "FR_hip_joint": 0.0,
        "RL_hip_joint": 0.0,
        "RR_hip_joint": 0.0,
      }
    )
    cfg.observations["actor"] = ObservationGroupCfg(
      terms={
        "source_backflip": ObservationTermCfg(
          func=mdp.go2_source_backflip_observation,
          params={"command_name": "twist"},
          noise=_go2_source_47_noise_cfg(),
          history_length=10,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=not play,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "source_backflip_privileged": ObservationTermCfg(
          func=mdp.go2_source_backflip_privileged_observation,
          params={"command_name": "twist"},
          history_length=3,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.rewards["inverted_orientation"] = RewardTermCfg(
      func=mdp.go2_inverted_orientation_reward,
      weight=1.0,
      params={},
    )
    cfg.rewards["line_z"] = RewardTermCfg(
      func=mdp.go2_upward_velocity_reward,
      weight=25.0,
      params={"command_name": "twist"},
    )
    cfg.rewards["angle_y"] = RewardTermCfg(
      func=mdp.go2_pitch_angular_velocity_reward,
      weight=10.0,
      params={"command_name": "twist"},
    )
    cfg.rewards["base_height_flight"] = RewardTermCfg(
      func=mdp.go2_base_height_phase_reward,
      weight=5.0,
      params={
        "target_height": 0.60,
        "phase": "airborne",
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "gain": 6.0,
      },
    )
    cfg.rewards["base_height_stance"] = RewardTermCfg(
      func=mdp.go2_base_height_phase_reward,
      weight=10.0,
      params={
        "target_height": 0.35,
        "phase": "contact",
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
      },
    )
    cfg.rewards["backflip_orientation"] = RewardTermCfg(
      func=mdp.go2_backflip_orientation_reward,
      weight=10.0,
      params={"command_name": "twist"},
    )
    cfg.rewards["symmetric_joints"] = RewardTermCfg(
      func=mdp.go2_joint_symmetry_penalty,
      weight=-0.3,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["default_hip_pos"] = RewardTermCfg(
      func=mdp.go2_default_hip_position_penalty,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["land_pos"] = RewardTermCfg(
      func=mdp.go2_landing_position_reward,
      weight=1.0,
      params={"command_name": "twist", "target_offset_scale": 0.0},
    )
    cfg.rewards["before_setting"] = RewardTermCfg(
      func=mdp.go2_before_setting_reward,
      weight=5.0,
      params={"command_name": "twist", "denominator": 4.0},
    )
    cfg.rewards["orientation_before"] = RewardTermCfg(
      func=mdp.go2_orientation_before_reward,
      weight=2.0,
      params={"command_name": "twist"},
    )
    cfg.rewards["dof_pos"] = RewardTermCfg(
      func=mdp.go2_joint_position_penalty,
      weight=-0.2,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["line_vel_stance"] = RewardTermCfg(
      func=mdp.go2_line_velocity_stance_penalty,
      weight=-1.0,
      params={"command_name": "twist"},
    )
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(
      func=mdp.go2_ang_vel_xy_penalty,
      weight=-0.2,
      params={"include_pitch": False},
    )
    cfg.rewards["dof_vel"] = RewardTermCfg(
      func=mdp.go2_joint_velocity_penalty,
      weight=-0.001,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["dof_vel_limits"] = RewardTermCfg(
      func=mdp.go2_joint_velocity_limit_penalty,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["torques"] = RewardTermCfg(
      func=mdp.go2_torque_penalty,
      weight=-0.0001,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["feet_contact_forces"] = RewardTermCfg(
      func=mdp.go2_foot_contact_force_penalty,
      weight=-0.1,
      params={"sensor_name": "feet_ground_contact", "max_contact_force": 150.0},
    )
    # Backflip follows the source scale table; disable inherited generic
    # orientation shaping and restore its stronger joint-limit/action-rate
    # terms.
    cfg.rewards["inverted_orientation"].weight = 0.0
    cfg.rewards["dof_pos_limits"].weight = -10.0
    cfg.rewards["action_rate_l2"].weight = -0.01
    # Backflip has explicit pre-trigger/post-landing shaping instead of the
    # generic velocity-task tracking terms.
    for name in ("track_linear_velocity", "track_angular_velocity", "pose", "upright", "foot_slip", "soft_landing"):
      cfg.rewards[name].weight = 0.0
    cfg.terminations["reset_height"] = TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.1},
    )
  return cfg


def unitree_go2_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_special_action_env_cfg("jump", play)


def unitree_go2_spring_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_special_action_env_cfg("spring_jump", play)


def unitree_go2_backflip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_special_action_env_cfg("backflip", play)


def unitree_go2_handstand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_special_action_env_cfg("handstand", play)


def unitree_go2_leggedstand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_special_action_env_cfg("leggedstand", play)


def _go2_custom_algorithm_env_cfg(
  kind: Literal[
    "dreamwaq",
    "amp_dreamwaq",
    "cts",
    "amp_cts",
    "amp_ts",
    "amp_ts_student",
    "ts",
    "ts_student",
  ],
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Configure source-shaped observations for a custom algorithm task."""
  cfg = unitree_go2_rough_env_cfg(play=play)
  # The custom source classes use multiplicative joint reset ranges. CTS is
  # narrower (0.9–1.1); DreamWaQ/AMP/TS use 0.5–1.5.
  cfg.events["reset_robot_joints"].func = mdp.reset_joints_by_scale
  cfg.events["reset_robot_joints"].params.pop("position_range", None)
  cfg.events["reset_robot_joints"].params["scale_range"] = (
    (0.9, 1.1) if kind == "cts" else (0.5, 1.5)
  )
  cfg.events["reset_base"].params["pose_range"].update(
    {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
  )
  # Go2_Cts/DreamWaQ and the TS base task all randomize root linear and
  # angular velocity in [-0.5, 0.5] at episode reset.
  cfg.events["reset_base"].params["velocity_range"] = {
    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
    "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
  }
  # Source custom Go2 tasks use five terrain families with proportions
  # [smooth slope, rough slope, stairs up, stairs down, discrete].  Keep the
  # public mjlab rough preset untouched, but make all eight migrated custom
  # tasks train on the source distribution.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=70.0 if kind in ("cts", "amp_cts") else 25.0,
    num_rows=10,
    num_cols=20,
    curriculum=True,
    sub_terrains={
      "smooth_slope": hf_pyramid_slope(
        proportion=0.15, slope_range=(0.0, 0.4), platform_width=3.0, border_width=1.0
      ),
      "rough_slope": random_rough(
        proportion=0.15, noise_range=(0.02, 0.10), noise_step=0.02, border_width=1.0
      ),
      "stairs_up": pyramid_stairs(
        proportion=0.30, step_height_range=(0.0, 0.1), step_width=0.31,
        platform_width=3.0, border_width=1.0,
      ),
      "stairs_down": pyramid_stairs_inv(
        proportion=0.30, step_height_range=(0.0, 0.1), step_width=0.31,
        platform_width=3.0, border_width=1.0,
      ),
      "discrete": discrete_obstacles(
        proportion=0.10, obstacle_width_range=(0.3, 1.0),
        obstacle_height_range=(0.05, 0.25), num_obstacles=40, border_width=1.0,
      ),
    },
    add_lights=True,
  )
  cfg.clip_rewards_to_positive = kind == "amp_cts"
  # The source CTS/DreamWaQ/TS families all enable HIMLoco-style random
  # action switching inside the four physics substeps.  Keep the public action
  # dimension and default offset unchanged while replacing only the action
  # term implementation.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  cfg.actions["joint_pos"] = mdp.Go2DelayedJointPositionActionCfg(
    entity_name=joint_pos_action.entity_name,
    actuator_names=joint_pos_action.actuator_names,
    scale=joint_pos_action.scale,
    offset=joint_pos_action.offset,
    clip=joint_pos_action.clip,
    preserve_order=joint_pos_action.preserve_order,
    use_default_offset=joint_pos_action.use_default_offset,
    delay=True,
  )
  if kind == "amp_cts":
    # AMP-CTS is the one source variant with the mirrored abduction preload.
    cfg.scene.entities["robot"].init_state.joint_pos.update(
      {
        "FL_hip_joint": -0.1,
        "FR_hip_joint": 0.1,
        "RL_hip_joint": -0.1,
        "RR_hip_joint": 0.1,
      }
    )
  # Source domain-randomization cadence samples calibration/model fields at
  # environment startup, with an interval push disturbance.  The generic rough
  # config already owns the event terms; these overrides align their ranges.
  # Source samples calibration and model parameters once when each simulator
  # environment is created; only the robot state/command is reset per episode.
  cfg.events["encoder_bias"].mode = "startup"
  cfg.events["encoder_bias"].params["bias_range"] = (-0.035, 0.035)
  custom_friction_range = {
    "cts": (0.2, 1.0),
    "amp_cts": (0.2, 1.0),
    "dreamwaq": (0.2, 1.25),
    "amp_dreamwaq": (0.2, 1.25),
    "amp_ts": (0.05, 3.0),
    "amp_ts_student": (0.05, 3.0),
    "ts": (0.05, 3.0),
    "ts_student": (0.05, 3.0),
  }[kind]
  _go2_source_geom_friction(cfg, custom_friction_range)
  cfg.events["base_com"].mode = "startup"
  is_ts_family = kind in ("ts", "ts_student", "amp_ts", "amp_ts_student")
  com_extent = 0.1 if is_ts_family else 0.05
  cfg.events["base_com"].params["ranges"] = {
    0: (-com_extent, com_extent),
    1: (-com_extent, com_extent),
    2: (-com_extent, com_extent),
  }
  cfg.events["go2_base_mass"] = EventTermCfg(
    func=envs_mdp.dr.body_mass,
    mode="startup",
    params={
      "ranges": (-1.0, 5.0),
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "operation": "add",
    },
  )
  cfg.events["go2_link_mass"] = EventTermCfg(
    func=envs_mdp.dr.body_mass,
    mode="startup",
    params={
      "ranges": (0.8, 1.2),
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=(r"(?:FL|FR|RL|RR)_(?:hip|thigh|calf)",)
      ),
      "operation": "scale",
    },
  )
  if "push_robot" in cfg.events:
    push_interval = 8.0 if is_ts_family else 4.0
    push_linear = 1.0 if is_ts_family else 0.4
    push_angular = 1.0 if is_ts_family else 0.6
    cfg.events["push_robot"].interval_range_s = (push_interval, push_interval)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-push_linear, push_linear), "y": (-push_linear, push_linear),
      "z": (0.0, 0.0),
      "roll": (-push_angular, push_angular),
      "pitch": (-push_angular, push_angular),
      "yaw": (-push_angular, push_angular),
    }
  gain_min, gain_max = (0.8, 1.2) if is_ts_family else (0.9, 1.1)
  cfg.events["go2_pd_gains"] = EventTermCfg(
    func=envs_mdp.dr.pd_gains,
    mode="startup",
    params={
      "kp_range": (gain_min, gain_max),
      "kd_range": (gain_min, gain_max),
      "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
      "operation": "scale",
    },
  )
  torque_multiplier_range = (0.8, 1.2) if is_ts_family else (0.9, 1.1)
  cfg.events["go2_torque_multiplier"] = EventTermCfg(
    func=mdp.go2_torque_multiplier,
    mode="startup",
    params={
      "torque_multiplier_range": torque_multiplier_range,
      "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
    },
  )
  # Replace the generic velocity rewards with the source LeggedRobot kernels.
  # The term names stay stable for logging, while the functions now separate
  # planar tracking, vertical velocity, tilt and terrain-relative height.
  tracking_lin_weight = 1.0 if kind.startswith("amp") or kind.startswith("ts") else 1.5
  tracking_ang_weight = 0.5 if kind.startswith("amp") or kind.startswith("ts") else 0.75
  base_height_weight = {
    "dreamwaq": -5.0,
    "amp_dreamwaq": -5.0,
    "amp_ts": -5.0,
    "amp_ts_student": -1.0,
    "ts": -1.0,
    "ts_student": -1.0,
  }.get(kind, -2.0)
  torque_weight = {
    "amp_cts": -1.0e-5,
    "amp_dreamwaq": -1.0e-5,
    "amp_ts_student": -1.0e-5,
    "ts": -1.0e-5,
    "ts_student": -1.0e-5,
  }.get(kind, -1.0e-4)
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.go2_source_tracking_linear_velocity,
    weight=tracking_lin_weight,
    params={"command_name": "twist", "sigma": 0.25},
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.go2_source_tracking_angular_velocity,
    weight=tracking_ang_weight,
    params={"command_name": "twist", "sigma": 0.25},
  )
  cfg.rewards["upright"] = RewardTermCfg(
    func=mdp.go2_orientation_penalty,
    weight=-0.2,
    params={},
  )
  cfg.rewards["body_ang_vel"] = RewardTermCfg(
    func=mdp.go2_angular_velocity_xy_penalty,
    weight=-0.05,
    params={},
  )
  cfg.rewards["lin_vel_z"] = RewardTermCfg(
    func=mdp.go2_linear_velocity_z_penalty,
    # CTS/DreamWaQ use the legacy -1 coefficient; AMP/TS configs retain -2.
    weight=-2.0 if kind.startswith("amp") or kind.startswith("ts") else -1.0,
    params={},
  )
  cfg.rewards["base_height"] = RewardTermCfg(
    func=mdp.go2_base_height_penalty,
    weight=base_height_weight,
    params={
      "target_height": 0.35 if kind == "amp_dreamwaq" else 0.4,
      "sensor_name": "terrain_scan",
    },
  )
  cfg.rewards["torques"] = RewardTermCfg(
    func=envs_mdp.joint_torques_l2,
    weight=torque_weight,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
  )
  cfg.rewards["dof_acc"] = RewardTermCfg(
    func=mdp.go2_joint_acceleration_penalty,
    weight=-2.5e-7,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["action_smoothness"] = RewardTermCfg(
    func=mdp.go2_action_smoothness_penalty,
    # TS configs inherit the source base scale ``smoothness=-0.005``;
    # non-TS custom configs override it to ``action_smoothness=-0.01``.
    weight=-0.005 if is_ts_family else -0.01,
    params={},
  )
  cfg.rewards["dof_pos_limits"].weight = (
    -2.0 if kind in ("cts", "dreamwaq", "amp_cts", "amp_dreamwaq", "amp_ts") else 0.0
  )
  cfg.rewards["action_rate_l2"].weight = -0.01
  # Source custom tasks use one aggregate collision term.  Remove the rough
  # baseline's three per-sensor terms to avoid applying the penalty twice.
  for name in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards.pop(name, None)
  cfg.rewards["collision"] = RewardTermCfg(
    func=mdp.go2_special_collision_penalty,
    weight=-0.5 if kind == "ts" else -1.0,
    params={
      "sensor_names": (
        "self_collision", "thigh_ground_touch", "shank_ground_touch", "trunk_ground_touch",
      ),
    },
  )
  cfg.rewards["foot_clearance"].weight = -0.5
  cfg.rewards["foot_swing_height"].weight = 0.0
  cfg.rewards["foot_slip"].weight = 0.0
  cfg.rewards["soft_landing"].weight = 0.0
  cfg.rewards["pose"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["stumble"] = RewardTermCfg(
    func=mdp.go2_stumble_penalty,
    weight=-0.5,
    params={"sensor_name": "feet_ground_contact"},
  )
  if kind == "amp_cts":
    cfg.rewards["hip_pos"] = RewardTermCfg(
      func=mdp.go2_hip_position_squared_penalty,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_hip_joint",))},
    )
  if kind == "amp_dreamwaq":
    cfg.rewards["rear_hip_limit"] = RewardTermCfg(
      func=mdp.go2_rear_hip_limit_penalty,
      weight=-1.0,
      params={
        "limit": 0.4,
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(r"(?:RL|RR)_hip_joint",), preserve_order=True
        ),
      },
    )
  cfg.rewards["air_time"].weight = 1.0 if kind in ("amp_cts", "amp_ts_student", "ts_student") else 0.0
  cfg.rewards["base_height"].params["target_height"] = 0.35 if kind == "amp_dreamwaq" else 0.4
  # Keep the per-world contact budget bounded for large parallel runs.  The
  # source Isaac-Gym tasks use a fixed contact buffer; an unbounded MuJoCo
  # Warp heuristic combined with the five rough-terrain variants can allocate
  # several GiB before the first step at 1024 environments.  SAP segmented
  # broadphase preserves the contact candidates while avoiding that startup
  # allocation, and 35 contacts/world matches the standard Go2 velocity
  # budget (users can still override this through ``--env.sim.nconmax``).
  cfg.sim.broadphase = "sap_segmented"
  cfg.sim.nconmax = 35
  custom_noise = _go2_source_noise_cfg(
    command_first=True,
    dof_pos_noise=0.03 if kind in ("amp_ts_student", "ts_student") else (0.02 if kind == "dreamwaq" else 0.01),
    ang_vel_noise=0.3 if kind in ("amp_ts_student", "ts_student") else 0.2,
  )
  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      "source_stand": ObservationTermCfg(
        func=mdp.go2_source_stand_observation,
        params={"command_name": "twist", "command_first": True},
        noise=custom_noise,
        history_length=1,
        flatten_history_dim=True,
      )
    },
    concatenate_terms=True,
    enable_corruption=not play,
  )
  cfg.observations["history"] = ObservationGroupCfg(
    terms={
      "source_history": ObservationTermCfg(
        func=mdp.go2_source_stand_observation,
        params={"command_name": "twist", "command_first": True},
        noise=custom_noise,
        history_length=5,
        flatten_history_dim=True,
      )
    },
    concatenate_terms=True,
    enable_corruption=not play,
  )
  cfg.observations["terrain"] = ObservationGroupCfg(
    terms={
      "terrain_scan": ObservationTermCfg(
        func=envs_mdp.height_scan,
        params={"sensor_name": "terrain_scan"},
        scale=1 / 5.0,
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  if kind.startswith("amp"):
    cfg.observations["amp"] = ObservationGroupCfg(
      terms={
        "amp_state": ObservationTermCfg(
          func=mdp.go2_amp_observation,
          params={"sensor_name": "terrain_scan"},
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )

  command_resample = 8.0 if kind == "cts" else 10.0
  if kind == "amp_cts":
    command_resample = 10.0
  cfg.commands["twist"].ranges.lin_vel_x = (-1.0, 1.0)
  # Keep the per-variant source command ranges.  TS teacher/student configs
  # intentionally use heading/yaw bounds of ±pi, while the other custom
  # variants use the narrower ±1 rad/s yaw range.
  lin_y_range = {
    "cts": (-1.0, 1.0),
    "dreamwaq": (-1.0, 1.0),
    "amp_cts": (-0.65, 0.65),
    "amp_dreamwaq": (-0.6, 0.6),
    "amp_ts": (-0.6, 0.6),
    "ts": (-0.6, 0.6),
    "amp_ts_student": (-0.5, 0.5),
    "ts_student": (-0.5, 0.5),
  }[kind]
  yaw_range = (-math.pi, math.pi) if is_ts_family else (-1.0, 1.0)
  # Source custom configs set heading_command=True globally; they do not use
  # mjlab's mixed heading/standing/forward environment fractions.
  cfg.commands["twist"].heading_command = True
  cfg.commands["twist"].ranges.heading = (-math.pi, math.pi)
  cfg.commands["twist"].rel_heading_envs = 1.0
  cfg.commands["twist"].rel_standing_envs = 0.0
  cfg.commands["twist"].rel_forward_envs = 0.0
  cfg.commands["twist"].source_min_lin_norm = 0.1 if kind == "cts" else 0.2
  cfg.commands["twist"].source_zero_command_prob = (
    0.05 if kind in ("amp_ts", "amp_ts_student", "ts", "ts_student") else 0.0
  )
  cfg.commands["twist"].source_zero_xy_prob = (
    0.05 if kind in ("amp_ts", "amp_ts_student", "ts", "ts_student") else 0.0
  )
  cfg.commands["twist"].ranges.lin_vel_y = lin_y_range
  cfg.commands["twist"].ranges.ang_vel_z = yaw_range
  cfg.commands["twist"].resampling_time_range = (command_resample, command_resample)
  if kind in ("amp_ts_student", "ts_student"):
    # Source student configs set commands.curriculum=False.
    cfg.curriculum.pop("command_vel", None)

  if kind in ("dreamwaq", "amp_dreamwaq"):
    cfg.observations["explicit"] = ObservationGroupCfg(
      terms={
        "dreamwaq_velocity": ObservationTermCfg(
          func=mdp.go2_dreamwaq_velocity_target,
          params={},
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "dreamwaq_privileged": ObservationTermCfg(
          func=mdp.go2_source_dreamwaq_privileged_observation,
          params={"command_name": "twist", "sensor_name": "terrain_scan"},
          history_length=3,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.observations["privileged"] = ObservationGroupCfg(
      terms={
        "dreamwaq_privileged_frame": ObservationTermCfg(
          func=mdp.go2_source_dreamwaq_privileged_observation,
          params={"command_name": "twist", "sensor_name": "terrain_scan"},
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
  elif kind in ("cts", "amp_cts"):
    # Source CTS uses a fixed 3:1 teacher/student environment split.  The mask
    # is kept in its own observation group so it does not alter the 45-D actor
    # contract consumed by checkpoints and deployment wrappers.
    cfg.observations["teacher_mask"] = ObservationGroupCfg(
      terms={"cts_teacher_mask": ObservationTermCfg(func=mdp.go2_cts_teacher_mask)},
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.observations["privileged"] = ObservationGroupCfg(
      terms={
        "cts_privileged": ObservationTermCfg(
          func=mdp.go2_source_cts_privileged_observation,
          params={"sensor_name": "terrain_scan", "contact_sensor_name": "feet_ground_contact"},
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "cts_critic": ObservationTermCfg(
          func=mdp.go2_source_cts_critic_observation,
          params={
            "sensor_name": "terrain_scan",
            "contact_sensor_name": "feet_ground_contact",
            "include_lin_vel": kind == "amp_cts",
          },
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
  else:
    cfg.observations["privileged"] = ObservationGroupCfg(
      terms={
        # The TS teacher encoder consumes the source 74-D
        # domain-randomization/contact buffer.  Keep the full 309-D value
        # observation in the separate ``critic`` group below.
        "ts_teacher_privileged": ObservationTermCfg(
          func=mdp.go2_source_ts_privileged_observation,
          params={"contact_sensor_name": "feet_ground_contact"},
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "ts_critic": ObservationTermCfg(
          func=mdp.go2_source_ts_critic_observation,
          params={"sensor_name": "terrain_scan", "contact_sensor_name": "feet_ground_contact"},
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
  return cfg


def unitree_go2_dreamwaq_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("dreamwaq", play)


def unitree_go2_amp_dreamwaq_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("amp_dreamwaq", play)


def unitree_go2_cts_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("cts", play)


def unitree_go2_amp_cts_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("amp_cts", play)


def unitree_go2_amp_ts_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("amp_ts", play)


def unitree_go2_amp_ts_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("amp_ts_student", play)


def unitree_go2_ts_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("ts", play)


def unitree_go2_ts_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("ts_student", play)
