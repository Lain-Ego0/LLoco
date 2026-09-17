"""Velocity environment construction shared by robot profiles."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from .velocity_types import VelocityRobotProfile


def _configure_height_sensors(
  cfg: ManagerBasedRlEnvCfg, profile: VelocityRobotProfile
) -> None:
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = profile.root_body
    elif sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in profile.foot_sites
      )
      radius = (
        profile.foot_radius
        if profile.foot_radius is not None
        else (0.04 if profile.kind == "quadruped" else 0.03)
      )
      samples = (
        profile.foot_sample_count
        if profile.foot_sample_count is not None
        else (4 if profile.kind == "quadruped" else 6)
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=radius, num_samples=samples)


def _contact_sensors(
  profile: VelocityRobotProfile,
) -> tuple[ContactSensorCfg, ContactSensorCfg]:
  primary_mode = "geom" if profile.kind == "quadruped" else "subtree"
  feet = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode=primary_mode,
      pattern=profile.foot_contact_pattern,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  if profile.kind == "humanoid":
    other = ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    )
  else:
    other = ContactSensorCfg(
      name="nonfoot_ground_touch",
      primary=ContactMatch(
        mode="geom",
        pattern=r".*_collision\d*$",
        exclude=profile.foot_geoms,
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    )
  return feet, other


def _configure_posture(
  cfg: ManagerBasedRlEnvCfg, profile: VelocityRobotProfile
) -> None:
  if profile.kind == "quadruped":
    cfg.rewards["pose"].params["std_standing"] = {
      r".*_(hip|thigh)_joint.*": 0.05,
      r".*_calf_joint.*": 0.1,
    }
    moving = {
      r".*_(hip|thigh)_joint.*": 0.3,
      r".*_calf_joint.*": 0.6,
    }
  else:
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    moving = {
      r".*hip_pitch.*": 0.3,
      r".*hip_(roll|yaw).*": 0.15,
      r".*knee.*": 0.35,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.1,
      r".*(waist|torso).*": 0.2,
      r".*shoulder.*": 0.15,
      r".*elbow.*": 0.15,
      r".*wrist.*": 0.3,
    }
  cfg.rewards["pose"].params["std_walking"] = moving
  cfg.rewards["pose"].params["std_running"] = moving


def make_rough_env_cfg(
  profile: VelocityRobotProfile, *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build a rough-terrain environment from a compact robot profile."""
  cfg = make_velocity_env_cfg()
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70 if profile.kind == "humanoid" else None
  cfg.scene.entities = {"robot": profile.robot_cfg()}

  _configure_height_sensors(cfg, profile)
  feet_sensor, other_sensor = _contact_sensors(profile)
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_sensor, other_sensor)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = profile.action_scale
  cfg.decimation = profile.decimation

  cfg.viewer.body_name = profile.viewer_body
  cfg.viewer.distance = profile.viewer_distance
  cfg.viewer.elevation = -10.0
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.viz.z_offset = profile.command_z_offset
  command.ranges.lin_vel_x = profile.command_ranges[0]
  command.ranges.lin_vel_y = profile.command_ranges[1]
  command.ranges.ang_vel_z = profile.command_ranges[2]
  if profile.command_velocity_stages is not None:
    cfg.curriculum["command_vel"].params["velocity_stages"] = list(
      profile.command_velocity_stages
    )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = profile.foot_geoms
  cfg.events["base_com"].params["asset_cfg"].body_names = (profile.viewer_body,)
  cfg.events["reset_base"].params["pose_range"]["z"] = profile.reset_height_range
  if profile.base_com_scale != 1.0:
    cfg.events["base_com"].params["ranges"] = {
      axis: (low * profile.base_com_scale, high * profile.base_com_scale)
      for axis, (low, high) in cfg.events["base_com"].params["ranges"].items()
    }
  _configure_posture(cfg, profile)
  cfg.rewards["upright"].params["asset_cfg"].body_names = (profile.viewer_body,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (profile.viewer_body,)
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = profile.foot_sites
  cfg.rewards["foot_clearance"].params["target_height"] = (
    profile.clearance_target_height
  )
  cfg.rewards["foot_swing_height"].params["target_height"] = (
    profile.clearance_target_height
  )

  if profile.kind == "humanoid":
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["self_collisions"] = RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": other_sensor.name, "force_threshold": 10.0},
    )
  else:
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": other_sensor.name, "force_threshold": 10.0},
    )

  if play:
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
    if (
      cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
    ):
      terrain = cfg.scene.terrain.terrain_generator
      terrain.curriculum = False
      terrain.num_cols = 5
      terrain.num_rows = 5
      terrain.border_width = 10.0
  return cfg


def make_flat_env_cfg(
  profile: VelocityRobotProfile, *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build a flat-ground variant of a robot's velocity task."""
  cfg = make_rough_env_cfg(profile, play=play)
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)
  if play:
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    command.ranges.lin_vel_x = profile.play_command_ranges[0]
    command.ranges.lin_vel_y = profile.play_command_ranges[1]
    command.ranges.ang_vel_z = profile.play_command_ranges[2]
  return cfg
