"""Native mjlab configuration for source-parity Go2 tasks."""

from copy import deepcopy

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from lloco.assets.robots import get_go2_robot_cfg
from lloco.tasks.rl import make_ppo_runner_cfg
from lloco.tasks.velocity import PROFILES, make_flat_env_cfg

from . import mdp
from .mdp.observations import JOINT_NAMES
from .profiles import (
  JUMP,
  REAR_STAND,
  TROT,
  JumpProfile,
  RearStandProfile,
  TrotProfile,
)

_GO2_VELOCITY_PROFILE = next(
  profile for profile in PROFILES if profile.task_name == "Go2"
)
_FEET_SENSOR = "feet_ground_contact"
_PENALIZED_SENSOR = "thigh_calf_ground_contact"
_BASE_SENSOR = "base_ground_contact"


def _trot_robot_cfg():
  # The public asset factory currently reuses its InitialStateCfg.  Copy the
  # whole entity locally so skill overrides cannot mutate Flat/Rough.
  cfg = deepcopy(get_go2_robot_cfg())
  cfg.init_state.pos = (0.0, 0.0, 0.42)
  cfg.init_state.joint_pos = {
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    ".*thigh_joint": 0.8,
    ".*calf_joint": -1.5,
  }
  cfg.articulation = EntityArticulationInfoCfg(
    actuators=(
      IdealPdActuatorCfg(
        target_names_expr=(".*hip_joint",),
        stiffness=20.0,
        damping=0.5,
        effort_limit=23.7,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*thigh_joint",),
        stiffness=20.0,
        damping=0.5,
        effort_limit=23.7,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*calf_joint",),
        stiffness=20.0,
        damping=0.5,
        effort_limit=35.55,
        armature=0.0,
      ),
    ),
    soft_joint_pos_limit_factor=0.9,
  )
  return cfg


def _jump_robot_cfg():
  cfg = deepcopy(get_go2_robot_cfg())
  cfg.init_state.pos = (0.0, 0.0, 0.42)
  cfg.init_state.joint_pos = {
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    ".*calf_joint": -1.5,
  }
  cfg.articulation = EntityArticulationInfoCfg(
    actuators=(
      IdealPdActuatorCfg(
        target_names_expr=(".*hip_joint",),
        stiffness=20.0,
        damping=0.5,
        effort_limit=23.7,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*thigh_joint",),
        stiffness=20.0,
        damping=0.5,
        effort_limit=23.7,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*calf_joint",),
        stiffness=20.0,
        damping=0.5,
        effort_limit=35.55,
        armature=0.0,
      ),
    ),
    soft_joint_pos_limit_factor=0.9,
  )
  return cfg


def _rear_stand_robot_cfg():
  cfg = deepcopy(get_go2_robot_cfg())
  cfg.init_state.pos = (0.0, 0.0, 0.42)
  cfg.init_state.joint_pos = {
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    ".*calf_joint": -1.5,
  }
  cfg.articulation = EntityArticulationInfoCfg(
    actuators=(
      IdealPdActuatorCfg(
        target_names_expr=(".*hip_joint",),
        stiffness=40.0,
        damping=1.0,
        effort_limit=21.33,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*thigh_joint",),
        stiffness=40.0,
        damping=1.0,
        effort_limit=21.33,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*calf_joint",),
        stiffness=40.0,
        damping=1.0,
        effort_limit=31.995,
        armature=0.0,
      ),
    ),
    soft_joint_pos_limit_factor=0.9,
  )
  return cfg


def _replace_sensors(cfg: ManagerBasedRlEnvCfg) -> None:
  terrain = ContactMatch(mode="body", pattern="terrain")
  feet = ContactSensorCfg(
    name=_FEET_SENSOR,
    primary=ContactMatch(
      mode="geom",
      pattern=(
        "FL_foot_collision",
        "FR_foot_collision",
        "RL_foot_collision",
        "RR_foot_collision",
      ),
      entity="robot",
    ),
    secondary=terrain,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  penalized = ContactSensorCfg(
    name=_PENALIZED_SENSOR,
    primary=ContactMatch(
      mode="geom", pattern=r".*_(thigh|calf)_collision", entity="robot"
    ),
    secondary=terrain,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  base = ContactSensorCfg(
    name=_BASE_SENSOR,
    primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
    secondary=terrain,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (feet, penalized, base)


def _replace_rear_stand_sensors(cfg: ManagerBasedRlEnvCfg) -> None:
  # Source asset.penalize_contacts_on contains only thigh and calf. Base contact
  # is handled separately as a termination and hip contact is not penalized.
  _replace_sensors(cfg)


def _trot_observations(cfg: ManagerBasedRlEnvCfg, profile: TrotProfile) -> None:
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "history": ObservationTermCfg(
          func=mdp.observations.TrotActorHistory,
          params={
            "command_name": "twist",
            "cycle_time": profile.cycle_time,
            "add_noise": True,
          },
          clip=(-100.0, 100.0),
        )
      },
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "history": ObservationTermCfg(
          func=mdp.observations.TrotCriticHistory,
          params={
            "command_name": "twist",
            "sensor_name": _FEET_SENSOR,
            "cycle_time": profile.cycle_time,
          },
          clip=(-100.0, 100.0),
        )
      },
      enable_corruption=False,
    ),
  }


def _trot_rewards(cfg: ManagerBasedRlEnvCfg, profile: TrotProfile) -> None:
  gait = {
    "sensor_name": _FEET_SENSOR,
    "command_name": "twist",
    "cycle_time": profile.cycle_time,
  }
  cfg.rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=mdp.rewards.tracking_lin_vel,
      weight=2.0,
      params={**gait, "sigma": 0.25},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=mdp.rewards.tracking_ang_vel,
      weight=2.0,
      params={**gait, "sigma": 0.25},
    ),
    "lin_vel_z": RewardTermCfg(func=mdp.rewards.lin_vel_z, weight=-2.0),
    "ang_vel_xy": RewardTermCfg(func=mdp.rewards.ang_vel_xy, weight=-0.05),
    "orientation": RewardTermCfg(func=mdp.rewards.orientation, weight=-2.0),
    "torques": RewardTermCfg(func=mdp.rewards.torques, weight=-0.0001),
    "dof_acc": RewardTermCfg(func=mdp.rewards.DofAcceleration, weight=-2.5e-7),
    "collision": RewardTermCfg(
      func=mdp.rewards.collision,
      weight=-1.0,
      params={"sensor_name": _PENALIZED_SENSOR},
    ),
    "action_rate": RewardTermCfg(func=mdp.rewards.action_rate, weight=-0.01),
    "stand_still": RewardTermCfg(
      func=mdp.rewards.stand_still,
      weight=-1.0,
      params={"command_name": "twist"},
    ),
    "base_height": RewardTermCfg(
      func=mdp.rewards.base_height,
      weight=-5.0,
      params={"target_height": profile.base_height_target},
    ),
    "trot": RewardTermCfg(func=mdp.rewards.trot, weight=0.8, params=gait),
    "feet_clearance": RewardTermCfg(
      func=mdp.rewards.feet_clearance,
      weight=0.1,
      params={
        "command_name": "twist",
        "cycle_time": profile.cycle_time,
        "target_foot_height": profile.target_foot_height,
      },
    ),
    "default_hip_pos": RewardTermCfg(func=mdp.rewards.default_hip_pos, weight=-0.2),
    "default_pos": RewardTermCfg(func=mdp.rewards.default_pos, weight=-0.1),
    "contact_without_command": RewardTermCfg(
      func=mdp.rewards.contact_without_command,
      weight=1.0,
      params={"sensor_name": _FEET_SENSOR, "command_name": "twist"},
    ),
  }


def _trot_events(cfg: ManagerBasedRlEnvCfg) -> None:
  all_actuators = SceneEntityCfg("robot", actuator_names=[".*"])
  cfg.events = {
    "reset_base": EventTermCfg(
      func=env_mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "velocity_range": {}},
    ),
    "reset_robot_joints": EventTermCfg(
      func=env_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=JOINT_NAMES, preserve_order=True
        ),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.events.overwrite_root_velocity,
      mode="interval",
      interval_range_s=(4.0, 4.0),
      is_global_time=True,
      params={"max_push_vel_xy": 0.4, "max_push_ang_vel": 0.6},
    ),
    "friction": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(r".*_collision",)),
        "ranges": (0.2, 1.2),
        "operation": "abs",
        "shared_random": True,
      },
    ),
    "base_mass": EventTermCfg(
      func=dr.body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "ranges": (-1.0, 2.0),
        "operation": "add",
      },
    ),
    "link_mass": EventTermCfg(
      func=dr.body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(r"(FL|FR|RL|RR)_.*",)),
        "ranges": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "base_com": EventTermCfg(
      func=dr.body_com_offset,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "ranges": {0: (-0.03, 0.03), 1: (-0.03, 0.03), 2: (-0.03, 0.03)},
        "operation": "add",
      },
    ),
    "pd_gains": EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": all_actuators,
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "motor_zero_offset": EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={"bias_range": (-0.035, 0.035)},
    ),
  }


def make_trot_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the first migrated task from its actual registered source config."""
  profile = TROT
  cfg = make_flat_env_cfg(_GO2_VELOCITY_PROFILE, play=False)
  cfg.scene.entities = {"robot": _trot_robot_cfg()}
  cfg.scene.num_envs = profile.num_envs
  cfg.episode_length_s = profile.episode_length_s
  cfg.decimation = profile.decimation
  cfg.sim.mujoco.timestep = profile.physics_dt
  cfg.scale_rewards_by_dt = True
  cfg.metrics = {}
  cfg.recorders = {}
  _replace_sensors(cfg)
  cfg.actions = {
    "joint_pos": mdp.actions.EpisodeDelayedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=JOINT_NAMES,
      preserve_order=True,
      scale=profile.action_scale,
      use_default_offset=True,
      delay_min_lag=1,
      delay_max_lag=3,
    )
  }
  cfg.commands = {
    "twist": mdp.commands.TrotVelocityCommandCfg(
      resampling_time_range=(5.0, 5.0),
      debug_vis=True,
      entity_name="robot",
      heading_command=False,
      rel_standing_envs=0.0,
      rel_heading_envs=0.0,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
        heading=None,
      ),
    )
  }
  _trot_observations(cfg, profile)
  _trot_rewards(cfg, profile)
  _trot_events(cfg)
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "base_contact": TerminationTermCfg(
      func=mdp.terminations.base_contact,
      params={"sensor_name": _BASE_SENSOR, "force_threshold": 1.0},
    ),
  }
  cfg.curriculum = {
    "command_velocity": CurriculumTermCfg(
      func=mdp.curriculums.source_trot_command_curriculum,
      params={"command_name": "twist", "max_curriculum": 2.0},
    )
  }
  if play:
    cfg = deepcopy(cfg)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["actor"].terms["history"].params["add_noise"] = False
    cfg.events.pop("push_robot")
    cfg.curriculum = {}
  return cfg


def make_trot_runner_cfg():
  cfg = make_ppo_runner_cfg(
    TROT.experiment_name,
    max_iterations=15_000,
    save_interval=100,
  )
  cfg.seed = 1
  cfg.clip_actions = 100.0
  cfg.actor.obs_normalization = False
  cfg.critic.obs_normalization = False
  cfg.algorithm.learning_rate = 1.0e-5
  return cfg


def _jump_observations(cfg: ManagerBasedRlEnvCfg, profile: JumpProfile) -> None:
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "history": ObservationTermCfg(
          func=mdp.observations.JumpActorHistory,
          params={
            "command_name": "twist",
            "cycle_time": profile.cycle_time,
            "add_noise": True,
          },
          clip=(-100.0, 100.0),
        )
      },
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "history": ObservationTermCfg(
          func=mdp.observations.JumpCriticHistory,
          params={
            "command_name": "twist",
            "sensor_name": _FEET_SENSOR,
            "cycle_time": profile.cycle_time,
          },
          clip=(-100.0, 100.0),
        )
      },
      enable_corruption=False,
    ),
  }


def _jump_rewards(cfg: ManagerBasedRlEnvCfg, profile: JumpProfile) -> None:
  cfg.rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=mdp.rewards.jump_tracking_lin_vel,
      weight=2.0,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=mdp.rewards.jump_tracking_ang_vel,
      weight=2.0,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "lin_vel_z": RewardTermCfg(func=mdp.rewards.jump_lin_vel_z, weight=0.05),
    "ang_vel_xy": RewardTermCfg(func=mdp.rewards.jump_ang_vel_xy, weight=0.2),
    "orientation": RewardTermCfg(func=mdp.rewards.jump_orientation, weight=0.6),
    "torques": RewardTermCfg(func=mdp.rewards.absolute_torques, weight=-0.0002),
    "dof_acc": RewardTermCfg(func=mdp.rewards.JointVelocityDifference, weight=-5.5e-4),
    "base_height": RewardTermCfg(
      func=mdp.rewards.jump_base_height,
      weight=1.0,
      params={
        "command_name": "twist",
        "target_height": profile.base_height_target,
      },
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.rewards.FeetAirTime,
      weight=1.0,
      params={"sensor_name": _FEET_SENSOR, "command_name": "twist"},
    ),
    "collision": RewardTermCfg(
      func=mdp.rewards.collision,
      weight=-1.0,
      params={"sensor_name": _PENALIZED_SENSOR},
    ),
    "action_rate": RewardTermCfg(func=mdp.rewards.action_rate, weight=-0.01),
    "stand_still": RewardTermCfg(
      func=mdp.rewards.stand_still,
      weight=-1.0,
      params={"command_name": "twist"},
    ),
    "default_pos": RewardTermCfg(func=mdp.rewards.default_pos, weight=-0.1),
    "default_hip_pos": RewardTermCfg(func=mdp.rewards.jump_default_hip_pos, weight=0.3),
    "feet_contact_forces": RewardTermCfg(
      func=mdp.rewards.feet_contact_forces,
      weight=-0.01,
      params={"sensor_name": _FEET_SENSOR, "max_contact_force": 100.0},
    ),
    "jump": RewardTermCfg(
      func=mdp.rewards.jump_contact_match,
      weight=2.0,
      params={
        "sensor_name": _FEET_SENSOR,
        "command_name": "twist",
        "cycle_time": profile.cycle_time,
      },
    ),
    "feet_clearance": RewardTermCfg(
      func=mdp.rewards.jump_feet_clearance,
      weight=0.5,
      params={
        "command_name": "twist",
        "cycle_time": profile.cycle_time,
        "max_height": profile.target_foot_height,
      },
    ),
    "contact_without_command": RewardTermCfg(
      func=mdp.rewards.contact_without_command,
      weight=1.0,
      params={"sensor_name": _FEET_SENSOR, "command_name": "twist"},
    ),
  }


def _jump_events(cfg: ManagerBasedRlEnvCfg) -> None:
  _trot_events(cfg)
  cfg.events["friction"] = EventTermCfg(
    func=mdp.events.source_friction_buckets,
    mode="startup",
    params={
      "low": 0.2,
      "high": 1.2,
      "num_buckets": 256,
      "entity_name": "robot",
    },
  )
  cfg.events["base_mass"].params["ranges"] = (-1.0, 1.0)


def make_jump_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the registered Go2 Jump task from its source implementation."""
  profile = JUMP
  cfg = make_flat_env_cfg(_GO2_VELOCITY_PROFILE, play=False)
  cfg.scene.entities = {"robot": _jump_robot_cfg()}
  cfg.scene.num_envs = profile.num_envs
  cfg.episode_length_s = profile.episode_length_s
  cfg.decimation = profile.decimation
  cfg.sim.mujoco.timestep = profile.physics_dt
  cfg.scale_rewards_by_dt = True
  cfg.metrics = {}
  cfg.recorders = {}
  _replace_sensors(cfg)
  cfg.actions = {
    "joint_pos": mdp.actions.EpisodeDelayedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=JOINT_NAMES,
      preserve_order=True,
      scale=profile.action_scale,
      use_default_offset=True,
      delay_min_lag=1,
      delay_max_lag=3,
    )
  }
  cfg.commands = {
    "twist": mdp.commands.TrotVelocityCommandCfg(
      resampling_time_range=(5.0, 5.0),
      debug_vis=True,
      entity_name="robot",
      heading_command=False,
      rel_standing_envs=0.0,
      rel_heading_envs=0.0,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
        heading=None,
      ),
    )
  }
  _jump_observations(cfg, profile)
  _jump_rewards(cfg, profile)
  _jump_events(cfg)
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "base_contact": TerminationTermCfg(
      func=mdp.terminations.base_contact,
      params={"sensor_name": _BASE_SENSOR, "force_threshold": 1.0},
    ),
  }
  cfg.curriculum = {
    "command_velocity": CurriculumTermCfg(
      func=mdp.curriculums.source_trot_command_curriculum,
      params={"command_name": "twist", "max_curriculum": 2.0},
    )
  }
  if play:
    cfg = deepcopy(cfg)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["actor"].terms["history"].params["add_noise"] = False
    cfg.events.pop("push_robot")
    cfg.curriculum = {}
  return cfg


def make_jump_runner_cfg():
  cfg = make_ppo_runner_cfg(
    JUMP.experiment_name,
    max_iterations=15_000,
    save_interval=100,
  )
  cfg.seed = 1
  cfg.clip_actions = 100.0
  cfg.actor.obs_normalization = False
  cfg.critic.obs_normalization = False
  cfg.algorithm.learning_rate = 1.0e-4
  return cfg


def _rear_stand_observations(
  cfg: ManagerBasedRlEnvCfg, profile: RearStandProfile
) -> None:
  del profile
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(
          func=mdp.observations.RearStandActorObservation,
          params={"command_name": "twist", "add_noise": True},
          clip=(-100.0, 100.0),
        )
      },
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(
          func=mdp.observations.RearStandCriticObservation,
          params={"sensor_name": _FEET_SENSOR},
          clip=(-100.0, 100.0),
        )
      },
      enable_corruption=False,
    ),
  }


def _rear_stand_rewards(cfg: ManagerBasedRlEnvCfg, profile: RearStandProfile) -> None:
  cfg.rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=mdp.rewards.rear_stand_tracking_lin_vel,
      weight=2.5,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=mdp.rewards.rear_stand_tracking_ang_vel,
      weight=2.5,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "lin_vel_z": RewardTermCfg(func=mdp.rewards.rear_stand_lin_vel_z, weight=0.2),
    "ang_vel_xy": RewardTermCfg(func=mdp.rewards.rear_stand_ang_vel_xy, weight=0.2),
    "rear_stand_orientation": RewardTermCfg(
      func=mdp.rewards.rear_stand_orientation, weight=-1.0
    ),
    "torques": RewardTermCfg(func=mdp.rewards.absolute_torques, weight=-0.0002),
    "dof_acc": RewardTermCfg(func=mdp.rewards.DofAcceleration, weight=-2.5e-7),
    "base_height": RewardTermCfg(
      func=mdp.rewards.rear_stand_base_height,
      weight=1.5,
      params={"target_height": profile.base_height_target},
    ),
    "rear_stand_feet_on_air": RewardTermCfg(
      func=mdp.rewards.rear_stand_front_feet_air,
      weight=0.4,
      params={"sensor_name": _FEET_SENSOR},
    ),
    "collision": RewardTermCfg(
      func=mdp.rewards.collision,
      weight=-2.0,
      params={"sensor_name": _PENALIZED_SENSOR},
    ),
    "action_rate": RewardTermCfg(func=mdp.rewards.action_rate, weight=-0.05),
    "default_pos": RewardTermCfg(func=mdp.rewards.rear_stand_default_pos, weight=-0.1),
    "default_hip_pos": RewardTermCfg(
      func=mdp.rewards.rear_stand_default_hip_pos, weight=-0.1
    ),
    "feet_clearance": RewardTermCfg(
      func=mdp.rewards.rear_stand_feet_clearance,
      weight=0.4,
      params={
        "cycle_time": profile.cycle_time,
        "target_foot_height": profile.target_foot_height,
      },
    ),
    "ang_xz": RewardTermCfg(func=mdp.rewards.rear_stand_roll, weight=-0.5),
    "contact": RewardTermCfg(
      func=mdp.rewards.rear_stand_rear_contact,
      weight=0.3,
      params={"sensor_name": _FEET_SENSOR},
    ),
    "symmetric_joints": RewardTermCfg(
      func=mdp.rewards.rear_stand_symmetric_joints, weight=-0.1
    ),
    "orientation_symmetry": RewardTermCfg(
      func=mdp.rewards.rear_stand_orientation_symmetry, weight=-0.5
    ),
    "feet_height_symmetry": RewardTermCfg(
      func=mdp.rewards.rear_stand_feet_height_symmetry, weight=-0.2
    ),
    "rear_stand_feet_height_exp": RewardTermCfg(
      func=mdp.rewards.rear_stand_front_feet_height_exp, weight=5.0
    ),
    "default_pos_reward": RewardTermCfg(
      func=mdp.rewards.rear_stand_default_pos_reward, weight=0.5
    ),
    "dof_pos_limits": RewardTermCfg(
      func=mdp.rewards.rear_stand_dof_pos_limits, weight=-2.0
    ),
    "alive": RewardTermCfg(func=mdp.rewards.alive, weight=1.0),
  }


def _rear_stand_events(cfg: ManagerBasedRlEnvCfg) -> None:
  joints = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
  all_actuators = SceneEntityCfg("robot", actuator_names=[".*"])
  cfg.events = {
    "reset_base": EventTermCfg(
      func=env_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.5, 0.5),
          "roll": (-0.5, 0.5),
          "pitch": (-0.5, 0.5),
          "yaw": (-0.5, 0.5),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.events.reset_joints_by_scale,
      mode="reset",
      params={"scale_range": (0.5, 1.5), "entity_name": "robot"},
    ),
    "push_robot": EventTermCfg(
      func=mdp.events.overwrite_root_velocity,
      mode="interval",
      interval_range_s=(8.0, 8.0),
      is_global_time=True,
      params={"max_push_vel_xy": 0.4, "max_push_ang_vel": 0.6},
    ),
    "friction": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(r".*_collision",)),
        "ranges": (0.2, 1.2),
        "operation": "abs",
        "shared_random": True,
      },
    ),
    "restitution_label": EventTermCfg(
      func=mdp.events.sample_restitution_label,
      mode="startup",
      params={"low": 0.0, "high": 0.3},
    ),
    "base_mass": EventTermCfg(
      func=dr.body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "ranges": (-1.0, 2.0),
        "operation": "add",
      },
    ),
    "link_mass": EventTermCfg(
      func=dr.body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(r"(FL|FR|RL|RR)_.*",)),
        "ranges": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "base_com": EventTermCfg(
      func=dr.body_com_offset,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
        "operation": "add",
      },
    ),
    "pd_gains": EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": all_actuators,
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "motor_zero_offset": EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={"bias_range": (-0.035, 0.035), "asset_cfg": joints},
    ),
    "joint_friction": EventTermCfg(
      func=dr.joint_friction,
      mode="startup",
      params={
        "asset_cfg": joints,
        "ranges": (0.01, 0.1),
        "operation": "abs",
        "shared_random": True,
      },
    ),
    "joint_damping": EventTermCfg(
      func=dr.joint_damping,
      mode="startup",
      params={
        "asset_cfg": joints,
        "ranges": (0.0, 0.1),
        "operation": "abs",
        "shared_random": True,
      },
    ),
    "joint_armature": EventTermCfg(
      func=dr.joint_armature,
      mode="startup",
      params={
        "asset_cfg": joints,
        "ranges": (0.003, 0.08),
        "operation": "abs",
        "shared_random": True,
      },
    ),
  }


def make_rear_stand_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build Go2 Rear Stand from the Gym project's ``go2_handstand`` source."""
  profile = REAR_STAND
  cfg = make_flat_env_cfg(_GO2_VELOCITY_PROFILE, play=False)
  cfg.scene.entities = {"robot": _rear_stand_robot_cfg()}
  cfg.scene.num_envs = profile.num_envs
  cfg.episode_length_s = profile.episode_length_s
  cfg.decimation = profile.decimation
  cfg.sim.mujoco.timestep = profile.physics_dt
  cfg.scale_rewards_by_dt = True
  cfg.metrics = {}
  cfg.recorders = {}
  _replace_rear_stand_sensors(cfg)
  cfg.actions = {
    "joint_pos": mdp.actions.EpisodeDelayedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=JOINT_NAMES,
      preserve_order=True,
      scale=profile.action_scale,
      use_default_offset=True,
      delay_min_lag=0,
      delay_max_lag=3,
      delay_update_period=4,
    )
  }
  cfg.commands = {
    "twist": mdp.commands.RearStandVelocityCommandCfg(
      resampling_time_range=(10.0, 10.0),
      debug_vis=True,
      entity_name="robot",
      heading_command=True,
      rel_standing_envs=0.0,
      rel_heading_envs=1.0,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.2, 0.6),
        lin_vel_y=(-0.0, 0.0),
        ang_vel_z=(-0.4, 0.4),
        heading=(-3.14, 3.14),
      ),
    )
  }
  _rear_stand_observations(cfg, profile)
  _rear_stand_rewards(cfg, profile)
  _rear_stand_events(cfg)
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "base_contact": TerminationTermCfg(
      func=mdp.terminations.base_contact,
      params={"sensor_name": _BASE_SENSOR, "force_threshold": 1.0},
    ),
  }
  cfg.curriculum = {}
  if play:
    cfg = deepcopy(cfg)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["actor"].terms["frame"].params["add_noise"] = False
    cfg.events.pop("push_robot")
  return cfg


def make_rear_stand_runner_cfg():
  cfg = make_ppo_runner_cfg(
    REAR_STAND.experiment_name,
    max_iterations=15_000,
    save_interval=100,
    symmetry_cfg={
      "data_augmentation_func": (
        "lloco.tasks.go2_skills.mdp.symmetry:rear_stand_symmetry"
      ),
      "use_data_augmentation": False,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 1.0,
    },
  )
  cfg.seed = 1
  cfg.clip_actions = 100.0
  cfg.actor.obs_normalization = False
  cfg.critic.obs_normalization = False
  cfg.algorithm.learning_rate = 1.0e-3
  cfg.algorithm.class_name = (
    "lloco.tasks.go2_skills.mdp.symmetry:SourceSymmetricPPO"
  )
  return cfg
