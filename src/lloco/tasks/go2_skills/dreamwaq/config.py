"""mjlab Manager-MDP configuration for source ``go2_dreamwaq``."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from lloco.tasks.rl import make_ppo_runner_cfg
from lloco.tasks.velocity import PROFILES, make_rough_env_cfg

from ..shared import actions as shared_actions
from ..shared import events as shared_events
from ..shared.contacts import JOINT_NAMES
from ..shared.robot import dreamwaq_robot_cfg
from ..shared.sensors import BASE_SENSOR, FEET_SENSOR, PENALIZED_SENSOR, replace_sensors
from ..shared.terminations import base_contact
from . import mdp
from .profile import DREAMWAQ

_GO2 = next(profile for profile in PROFILES if profile.task_name == "Go2")


def _observations(cfg: ManagerBasedRlEnvCfg) -> None:
  # The separate ``history`` and ``velocity`` groups are consumed only by the
  # local DreamWaQ PPO extension; actor/critic dimensions stay 45 and 783.
  cfg.observations = {
    "actor": ObservationGroupCfg(terms={"frame": ObservationTermCfg(func=mdp.observations.DreamActor, params={"command_name": "twist", "add_noise": True})}, enable_corruption=False),
    "critic": ObservationGroupCfg(terms={"history": ObservationTermCfg(func=mdp.observations.DreamCriticHistory, params={"command_name": "twist", "sensor_name": "terrain_scan"})}, enable_corruption=False),
    "history": ObservationGroupCfg(terms={"frames": ObservationTermCfg(func=mdp.observations.DreamActorHistory)}, enable_corruption=False),
    "velocity": ObservationGroupCfg(terms={"label": ObservationTermCfg(func=mdp.observations.base_linear_velocity)}, enable_corruption=False),
  }


def _events(cfg: ManagerBasedRlEnvCfg) -> None:
  all_actuators = SceneEntityCfg("robot", actuator_names=[".*"])
  cfg.events = {
    "reset_base": EventTermCfg(func=env_mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}, "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5), "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}}),
    "reset_robot_joints": EventTermCfg(func=shared_events.reset_joints_by_scale, mode="reset", params={"scale_range": (0.5, 1.5)}),
    "push_robot": EventTermCfg(func=shared_events.overwrite_root_velocity, mode="interval", interval_range_s=(4.0, 4.0), is_global_time=True, params={"max_push_vel_xy": 0.4, "max_push_ang_vel": 0.6}),
    "friction": EventTermCfg(func=shared_events.source_friction_buckets, mode="startup", params={"low": 0.2, "high": 1.25, "num_buckets": 64}),
    "restitution": EventTermCfg(func=shared_events.sample_restitution_label, mode="startup", params={"low": 0.0, "high": 0.5, "attribute_name": "_dreamwaq_restitution"}),
    "base_mass": EventTermCfg(func=dr.body_mass, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)), "ranges": (-1.0, 1.0), "operation": "add"}),
    "link_mass": EventTermCfg(func=dr.body_mass, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names=(r"(FL|FR|RL|RR)_.*",)), "ranges": (0.9, 1.1), "operation": "scale"}),
    "base_com": EventTermCfg(func=dr.body_com_offset, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)), "ranges": {0: (-.03, .03), 1: (-.03, .03), 2: (-.03, .03)}, "operation": "add"}),
    "pd_gains": EventTermCfg(func=dr.pd_gains, mode="startup", params={"asset_cfg": all_actuators, "kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1), "operation": "scale"}),
    "pd_labels": EventTermCfg(func=shared_events.dreamwaq_pd_labels, mode="startup", params={}),
    "motor_zero_offset": EventTermCfg(func=dr.encoder_bias, mode="startup", params={"bias_range": (-.035, .035)}),
  }


def _rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.rewards = {
    "tracking_lin_vel": RewardTermCfg(func=mdp.rewards.tracking_lin_vel, weight=1.5, params={"command_name": "twist", "sigma": .25}),
    "tracking_ang_vel": RewardTermCfg(func=mdp.rewards.tracking_ang_vel, weight=.75, params={"command_name": "twist", "sigma": .25}),
    "lin_vel_z": RewardTermCfg(func=mdp.rewards.lin_vel_z, weight=-1.0),
    "ang_vel_xy": RewardTermCfg(func=mdp.rewards.ang_vel_xy, weight=-.05),
    "orientation": RewardTermCfg(func=mdp.rewards.orientation, weight=-.2),
    "base_height": RewardTermCfg(func=mdp.rewards.base_height, weight=-5.0, params={"target_height": .40}),
    "torques": RewardTermCfg(func=mdp.rewards.torques, weight=-.0001),
    "dof_acc": RewardTermCfg(func=mdp.rewards.dof_acc, weight=-2.5e-7),
    "collision": RewardTermCfg(func=mdp.rewards.collision, weight=-1.0, params={"sensor_name": PENALIZED_SENSOR}),
    "action_rate": RewardTermCfg(func=mdp.rewards.action_rate, weight=-.01),
    "dof_pos_limits": RewardTermCfg(func=mdp.rewards.dof_pos_limits, weight=-2.0),
    "action_smoothness": RewardTermCfg(func=mdp.rewards.action_smoothness, weight=-.01),
    "stumble": RewardTermCfg(func=mdp.rewards.stumble, weight=-.5, params={"sensor_name": FEET_SENSOR}),
    "foot_clearance": RewardTermCfg(func=mdp.rewards.foot_clearance, weight=-.5, params={"sensor_name": FEET_SENSOR}),
  }


def make_dreamwaq_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_rough_env_cfg(_GO2, play=False)
  cfg.scene.entities = {"robot": dreamwaq_robot_cfg()}
  cfg.scene.num_envs = DREAMWAQ.num_envs
  cfg.episode_length_s = DREAMWAQ.episode_length_s
  cfg.decimation = DREAMWAQ.decimation
  cfg.sim.mujoco.timestep = DREAMWAQ.physics_dt
  # The shared rough preset's 500 CCD iterations allocates >2.6 GB of EPA
  # scratch for 2048 heightfield environments.  Gym PhysX source uses its
  # normal four-position-iteration contact solve, not an equivalent 500-pass
  # CCD scheme; 50 preserves stable Go2 terrain contact within this backend.
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.scale_rewards_by_dt = True
  cfg.metrics = {}; cfg.recorders = {}
  terrain_scan = tuple(sensor for sensor in (cfg.scene.sensors or ()) if sensor.name == "terrain_scan")
  replace_sensors(cfg)
  # Unlike the flat skill tasks, DreamWaQ's 261-field critic requires the
  # original 17×11 rough-terrain ray grid in addition to contact sensors.
  cfg.scene.sensors = terrain_scan + tuple(cfg.scene.sensors or ())
  cfg.actions = {"joint_pos": shared_actions.EpisodeDelayedJointPositionActionCfg(entity_name="robot", actuator_names=JOINT_NAMES, preserve_order=True, scale=.25, use_default_offset=True, delay_min_lag=0, delay_max_lag=0)}
  cfg.commands = {"twist": mdp.commands.DreamWaQVelocityCommandCfg(entity_name="robot", resampling_time_range=(10.0, 10.0), rel_standing_envs=0.0, rel_heading_envs=1.0, heading_command=True, heading_control_stiffness=.5, debug_vis=True, ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1., 1.), lin_vel_y=(-1., 1.), ang_vel_z=(-1., 1.), heading=(-3.14, 3.14)))}
  _observations(cfg); _events(cfg); _rewards(cfg)
  cfg.terminations = {"time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True), "base_contact": TerminationTermCfg(func=base_contact, params={"sensor_name": BASE_SENSOR, "force_threshold": 1.0})}
  cfg.curriculum = {"command_velocity": CurriculumTermCfg(func=mdp.commands.source_command_curriculum, params={"command_name": "twist", "max_curriculum": 2.0})}
  if play:
    cfg = deepcopy(cfg)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].terms["frame"].params["add_noise"] = False
    cfg.events.pop("push_robot")
  return cfg


def make_dreamwaq_runner_cfg():
  cfg = make_ppo_runner_cfg(DREAMWAQ.experiment_name, max_iterations=20_000, save_interval=500)
  cfg.seed = 1; cfg.clip_actions = 100.; cfg.actor.obs_normalization = False; cfg.critic.obs_normalization = False
  cfg.actor.class_name = "lloco.tasks.go2_skills.dreamwaq.mdp.rl:DreamWaQActor"
  cfg.algorithm.class_name = "lloco.tasks.go2_skills.dreamwaq.mdp.rl:DreamWaQPPO"
  return cfg
