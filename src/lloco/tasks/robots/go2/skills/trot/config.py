"""Task-owned Go2 environment and runner configuration."""

from copy import deepcopy

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
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from lloco.tasks.rl import make_ppo_runner_cfg
from lloco.tasks.robots.go2.velocity import GO2_PROFILE
from lloco.tasks.velocity import make_flat_env_cfg

from ..shared import actions as shared_actions
from ..shared import terminations as shared_terminations
from ..shared.contacts import JOINT_NAMES
from ..shared.robot import trot_robot_cfg
from ..shared.sensors import (
  BASE_SENSOR,
  FEET_SENSOR,
  PENALIZED_SENSOR,
  replace_sensors,
)
from . import mdp
from .mdp import observations as trot_observations
from .mdp import rewards as trot_rewards
from .profile import TROT, TrotProfile

_GO2_VELOCITY_PROFILE = GO2_PROFILE


def _trot_observations(cfg: ManagerBasedRlEnvCfg, profile: TrotProfile) -> None:
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "history": ObservationTermCfg(
          func=trot_observations.TrotActorHistory,
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
          func=trot_observations.TrotCriticHistory,
          params={
            "command_name": "twist",
            "sensor_name": FEET_SENSOR,
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
    "sensor_name": FEET_SENSOR,
    "command_name": "twist",
    "cycle_time": profile.cycle_time,
  }
  cfg.rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=trot_rewards.tracking_lin_vel,
      weight=2.0,
      params={**gait, "sigma": 0.25},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=trot_rewards.tracking_ang_vel,
      weight=2.0,
      params={**gait, "sigma": 0.25},
    ),
    "lin_vel_z": RewardTermCfg(func=trot_rewards.lin_vel_z, weight=-2.0),
    "ang_vel_xy": RewardTermCfg(func=trot_rewards.ang_vel_xy, weight=-0.05),
    "orientation": RewardTermCfg(func=trot_rewards.orientation, weight=-2.0),
    "torques": RewardTermCfg(func=trot_rewards.torques, weight=-0.0001),
    "dof_acc": RewardTermCfg(func=trot_rewards.DofAcceleration, weight=-2.5e-7),
    "collision": RewardTermCfg(
      func=trot_rewards.collision,
      weight=-1.0,
      params={"sensor_name": PENALIZED_SENSOR},
    ),
    "action_rate": RewardTermCfg(func=trot_rewards.action_rate, weight=-0.01),
    "stand_still": RewardTermCfg(
      func=trot_rewards.stand_still,
      weight=-1.0,
      params={"command_name": "twist"},
    ),
    "base_height": RewardTermCfg(
      func=trot_rewards.base_height,
      weight=-5.0,
      params={"target_height": profile.base_height_target},
    ),
    "trot": RewardTermCfg(func=trot_rewards.trot, weight=0.8, params=gait),
    "feet_clearance": RewardTermCfg(
      func=trot_rewards.feet_clearance,
      weight=0.1,
      params={
        "command_name": "twist",
        "cycle_time": profile.cycle_time,
        "target_foot_height": profile.target_foot_height,
      },
    ),
    "default_hip_pos": RewardTermCfg(func=trot_rewards.default_hip_pos, weight=-0.2),
    "default_pos": RewardTermCfg(func=trot_rewards.default_pos, weight=-0.1),
    "contact_without_command": RewardTermCfg(
      func=trot_rewards.contact_without_command,
      weight=1.0,
      params={"sensor_name": FEET_SENSOR, "command_name": "twist"},
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
  cfg.scene.entities = {"robot": trot_robot_cfg()}
  cfg.scene.num_envs = profile.num_envs
  cfg.episode_length_s = profile.episode_length_s
  cfg.decimation = profile.decimation
  cfg.sim.mujoco.timestep = profile.physics_dt
  cfg.scale_rewards_by_dt = True
  cfg.metrics = {}
  cfg.recorders = {}
  replace_sensors(cfg)
  cfg.actions = {
    "joint_pos": shared_actions.EpisodeDelayedJointPositionActionCfg(
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
      func=shared_terminations.base_contact,
      params={"sensor_name": BASE_SENSOR, "force_threshold": 1.0},
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
