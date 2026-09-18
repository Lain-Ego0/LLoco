"""Task-owned Go2 environment and runner configuration."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.managers import (
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
from ..shared.robot import rear_stand_robot_cfg
from ..shared.sensors import (
  BASE_SENSOR,
  FEET_SENSOR,
  PENALIZED_SENSOR,
  replace_rear_stand_sensors,
)
from . import mdp
from .mdp import observations as rear_stand_observations
from .mdp import rewards as rear_stand_rewards
from .profile import REAR_STAND, RearStandProfile

_GO2_VELOCITY_PROFILE = GO2_PROFILE


def _rear_stand_observations(
  cfg: ManagerBasedRlEnvCfg, profile: RearStandProfile
) -> None:
  del profile
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(
          func=rear_stand_observations.RearStandActorObservation,
          params={"command_name": "twist", "add_noise": True},
          clip=(-100.0, 100.0),
        )
      },
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(
          func=rear_stand_observations.RearStandCriticObservation,
          params={"sensor_name": FEET_SENSOR},
          clip=(-100.0, 100.0),
        )
      },
      enable_corruption=False,
    ),
  }


def _rear_stand_rewards(cfg: ManagerBasedRlEnvCfg, profile: RearStandProfile) -> None:
  cfg.rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_tracking_lin_vel,
      weight=2.5,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_tracking_ang_vel,
      weight=2.5,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "lin_vel_z": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_lin_vel_z, weight=0.2
    ),
    "ang_vel_xy": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_ang_vel_xy, weight=0.2
    ),
    "rear_stand_orientation": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_orientation, weight=-1.0
    ),
    "torques": RewardTermCfg(func=rear_stand_rewards.absolute_torques, weight=-0.0002),
    "dof_acc": RewardTermCfg(func=rear_stand_rewards.DofAcceleration, weight=-2.5e-7),
    "base_height": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_base_height,
      weight=1.5,
      params={"target_height": profile.base_height_target},
    ),
    "rear_stand_feet_on_air": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_front_feet_air,
      weight=0.4,
      params={"sensor_name": FEET_SENSOR},
    ),
    "collision": RewardTermCfg(
      func=rear_stand_rewards.collision,
      weight=-2.0,
      params={"sensor_name": PENALIZED_SENSOR},
    ),
    "action_rate": RewardTermCfg(func=rear_stand_rewards.action_rate, weight=-0.05),
    "default_pos": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_default_pos, weight=-0.1
    ),
    "default_hip_pos": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_default_hip_pos, weight=-0.1
    ),
    "feet_clearance": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_feet_clearance,
      weight=0.4,
      params={
        "cycle_time": profile.cycle_time,
        "target_foot_height": profile.target_foot_height,
      },
    ),
    "ang_xz": RewardTermCfg(func=rear_stand_rewards.rear_stand_roll, weight=-0.5),
    "contact": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_rear_contact,
      weight=0.3,
      params={"sensor_name": FEET_SENSOR},
    ),
    "symmetric_joints": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_symmetric_joints, weight=-0.1
    ),
    "orientation_symmetry": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_orientation_symmetry, weight=-0.5
    ),
    "feet_height_symmetry": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_feet_height_symmetry, weight=-0.2
    ),
    "rear_stand_feet_height_exp": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_front_feet_height_exp, weight=5.0
    ),
    "default_pos_reward": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_default_pos_reward, weight=0.5
    ),
    "dof_pos_limits": RewardTermCfg(
      func=rear_stand_rewards.rear_stand_dof_pos_limits, weight=-2.0
    ),
    "alive": RewardTermCfg(func=rear_stand_rewards.alive, weight=1.0),
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
  cfg.scene.entities = {"robot": rear_stand_robot_cfg()}
  cfg.scene.num_envs = profile.num_envs
  cfg.episode_length_s = profile.episode_length_s
  cfg.decimation = profile.decimation
  cfg.sim.mujoco.timestep = profile.physics_dt
  cfg.scale_rewards_by_dt = True
  cfg.metrics = {}
  cfg.recorders = {}
  replace_rear_stand_sensors(cfg)
  cfg.actions = {
    "joint_pos": shared_actions.EpisodeDelayedJointPositionActionCfg(
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
      func=shared_terminations.base_contact,
      params={"sensor_name": BASE_SENSOR, "force_threshold": 1.0},
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
        "lloco.tasks.robots.go2.skills.rear_stand.mdp.symmetry:rear_stand_symmetry"
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
    "lloco.tasks.robots.go2.skills.rear_stand.mdp.symmetry:SourceSymmetricPPO"
  )
  return cfg
