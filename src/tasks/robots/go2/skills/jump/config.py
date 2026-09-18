"""Task-owned Go2 environment and runner configuration."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.rl import make_ppo_runner_cfg
from src.tasks.robots.go2.velocity import GO2_PROFILE
from src.tasks.velocity import make_flat_env_cfg

from ..shared import actions as shared_actions
from ..shared import terminations as shared_terminations
from ..shared.contacts import JOINT_NAMES
from ..shared.robot import jump_robot_cfg
from ..shared.sensors import (
  BASE_SENSOR,
  FEET_SENSOR,
  PENALIZED_SENSOR,
  replace_sensors,
)
from ..trot.config import _trot_events
from . import mdp
from .mdp import observations as jump_observations
from .mdp import rewards as jump_rewards
from .profile import JUMP, JumpProfile

_GO2_VELOCITY_PROFILE = GO2_PROFILE


def _jump_observations(cfg: ManagerBasedRlEnvCfg, profile: JumpProfile) -> None:
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "history": ObservationTermCfg(
          func=jump_observations.JumpActorHistory,
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
          func=jump_observations.JumpCriticHistory,
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


def _jump_rewards(cfg: ManagerBasedRlEnvCfg, profile: JumpProfile) -> None:
  cfg.rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=jump_rewards.jump_tracking_lin_vel,
      weight=2.0,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=jump_rewards.jump_tracking_ang_vel,
      weight=2.0,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "lin_vel_z": RewardTermCfg(func=jump_rewards.jump_lin_vel_z, weight=0.05),
    "ang_vel_xy": RewardTermCfg(func=jump_rewards.jump_ang_vel_xy, weight=0.2),
    "orientation": RewardTermCfg(func=jump_rewards.jump_orientation, weight=0.6),
    "torques": RewardTermCfg(func=jump_rewards.absolute_torques, weight=-0.0002),
    "dof_acc": RewardTermCfg(func=jump_rewards.JointVelocityDifference, weight=-5.5e-4),
    "base_height": RewardTermCfg(
      func=jump_rewards.jump_base_height,
      weight=1.0,
      params={
        "command_name": "twist",
        "target_height": profile.base_height_target,
      },
    ),
    "feet_air_time": RewardTermCfg(
      func=jump_rewards.FeetAirTime,
      weight=1.0,
      params={"sensor_name": FEET_SENSOR, "command_name": "twist"},
    ),
    "collision": RewardTermCfg(
      func=jump_rewards.collision,
      weight=-1.0,
      params={"sensor_name": PENALIZED_SENSOR},
    ),
    "action_rate": RewardTermCfg(func=jump_rewards.action_rate, weight=-0.01),
    "stand_still": RewardTermCfg(
      func=jump_rewards.stand_still,
      weight=-1.0,
      params={"command_name": "twist"},
    ),
    "default_pos": RewardTermCfg(func=jump_rewards.default_pos, weight=-0.1),
    "default_hip_pos": RewardTermCfg(
      func=jump_rewards.jump_default_hip_pos, weight=0.3
    ),
    "feet_contact_forces": RewardTermCfg(
      func=jump_rewards.feet_contact_forces,
      weight=-0.01,
      params={"sensor_name": FEET_SENSOR, "max_contact_force": 100.0},
    ),
    "jump": RewardTermCfg(
      func=jump_rewards.jump_contact_match,
      weight=2.0,
      params={
        "sensor_name": FEET_SENSOR,
        "command_name": "twist",
        "cycle_time": profile.cycle_time,
      },
    ),
    "feet_clearance": RewardTermCfg(
      func=jump_rewards.jump_feet_clearance,
      weight=0.5,
      params={
        "command_name": "twist",
        "cycle_time": profile.cycle_time,
        "max_height": profile.target_foot_height,
      },
    ),
    "contact_without_command": RewardTermCfg(
      func=jump_rewards.contact_without_command,
      weight=1.0,
      params={"sensor_name": FEET_SENSOR, "command_name": "twist"},
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
  cfg.scene.entities = {"robot": jump_robot_cfg()}
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
  _jump_observations(cfg, profile)
  _jump_rewards(cfg, profile)
  _jump_events(cfg)
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
