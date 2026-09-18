"""Task-owned Go2 environment and runner configuration."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.managers import (
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

from ..rear_stand.config import _rear_stand_events
from ..shared import actions as shared_actions
from ..shared import terminations as shared_terminations
from ..shared.contacts import JOINT_NAMES
from ..shared.robot import handstand_robot_cfg
from ..shared.sensors import (
  BASE_SENSOR,
  FEET_SENSOR,
  PENALIZED_SENSOR,
  replace_sensors,
)
from . import mdp
from .mdp import observations as handstand_observations
from .mdp import rewards as handstand_rewards
from .profile import HANDSTAND, HandstandProfile

_GO2_VELOCITY_PROFILE = GO2_PROFILE


def _handstand_observations(
  cfg: ManagerBasedRlEnvCfg, profile: HandstandProfile
) -> None:
  del profile
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(
          func=handstand_observations.HandstandActorObservation,
          params={"command_name": "twist", "add_noise": True},
          clip=(-100.0, 100.0),
        )
      },
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(
          func=handstand_observations.HandstandCriticObservation,
          params={"sensor_name": FEET_SENSOR},
          clip=(-100.0, 100.0),
        )
      },
      enable_corruption=False,
    ),
  }


def _handstand_rewards(cfg: ManagerBasedRlEnvCfg, profile: HandstandProfile) -> None:
  cfg.rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=handstand_rewards.handstand_tracking_lin_vel,
      weight=2.5,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=handstand_rewards.handstand_tracking_ang_vel,
      weight=2.5,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "tracking_lin_vel_zero": RewardTermCfg(
      func=handstand_rewards.handstand_tracking_lin_vel_zero,
      weight=-0.2,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "tracking_ang_vel_zero": RewardTermCfg(
      func=handstand_rewards.handstand_tracking_ang_vel_zero,
      weight=-0.2,
      params={"command_name": "twist"},
    ),
    "lin_vel_z": RewardTermCfg(func=handstand_rewards.handstand_lin_vel_z, weight=0.2),
    "ang_vel_xy": RewardTermCfg(
      func=handstand_rewards.handstand_ang_vel_xy, weight=0.2
    ),
    "handstand_orientation": RewardTermCfg(
      func=handstand_rewards.handstand_orientation, weight=-1.0
    ),
    "torques": RewardTermCfg(func=handstand_rewards.absolute_torques, weight=-0.0002),
    "dof_acc": RewardTermCfg(func=handstand_rewards.DofAcceleration, weight=-2.5e-7),
    "base_height": RewardTermCfg(
      func=handstand_rewards.handstand_base_height,
      weight=1.0,
      params={"target_height": profile.base_height_target},
    ),
    "handstand_feet_on_air": RewardTermCfg(
      func=handstand_rewards.handstand_rear_feet_air,
      weight=0.4,
      params={"sensor_name": FEET_SENSOR},
    ),
    "collision": RewardTermCfg(
      func=handstand_rewards.collision,
      weight=-1.0,
      params={"sensor_name": PENALIZED_SENSOR},
    ),
    "action_rate": RewardTermCfg(func=handstand_rewards.action_rate, weight=-0.05),
    "default_pos": RewardTermCfg(
      func=handstand_rewards.handstand_default_pos, weight=-0.05
    ),
    "default_hip_pos": RewardTermCfg(
      func=handstand_rewards.handstand_default_hip_pos, weight=-0.1
    ),
    "feet_clearance": RewardTermCfg(
      func=handstand_rewards.handstand_feet_clearance,
      weight=0.4,
      params={
        "cycle_time": profile.cycle_time,
        "target_foot_height": profile.target_foot_height,
      },
    ),
    "ang_xz": RewardTermCfg(func=handstand_rewards.handstand_roll, weight=-0.5),
    "contact": RewardTermCfg(
      func=handstand_rewards.handstand_front_contact,
      weight=0.3,
      params={"sensor_name": FEET_SENSOR},
    ),
    "feet_air_time": RewardTermCfg(
      func=handstand_rewards.HandstandFeetAirTime,
      weight=2.0,
      params={"sensor_name": FEET_SENSOR},
    ),
    "symmetric_joints": RewardTermCfg(
      func=handstand_rewards.handstand_symmetric_joints, weight=-0.1
    ),
    "handstand_feet_height_exp": RewardTermCfg(
      func=handstand_rewards.handstand_rear_feet_height_exp, weight=5.0
    ),
    "default_pos_reward": RewardTermCfg(
      func=handstand_rewards.handstand_default_pos_reward, weight=0.5
    ),
    # PhysX's contact transient let the source discover a viable basin despite
    # its zero-weight termination term. In MuJoCo, that omission makes falling
    # immediately the dominant local optimum. These two terms remove that
    # objective loophole while leaving all 22 source shaping terms unchanged.
    "alive": RewardTermCfg(func=env_mdp.is_alive, weight=1.0),
    "termination": RewardTermCfg(func=handstand_rewards.terminal_cost, weight=-5.0),
    "from_zero_guidance": RewardTermCfg(
      func=handstand_rewards.handstand_from_zero_guidance,
      weight=1.0,
      params={
        # 24 policy steps per PPO iteration: reach the final target near
        # iteration 400, then remove the auxiliary reward by iteration 600.
        "target_steps": 9_600,
        "fade_steps": 4_800,
        "initial_foot_height": 0.022,
        "target_foot_height": 0.67,
        "initial_base_height": 0.30,
        "target_base_height": profile.base_height_target,
      },
    ),
  }


def _handstand_events(cfg: ManagerBasedRlEnvCfg) -> None:
  _rear_stand_events(cfg)
  cfg.events["push_robot"].func = mdp.events.add_root_velocity
  cfg.events["push_robot"].params = {
    "max_push_vel_xy": 1.0,
    "max_push_ang_vel": 1.0,
  }
  cfg.events["joint_friction"] = EventTermCfg(
    func=mdp.events.scale_joint_friction_and_store_label,
    mode="startup",
    params={"low": 0.01, "high": 0.2, "entity_name": "robot"},
  )
  cfg.events["joint_damping"] = EventTermCfg(
    func=mdp.events.scale_joint_damping_and_store_label,
    mode="startup",
    params={"low": 0.0, "high": 0.2, "entity_name": "robot"},
  )
  cfg.events["joint_armature"].params["ranges"] = (0.005, 0.015)
  cfg.events["restitution_label"].params["attribute_name"] = "_handstand_restitution"


def make_handstand_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build semantic Handstand from the Gym project's ``go2_leggedstand``."""
  profile = HANDSTAND
  cfg = make_flat_env_cfg(_GO2_VELOCITY_PROFILE, play=False)
  cfg.scene.entities = {"robot": handstand_robot_cfg()}
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
      delay_min_lag=0,
      delay_max_lag=3,
      delay_update_period=4,
    )
  }
  cfg.commands = {
    "twist": mdp.commands.HandstandVelocityCommandCfg(
      resampling_time_range=(5.0, 5.0),
      debug_vis=True,
      entity_name="robot",
      heading_command=False,
      rel_standing_envs=0.0,
      rel_heading_envs=0.0,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.4, 0.4),
        lin_vel_y=(-0.0, 0.0),
        ang_vel_z=(-0.4, 0.4),
        heading=None,
      ),
    )
  }
  _handstand_observations(cfg, profile)
  _handstand_rewards(cfg, profile)
  _handstand_events(cfg)
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


def make_handstand_runner_cfg():
  cfg = make_ppo_runner_cfg(
    HANDSTAND.experiment_name,
    max_iterations=15_000,
    save_interval=100,
  )
  cfg.seed = 1
  cfg.clip_actions = 100.0
  cfg.actor.obs_normalization = False
  cfg.critic.obs_normalization = False
  cfg.algorithm.learning_rate = 1.0e-3
  return cfg
