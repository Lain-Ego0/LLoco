from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
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

from ..shared.actions import EpisodeDelayedJointPositionActionCfg
from ..shared.contacts import JOINT_NAMES
from ..shared.robot import jump_robot_cfg
from ..shared.sensors import BASE_SENSOR, FEET_SENSOR, PENALIZED_SENSOR, replace_sensors
from ..shared.terminations import base_contact
from ..trot.config import _trot_events
from . import mdp
from .profile import BACKFLIP

_GO2 = GO2_PROFILE


def make_backflip_env_cfg(*, play=False) -> ManagerBasedRlEnvCfg:
  c = make_flat_env_cfg(_GO2, play=False)
  c.scene.entities = {"robot": jump_robot_cfg()}
  c.scene.num_envs = 4096
  c.episode_length_s = 4.0
  c.decimation = 4
  c.sim.mujoco.timestep = 0.005
  c.scale_rewards_by_dt = True
  c.metrics = {}
  c.recorders = {}
  replace_sensors(c)
  c.actions = {
    "joint_pos": EpisodeDelayedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=JOINT_NAMES,
      preserve_order=True,
      scale=0.25,
      use_default_offset=True,
      delay_min_lag=1,
      delay_max_lag=3,
    )
  }
  c.commands = {
    "flip": mdp.commands.BackflipCommandCfg(
      resampling_time_range=(5.0, 5.0),
      debug_vis=True,
      entity_name="robot",
      heading_command=False,
      rel_standing_envs=0.0,
      rel_heading_envs=0.0,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=None
      ),
    )
  }
  c.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "history": ObservationTermCfg(
          func=mdp.observations.BackflipActorHistory,
          params={"command_name": "flip", "add_noise": True},
          clip=(-100.0, 100.0),
        )
      },
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "history": ObservationTermCfg(
          func=mdp.observations.BackflipCriticHistory,
          params={"command_name": "flip"},
          clip=(-100.0, 100.0),
        )
      },
      enable_corruption=False,
    ),
  }
  s = {"command_name": "flip", "sensor_name": FEET_SENSOR}
  r = mdp.rewards
  c.rewards = {
    "before_setting": RewardTermCfg(func=r.BeforeSetting, weight=5.0, params=s),
    "line_z": RewardTermCfg(func=r.line_z, weight=25.0, params=s),
    "angle_y": RewardTermCfg(func=r.angle_y, weight=10.0, params=s),
    "base_height_flight": RewardTermCfg(func=r.height_flight, weight=5.0, params=s),
    "base_height_stance": RewardTermCfg(func=r.height_stance, weight=10.0, params=s),
    "orientation": RewardTermCfg(func=r.orientation, weight=10.0, params=s),
    "orientation_before": RewardTermCfg(
      func=r.orientation_before, weight=2.0, params={"command_name": "flip"}
    ),
    "dof_pos": RewardTermCfg(func=r.dof_pos, weight=-0.2),
    "line_vel_stance": RewardTermCfg(func=r.line_vel_stance, weight=-1.0),
    "ang_vel_xy": RewardTermCfg(func=r.ang_xy, weight=-0.2),
    "torques": RewardTermCfg(func=r.torques, weight=-0.0001),
    "dof_pos_limits": RewardTermCfg(func=r.dof_pos_limits, weight=-10.0),
    "dof_vel_limits": RewardTermCfg(func=r.dof_vel_limits, weight=-2.0),
    "dof_vel": RewardTermCfg(func=r.dof_vel, weight=-0.001),
    "collision": RewardTermCfg(
      func=r.collision, weight=-10.0, params={"sensor_name": PENALIZED_SENSOR}
    ),
    "action_rate": RewardTermCfg(func=r.action_rate, weight=-0.01),
    "feet_contact_forces": RewardTermCfg(
      func=r.feet_force,
      weight=-0.1,
      params={"sensor_name": FEET_SENSOR, "max_contact_force": 150.0},
    ),
    "land_pos": RewardTermCfg(func=r.land_pos, weight=1.0, params=s),
    "symmetric_joints": RewardTermCfg(func=r.symmetric_joints, weight=-0.3),
    "default_hip_pos": RewardTermCfg(func=r.hip, weight=-0.5),
  }
  _trot_events(c)
  c.events["friction"] = EventTermCfg(
    func=mdp.events.source_friction_buckets,
    mode="startup",
    params={"low": 0.2, "high": 1.25, "num_buckets": 64, "entity_name": "robot"},
  )
  c.events["base_mass"].params["ranges"] = (-1.0, 1.0)
  c.events["reset_robot_joints"] = EventTermCfg(
    func=env_mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=JOINT_NAMES, preserve_order=True
      ),
    },
  )
  c.terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "below_height": TerminationTermCfg(
      func=mdp.terminations.below_reset_height, params={"reset_height": 0.1}
    ),
    "base_contact": TerminationTermCfg(
      func=base_contact, params={"sensor_name": BASE_SENSOR, "force_threshold": 1.0}
    ),
  }
  c.curriculum = {}
  if play:
    c = deepcopy(c)
    c.scene.num_envs = 1
    c.observations["actor"].enable_corruption = False
    c.observations["actor"].terms["history"].params["add_noise"] = False
    c.events.pop("push_robot")
  return c


def make_backflip_runner_cfg():
  c = make_ppo_runner_cfg(
    BACKFLIP.experiment_name, max_iterations=50_000, save_interval=100
  )
  c.seed = 1
  c.clip_actions = 100.0
  c.actor.obs_normalization = False
  c.critic.obs_normalization = False
  c.algorithm.learning_rate = 1e-5
  return c
