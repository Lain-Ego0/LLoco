"""mjlab configuration for the registered Gym spring-jump task."""

from copy import deepcopy

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from lloco.tasks.rl import make_ppo_runner_cfg
from lloco.tasks.velocity import PROFILES, make_flat_env_cfg

from ..shared import actions as shared_actions
from ..shared.contacts import JOINT_NAMES
from ..shared.robot import spring_jump_robot_cfg
from ..shared.sensors import BASE_SENSOR, FEET_SENSOR, PENALIZED_SENSOR, replace_sensors
from ..trot.config import _trot_events
from . import mdp
from .profile import SPRING_JUMP, SpringJumpProfile

_GO2 = next(profile for profile in PROFILES if profile.task_name == "Go2")


def _observations(cfg: ManagerBasedRlEnvCfg, profile: SpringJumpProfile) -> None:
  cfg.observations = {
    "actor": ObservationGroupCfg(terms={"history": ObservationTermCfg(func=mdp.observations.SpringActorHistory, params={"command_name": "jump_target", "add_noise": True}, clip=(-100.0, 100.0))}, enable_corruption=True),
    "critic": ObservationGroupCfg(terms={"history": ObservationTermCfg(func=mdp.observations.SpringCriticHistory, params={"command_name": "jump_target", "sensor_name": FEET_SENSOR}, clip=(-100.0, 100.0))}, enable_corruption=False),
  }


def _rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  state = {"command_name": "jump_target", "sensor_name": FEET_SENSOR}
  cfg.rewards = {
    "before_setting": RewardTermCfg(func=mdp.rewards.BeforeSetting, weight=5.0, params=state),
    "line_z": RewardTermCfg(func=mdp.rewards.line_z, weight=16.0, params=state),
    "flight": RewardTermCfg(func=mdp.rewards.flight, weight=2.0, params=state),
    "base_height_flight": RewardTermCfg(func=mdp.rewards.base_height_flight, weight=3.0, params=state),
    "base_height_stance": RewardTermCfg(func=mdp.rewards.base_height_stance, weight=-10.0, params=state),
    "orientation": RewardTermCfg(func=mdp.rewards.orientation, weight=2.0),
    "dof_pos": RewardTermCfg(func=mdp.rewards.dof_pos, weight=-0.1),
    "dof_hip_pos": RewardTermCfg(func=mdp.rewards.hip_pos, weight=-1.0),
    "ang_vel_xy": RewardTermCfg(func=mdp.rewards.ang_vel_xy, weight=-0.2),
    "torques": RewardTermCfg(func=mdp.rewards.torques, weight=-0.0001),
    "dof_pos_limits": RewardTermCfg(func=mdp.rewards.dof_pos_limits, weight=-10.0),
    # mjlab does not currently expose the source URDF velocity-limit buffer;
    # Go2's 30 rad/s limits are preserved here as the source threshold.
    "dof_vel_limits": RewardTermCfg(func=lambda env: (torch.abs(env.scene["robot"].data.joint_vel[:, :12]) - 30.0).clamp(min=0.0).sum(1), weight=-1.0),
    "dof_vel": RewardTermCfg(func=mdp.rewards.dof_vel, weight=-0.001),
    "collision": RewardTermCfg(func=mdp.rewards.collision, weight=-50.0, params={"sensor_name": PENALIZED_SENSOR}),
    "action_rate": RewardTermCfg(func=mdp.rewards.action_rate, weight=-0.01),
    "land_pos": RewardTermCfg(func=mdp.rewards.land_pos, weight=25.0, params=state),
    "tracking_lin_vel": RewardTermCfg(func=mdp.rewards.tracking_lin_vel, weight=5.0, params=state),
    "line_vel_stance": RewardTermCfg(func=mdp.rewards.line_vel_stance, weight=-3.0, params=state),
    "feet_contact_forces": RewardTermCfg(func=mdp.rewards.feet_contact_forces, weight=-0.1, params={"sensor_name": FEET_SENSOR, "max_contact_force": 150.0}),
    "foot_clearance": RewardTermCfg(func=mdp.rewards.foot_clearance, weight=-3.0, params=state),
  }


def make_spring_jump_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  profile = SPRING_JUMP
  cfg = make_flat_env_cfg(_GO2, play=False)
  cfg.scene.entities = {"robot": spring_jump_robot_cfg()}
  cfg.scene.num_envs = profile.num_envs
  cfg.episode_length_s = profile.episode_length_s
  cfg.decimation = profile.decimation
  cfg.sim.mujoco.timestep = profile.physics_dt
  cfg.scale_rewards_by_dt = True
  cfg.metrics = {}; cfg.recorders = {}
  replace_sensors(cfg)
  cfg.actions = {"joint_pos": shared_actions.EpisodeDelayedJointPositionActionCfg(entity_name="robot", actuator_names=JOINT_NAMES, preserve_order=True, scale=profile.action_scale, use_default_offset=True, delay_min_lag=1, delay_max_lag=3)}
  cfg.commands = {"jump_target": mdp.commands.SpringJumpCommandCfg(resampling_time_range=(5.0, 5.0), debug_vis=True, entity_name="robot", heading_command=False, rel_standing_envs=0.0, rel_heading_envs=0.0, ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.8, 1.2), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=None))}
  _observations(cfg, profile); _rewards(cfg); _trot_events(cfg)
  # This source resets exact default joints/root state, unlike trot's q offset.
  cfg.events["reset_robot_joints"] = EventTermCfg(func=env_mdp.reset_joints_by_offset, mode="reset", params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)})
  cfg.events["friction"] = EventTermCfg(func=mdp.events.source_friction_buckets, mode="startup", params={"low": 0.3, "high": 1.0, "num_buckets": 64, "entity_name": "robot"})
  cfg.events["base_mass"].params["ranges"] = (-1.0, 1.0)
  cfg.terminations = {"time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True), "base_contact": TerminationTermCfg(func=mdp.terminations.below_reset_height, params={"reset_height": .15}), "base_collision": TerminationTermCfg(func=__import__("lloco.tasks.go2_skills.shared.terminations", fromlist=["base_contact"]).base_contact, params={"sensor_name": BASE_SENSOR, "force_threshold": 1.0})}
  cfg.curriculum = {}
  if play:
    cfg = deepcopy(cfg); cfg.scene.num_envs = 1; cfg.observations["actor"].enable_corruption = False; cfg.observations["actor"].terms["history"].params["add_noise"] = False; cfg.events.pop("push_robot")
  return cfg


def make_spring_jump_runner_cfg():
  cfg = make_ppo_runner_cfg(SPRING_JUMP.experiment_name, max_iterations=50_000, save_interval=100, symmetry_cfg={"data_augmentation_func": "lloco.tasks.go2_skills.spring_jump.mdp.symmetry:spring_jump_symmetry", "use_data_augmentation": False, "use_mirror_loss": True, "mirror_loss_coeff": 1.0})
  cfg.seed = 1; cfg.clip_actions = 100.0; cfg.actor.obs_normalization = False; cfg.critic.obs_normalization = False; cfg.algorithm.learning_rate = 1.0e-5
  cfg.algorithm.class_name = "lloco.tasks.go2_skills.spring_jump.mdp.symmetry:SourceSymmetricPPO"
  return cfg
