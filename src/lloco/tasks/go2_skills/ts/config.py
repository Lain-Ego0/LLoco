"""Manager-MDP configuration matching Gym's Go2 TS teacher task."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from mjlab.sensor import ContactMatch

from lloco.tasks.rl import make_ppo_runner_cfg

from ..dreamwaq import mdp as dream_mdp
from ..dreamwaq.config import make_dreamwaq_env_cfg
from ..shared.robot import dreamwaq_robot_cfg
from ..shared.sensors import BASE_SENSOR, FEET_SENSOR, PENALIZED_SENSOR
from . import mdp
from .profile import TS


def make_ts_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = deepcopy(make_dreamwaq_env_cfg(play=False))
  cfg.scene.entities = {"robot": dreamwaq_robot_cfg()}
  cfg.scene.num_envs = TS.num_envs
  cfg.episode_length_s = TS.episode_length_s
  cfg.decimation = TS.decimation
  cfg.sim.mujoco.timestep = TS.physics_dt
  # Gym's base-contact termination is the URDF base only, not the merged head.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == BASE_SENSOR:
      sensor.primary = ContactMatch(
        mode="geom", pattern="base1_collision", entity="robot"
      )

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(func=mdp.TsActor, params={"command_name": "twist"})
      },
      enable_corruption=False,
    ),
    "terrain": ObservationGroupCfg(
      terms={"scan": ObservationTermCfg(func=mdp.TsTerrain)}, enable_corruption=False
    ),
    "domain": ObservationGroupCfg(
      terms={"labels": ObservationTermCfg(func=mdp.TsDomain)}, enable_corruption=False
    ),
    "critic": ObservationGroupCfg(
      terms={
        "state": ObservationTermCfg(func=mdp.TsCritic, params={"command_name": "twist"})
      },
      enable_corruption=False,
    ),
  }

  twist = cfg.commands["twist"]
  twist.resampling_time_range = (10.0, 10.0)
  twist.ranges.lin_vel_x = (-1.0, 1.0)
  twist.ranges.lin_vel_y = (-0.6, 0.6)
  twist.ranges.ang_vel_z = (-3.14, 3.14)
  twist.ranges.heading = (-3.14, 3.14)
  cfg.curriculum["command_velocity"].params["max_curriculum"] = 1.5

  # Keep exactly the source teacher reward set and weights.
  cfg.rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=dream_mdp.rewards.tracking_lin_vel,
      weight=1.0,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=dream_mdp.rewards.tracking_ang_vel,
      weight=0.5,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "lin_vel_z": RewardTermCfg(func=dream_mdp.rewards.lin_vel_z, weight=-2.0),
    "ang_vel_xy": RewardTermCfg(func=dream_mdp.rewards.ang_vel_xy, weight=-0.05),
    "orientation": RewardTermCfg(func=dream_mdp.rewards.orientation, weight=-0.2),
    "base_height": RewardTermCfg(
      func=mdp.rewards.base_height, weight=-1.0, params={"target_height": 0.40}
    ),
    "torques": RewardTermCfg(func=dream_mdp.rewards.torques, weight=-1e-5),
    "dof_acc": RewardTermCfg(func=dream_mdp.rewards.dof_acc, weight=-2.5e-7),
    "collision": RewardTermCfg(
      func=mdp.rewards.collision,
      weight=-0.5,
      params={"sensor_names": (PENALIZED_SENSOR, BASE_SENSOR)},
    ),
    "action_rate": RewardTermCfg(func=dream_mdp.rewards.action_rate, weight=-0.01),
    # Present in Gym TS (and omitted in the first port).  This is important
    # for preventing a lateral foot impact from becoming a stable sprawled
    # front-leg equilibrium.
    "stumble": RewardTermCfg(
      func=dream_mdp.rewards.stumble,
      weight=-0.5,
      params={"sensor_name": FEET_SENSOR},
    ),
  }
  # Gym TS uses source ranges (the shared event functions retain labels).
  cfg.events["friction"].func = mdp.events.friction
  cfg.events["friction"].params.update(low=0.05, high=3.0)
  cfg.events["restitution"].func = mdp.events.restitution
  cfg.events["restitution"].params.update(high=0.5, attribute_name="_ts_restitution")
  cfg.events["base_mass"].func = mdp.events.base_mass
  cfg.events["base_mass"].params["ranges"] = (-1.0, 5.0)
  cfg.events["link_mass"].func = mdp.events.link_mass
  cfg.events["link_mass"].params["ranges"] = (0.8, 1.2)
  cfg.events["base_com"].func = mdp.events.base_com
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.1, 0.1),
    1: (-0.1, 0.1),
    2: (-0.1, 0.1),
  }
  cfg.events["pd_gains"].func = mdp.events.pd_and_torque
  cfg.events["pd_gains"].params = {"low": 0.8, "high": 1.2}
  cfg.events["motor_zero_offset"].params["bias_range"] = (-0.035, 0.035)
  cfg.events["push_robot"].interval_range_s = (8.0, 8.0)
  cfg.events["push_robot"].params.update(max_push_vel_xy=1.0, max_push_ang_vel=1.0)
  cast_action = cfg.actions["joint_pos"]
  # Gym TS has delay=True with a random 0..decimation substep switch.
  cast_action.source_substep_delay = True
  if play:
    cfg = deepcopy(cfg)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].terms["frame"].params["add_noise"] = False
    cfg.events.pop("push_robot", None)
  return cfg


def make_ts_runner_cfg():
  cfg = make_ppo_runner_cfg(
    TS.experiment_name, max_iterations=20_000, save_interval=500
  )
  cfg.seed = 1
  cfg.clip_actions = 100.0
  cfg.actor.obs_normalization = False
  cfg.critic.obs_normalization = False
  cfg.actor.class_name = "lloco.tasks.go2_skills.ts.rl:TsTeacherPolicy"
  cfg.critic.class_name = "lloco.tasks.go2_skills.ts.rl:TsTeacherPolicy"
  cfg.algorithm.class_name = "lloco.tasks.go2_skills.ts.rl:TsPPO"
  return cfg
