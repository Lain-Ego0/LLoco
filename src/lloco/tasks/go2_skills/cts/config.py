"""Manager-MDP configuration for the source Go2 CTS task."""

from copy import deepcopy
from typing import Any, cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
)
from mjlab.sensor import ContactMatch

from lloco.tasks.rl import make_ppo_runner_cfg

from ..dreamwaq import mdp as dream_mdp
from ..dreamwaq.config import make_dreamwaq_env_cfg
from ..shared import rl as shared_rl
from ..shared.robot import cts_robot_cfg
from ..shared.sensors import BASE_SENSOR, PENALIZED_SENSOR
from . import mdp
from .profile import CTS


def make_cts_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = deepcopy(make_dreamwaq_env_cfg(play=False))
  cfg.scene.entities = {"robot": cts_robot_cfg()}
  # Gym terminates on the URDF ``base`` rigid body only.  The shared MJCF
  # merges Head_upper/Head_lower collision geoms into base_link, so matching
  # the body would incorrectly terminate on head contact as well.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == BASE_SENSOR:
      sensor.primary = ContactMatch(
        mode="geom", pattern="base1_collision", entity="robot"
      )
  cfg.scene.num_envs = CTS.num_envs
  cfg.episode_length_s = CTS.episode_length_s
  cfg.decimation = CTS.decimation
  cfg.sim.mujoco.timestep = CTS.physics_dt
  twist_cfg = cast(Any, cfg.commands["twist"])
  twist_cfg.__class__ = mdp.CtsVelocityCommandCfg
  twist_cfg.resampling_time_range = (8.0, 8.0)
  twist_cfg.ranges = mdp.CtsVelocityCommandCfg.Ranges(
    lin_vel_x=(-1.0, 1.0),
    lin_vel_y=(-1.0, 1.0),
    ang_vel_z=(-1.0, 1.0),
    heading=(-3.14, 3.14),
  )
  # Gym CTS domain-randomization ranges (not DreamWaQ's defaults).
  cfg.events["friction"].func = mdp.randomize_friction
  cfg.events["friction"].params = {"low": 0.2, "high": 1.0}
  cfg.events["restitution"].params.update(
    high=1.0, attribute_name="_cts_restitution"
  )
  cfg.events["base_mass"].func = mdp.randomize_base_mass
  cfg.events["base_mass"].params["ranges"] = (-1.0, 2.0)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.05, 0.05),
    1: (-0.05, 0.05),
    2: (-0.05, 0.05),
  }
  cfg.events["base_com"].func = mdp.randomize_base_com
  # Source CTS resets joints in the narrow [0.9, 1.1] nominal-pose range;
  # DreamWaQ's inherited [0.5, 1.5] range creates falls the Gym task never
  # presents during training.
  cfg.events["reset_robot_joints"].params["scale_range"] = (0.9, 1.1)
  cfg.events["pd_gains"].func = mdp.randomize_pd_and_torque
  cfg.events["pd_gains"].params = {"low": 0.9, "high": 1.1}
  cfg.events.pop("pd_labels", None)
  cast(Any, cfg.actions["joint_pos"]).source_substep_delay = True
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(
          func=dream_mdp.observations.DreamActor,
          params={
            "command_name": "twist",
            "add_noise": True,
            "joint_position_noise": 0.01,
          },
        )
      },
      enable_corruption=False,
    ),
    "teacher": ObservationGroupCfg(
      terms={
        "state": ObservationTermCfg(
          func=mdp.CtsTeacherObservation,
          params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
        )
      },
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "state": ObservationTermCfg(
          func=mdp.CtsCriticObservation,
          params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
        )
      },
      enable_corruption=False,
    ),
    "history": ObservationGroupCfg(
      terms={
        "frames": ObservationTermCfg(func=dream_mdp.observations.DreamActorHistory)
      },
      enable_corruption=False,
    ),
  }
  # Source CTS rewards.  In particular there is no survival bonus or terminal
  # cost; earlier MuJoCo-only bridges taught a low, rear-supported sitting pose.
  cfg.rewards["base_height"] = RewardTermCfg(
    func=dream_mdp.rewards.base_height,
    weight=-2.0,
    params={"target_height": 0.40},
  )
  # MuJoCo admits a stable folded-leg equilibrium around 0.18 m.  Keep the
  # source reward ratio in the normal gait region and add a localized barrier
  # that is exactly zero above 0.30 m.
  cfg.rewards["low_base_height_barrier"] = RewardTermCfg(
    func=mdp.low_base_height_barrier,
    weight=-100.0,
    params={"minimum_height": 0.30},
  )
  cfg.rewards["dof_acc"] = RewardTermCfg(func=mdp.CtsDofAcceleration, weight=-2.5e-7)
  cfg.rewards["action_smoothness"] = RewardTermCfg(
    func=mdp.CtsActionSmoothness, weight=-0.01
  )
  cfg.rewards["collision"] = RewardTermCfg(
    func=mdp.collision,
    weight=-1.0,
    params={"sensor_names": (PENALIZED_SENSOR, BASE_SENSOR)},
  )
  cfg.rewards.pop("alive", None)
  cfg.rewards["termination"] = RewardTermCfg(
    func=shared_rl.terminal_cost, weight=0.0
  )
  cfg.curriculum["command_velocity"].func = mdp.cts_command_curriculum
  if play:
    cfg = deepcopy(cfg)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].terms["frame"].params["add_noise"] = False
    cfg.events.pop("push_robot", None)
  return cfg


def make_cts_runner_cfg():
  cfg = make_ppo_runner_cfg(
    CTS.experiment_name, max_iterations=20_000, save_interval=500
  )
  cfg.seed = 1
  cfg.clip_actions = 100.0
  cfg.actor.obs_normalization = False
  cfg.critic.obs_normalization = False
  cfg.actor.class_name = "lloco.tasks.go2_skills.cts.rl:CtsStudentPolicy"
  cfg.critic.class_name = "lloco.tasks.go2_skills.cts.rl:CtsStudentPolicy"
  cfg.algorithm.class_name = "lloco.tasks.go2_skills.cts.rl:CtsPPO"
  return cfg
