"""Shared builder for source-compatible Go2 aerial and balance tasks."""

import math
from typing import Literal

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import (
  pyramid_stairs,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from .. import mdp
from ..mdp import Go2TriggeredCommandCfg
from .common import (
  _go2_source_47_noise_cfg,
  _go2_source_geom_friction,
  _go2_source_noise_cfg,
  _go2_source_observation_clipping,
)
from .locomotion import (
  unitree_go2_flat_env_cfg as unitree_go2_flat_env_cfg,
)
from .locomotion import (
  unitree_go2_rough_env_cfg as unitree_go2_rough_env_cfg,
)
from .locomotion import (
  unitree_go2_trot_env_cfg as unitree_go2_trot_env_cfg,
)


def _go2_special_action_env_cfg(
  task: Literal["jump", "spring_jump", "backflip", "handstand", "leggedstand"],
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build a registered scaffold for a source Go2 special-action task.

  The common scene, actuator, reset and PPO plumbing is shared with the flat
  task.  Each task gets a distinct reward shaping term so its source-specific
  controller can be filled in without duplicating asset configuration.
  """
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_base"].params["pose_range"].update(
    {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
  )
  # Flat-task construction removes terrain collision sensors for the common
  # velocity baseline. Special-action source rewards still depend on body
  # collisions, so restore only the non-foot sensors here.
  rough_sensor_cfgs = unitree_go2_rough_env_cfg(play=play).scene.sensors or ()
  collision_sensor_names = {"thigh_ground_touch", "shank_ground_touch"}
  required_sensor_names = collision_sensor_names | {"trunk_ground_touch"}
  cfg.scene.sensors = (cfg.scene.sensors or ()) + tuple(
    sensor for sensor in rough_sensor_cfgs if sensor.name in required_sensor_names
  )
  # The source special tasks allow deliberate inversion/rotation.  The flat
  # baseline's 70-degree ``fell_over`` termination would stop a jump or
  # handstand before it can reach its target orientation, so replace it with
  # the source base-contact termination.
  cfg.terminations.pop("fell_over", None)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "trunk_ground_touch"},
  )
  collision_weight = {
    "jump": -1.0,
    "spring_jump": -50.0,
    "backflip": -10.0,
    "handstand": -2.0,
    "leggedstand": -1.0,
  }[task]
  cfg.rewards["collision"] = RewardTermCfg(
    func=mdp.go2_special_collision_penalty,
    weight=collision_weight,
    params={"sensor_names": tuple(sorted(collision_sensor_names))},
  )
  # Source special-task startup randomization.  The shared rough config owns
  # the event plumbing; specialize its ranges here instead of using the
  # public rough-task defaults.
  friction_range = {
    "jump": (0.2, 1.2),
    "spring_jump": (0.3, 1.0),
    "backflip": (0.2, 1.25),
    "handstand": (0.2, 1.2),
    "leggedstand": (0.2, 1.2),
  }[task]
  _go2_source_geom_friction(cfg, friction_range)
  cfg.events["encoder_bias"].params["bias_range"] = (-0.035, 0.035)
  com_extent = 0.05 if task in ("handstand", "leggedstand") else 0.03
  cfg.events["base_com"].params["ranges"] = {
    0: (-com_extent, com_extent),
    1: (-com_extent, com_extent),
    2: (-com_extent, com_extent),
  }
  base_mass_range = (-1.0, 2.0) if task in ("handstand", "leggedstand") else (-1.0, 1.0)
  cfg.events["go2_base_mass"] = EventTermCfg(
    func=envs_mdp.dr.body_mass,
    mode="startup",
    params={
      "ranges": base_mass_range,
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "operation": "add",
    },
  )
  cfg.events["go2_link_mass"] = EventTermCfg(
    func=envs_mdp.dr.body_mass,
    mode="startup",
    params={
      "ranges": (0.9, 1.1),
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=(r"(?:FL|FR|RL|RR)_(?:hip|thigh|calf)",)
      ),
      "operation": "scale",
    },
  )
  cfg.events["go2_pd_gains"] = EventTermCfg(
    func=envs_mdp.dr.pd_gains,
    mode="startup",
    params={
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
      "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
      "operation": "scale",
    },
  )
  if task in ("handstand", "leggedstand"):
    cfg.events["reset_robot_joints"].func = mdp.reset_joints_by_scale
    cfg.events["reset_robot_joints"].params.pop("position_range", None)
    cfg.events["reset_robot_joints"].params["scale_range"] = (0.5, 1.5)
    # The source stand tasks initialize all six root velocities uniformly in
    # [-0.5, 0.5], unlike the flat locomotion tasks which start at rest.
    cfg.events["reset_base"].params["velocity_range"] = {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.5, 0.5),
      "roll": (-0.5, 0.5),
      "pitch": (-0.5, 0.5),
      "yaw": (-0.5, 0.5),
    }
    joint_extent = (0.01, 0.1) if task == "handstand" else (0.01, 0.2)
    damping_extent = (0.0, 0.1) if task == "handstand" else (0.0, 0.2)
    armature_extent = (0.003, 0.08) if task == "handstand" else (0.005, 0.015)
    joint_cfg = SceneEntityCfg("robot", joint_names=(".*",))
    cfg.events["go2_joint_friction"] = EventTermCfg(
      func=envs_mdp.dr.joint_friction,
      mode="startup",
      params={"ranges": joint_extent, "asset_cfg": joint_cfg, "operation": "abs"},
    )
    cfg.events["go2_joint_damping"] = EventTermCfg(
      func=envs_mdp.dr.joint_damping,
      mode="startup",
      params={"ranges": damping_extent, "asset_cfg": joint_cfg, "operation": "abs"},
    )
    cfg.events["go2_joint_armature"] = EventTermCfg(
      func=envs_mdp.dr.joint_armature,
      mode="startup",
      params={"ranges": armature_extent, "asset_cfg": joint_cfg, "operation": "abs"},
    )
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  cfg.actions["joint_pos"] = mdp.Go2DelayedJointPositionActionCfg(
    entity_name=joint_pos_action.entity_name,
    actuator_names=joint_pos_action.actuator_names,
    scale=joint_pos_action.scale,
    offset=joint_pos_action.offset,
    clip=joint_pos_action.clip,
    preserve_order=joint_pos_action.preserve_order,
    use_default_offset=joint_pos_action.use_default_offset,
    delay=True,
    observation_latency=task in ("jump", "spring_jump", "backflip", "leggedstand"),
    observation_imu_orientation="gravity" if task == "backflip" else "euler",
    observation_motor_latency_range=(1, 3),
    observation_imu_latency_range=(1, 3),
  )
  if task == "jump":
    # Jump uses a reset-sampled 1–3 substep command buffer in the source.
    delayed_action = cfg.actions["joint_pos"]
    assert isinstance(delayed_action, mdp.Go2DelayedJointPositionActionCfg)
    delayed_action.delay_mode = "buffer"
    delayed_action.delay_steps_range = (1, 3)
  elif task == "spring_jump":
    # Source Spring-Jump starts exactly at its configured default pose.
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
  elif task == "backflip":
    # Source Backflip uses default_q * U(0.5, 1.25).
    cfg.events["reset_robot_joints"].func = mdp.reset_joints_by_scale
    cfg.events["reset_robot_joints"].params.pop("position_range", None)
    cfg.events["reset_robot_joints"].params["scale_range"] = (0.5, 1.25)
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  # All source special-action configs sample angular velocity directly; only
  # handstand enables heading tracking below.
  twist.heading_command = False
  twist.ranges.heading = None
  twist.resampling_time_range = (5.0, 5.0)
  # The source jump/spring/backflip configs sample yaw rate directly from
  # [-1, 1] rad/s (the common mjlab velocity template defaults to [-0.5, 0.5]).
  # Stand tasks override this below with their narrower source ranges.
  if task in ("jump", "spring_jump", "backflip"):
    twist.ranges.ang_vel_z = (-1.0, 1.0)
  if task == "jump":
    twist.source_zero_command_prob = 0.05
    twist.source_zero_xy_prob = 0.05
    twist.source_min_lin_norm = 0.1
  elif task in ("spring_jump", "backflip"):
    twist.source_zero_command_prob = 0.05
    twist.source_zero_xy_prob = 0.05
    twist.source_min_lin_norm = 0.1
  if task in ("spring_jump", "backflip", "handstand", "leggedstand"):
    # These source command configs explicitly disable command curriculum.
    cfg.curriculum.pop("command_vel", None)
  if "push_robot" in cfg.events:
    push_interval = 8.0 if task in ("handstand", "leggedstand") else 4.0
    push_linear = 1.0 if task == "leggedstand" else 0.4
    push_angular = 1.0 if task == "leggedstand" else 0.6
    cfg.events["push_robot"].interval_range_s = (push_interval, push_interval)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-push_linear, push_linear),
      "y": (-push_linear, push_linear),
      "z": (0.0, 0.0),
      "roll": (-push_angular, push_angular),
      "pitch": (-push_angular, push_angular),
      "yaw": (-push_angular, push_angular),
    }
  if task in ("spring_jump", "backflip"):
    # In the source flip environments command[2] is a delayed one-shot
    # trigger, not a yaw-rate target.  Keep the same three-value observation
    # contract while exposing the trigger/flight state through the term.
    if task == "spring_jump":
      initial_lin_vel_x = (0.8, 1.2)
      trigger_steps = (50, 60)
    else:
      initial_lin_vel_x = (0.0, 0.0)
      # BackFlip resets ``command_frame`` with randint(50, 60) in the
      # source task.  The later 50--100 window was an accidental broadening
      # that delayed take-off for a substantial fraction of episodes.
      trigger_steps = (50, 60)
    twist.ranges.lin_vel_x = initial_lin_vel_x
    twist.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands["twist"] = Go2TriggeredCommandCfg(
      entity_name=twist.entity_name,
      resampling_time_range=twist.resampling_time_range,
      ranges=twist.ranges,
      heading_command=False,
      rel_standing_envs=0.0,
      rel_forward_envs=0.0,
      source_zero_command_prob=twist.source_zero_command_prob,
      source_zero_xy_prob=twist.source_zero_xy_prob,
      source_min_lin_norm=twist.source_min_lin_norm,
      trigger_steps=trigger_steps,
      initial_lin_vel_x=initial_lin_vel_x,
      # Spring-Jump samples one scalar and broadcasts it to all environments
      # reset in the same call; BackFlip's fixed zero command is unaffected.
      shared_initial_lin_vel_x=task == "spring_jump",
      contact_sensor_name="feet_ground_contact",
      push_towards_goal=True,
      upward_push_range=(1.5, 2.2) if task == "spring_jump" else (2.0, 3.5),
      pitch_push_range=(0.0, 0.0) if task == "spring_jump" else (2.0, 2.5),
    )
    twist = cfg.commands["twist"]
    assert isinstance(twist, Go2TriggeredCommandCfg)
  if task == "jump":
    # The source jump task is the one special-action task trained on a
    # curriculum of upward stairs (spring-jump and backflip use a plane).
    # Re-enable a generator after the common flat config removed it, while
    # retaining the source observation contract that does not expose heights.
    assert cfg.scene.terrain is not None
    cfg.events["reset_base"].params["pose_range"].update(
      {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
    )
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
      size=(8.0, 8.0),
      border_width=25.0,
      num_rows=10,
      num_cols=20,
      curriculum=not play,
      sub_terrains={
        "stairs": pyramid_stairs(
          proportion=1.0,
          step_height_range=(0.0, 0.1),
          step_width=0.31,
          platform_width=3.0,
          border_width=1.0,
        )
      },
      add_lights=True,
    )
    if not play:
      cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=mdp.terrain_levels_vel,
        params={"command_name": "twist"},
      )
    cfg.episode_length_s = 24.0
    cfg.observations["actor"] = ObservationGroupCfg(
      terms={
        "source_jump": ObservationTermCfg(
          func=mdp.go2_source_trot_observation,
          params={"command_name": "twist", "cycle_time": 1.5},
          noise=_go2_source_47_noise_cfg(),
          history_length=10,
          flatten_history_dim=True,
          history_fill="zero",
        )
      },
      concatenate_terms=True,
      enable_corruption=not play,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "source_jump_privileged": ObservationTermCfg(
          func=mdp.go2_source_jump_privileged_observation,
          params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
          history_length=3,
          flatten_history_dim=True,
          history_fill="zero",
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.rewards["jump"] = RewardTermCfg(
      func=mdp.go2_jump_contact_reward,
      weight=2.0,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )
    cfg.rewards["feet_clearance"] = RewardTermCfg(
      func=mdp.go2_jump_feet_clearance,
      weight=0.5,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL")),
        "command_name": "twist",
      },
    )
    cfg.rewards["foot_clearance"].weight = 0.0
    cfg.rewards["foot_swing_height"].weight = 0.0
  elif task == "spring_jump":
    cfg.episode_length_s = 5.0
    # The source spring-jump controller starts from a slightly crouched pose.
    robot = cfg.scene.entities["robot"]
    robot.init_state.pos = (0.0, 0.0, 0.39)
    spring_joint_pos = robot.init_state.joint_pos
    assert spring_joint_pos is not None
    spring_joint_pos.update(
      {
        "FL_hip_joint": 0.0,
        "FR_hip_joint": 0.0,
        "RL_hip_joint": 0.0,
        "RR_hip_joint": 0.0,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
      }
    )
    cfg.observations["actor"] = ObservationGroupCfg(
      terms={
        "source_spring_jump": ObservationTermCfg(
          func=mdp.go2_source_spring_jump_observation,
          params={"command_name": "twist"},
          noise=_go2_source_47_noise_cfg(),
          history_length=10,
          flatten_history_dim=True,
          history_fill="zero",
        )
      },
      concatenate_terms=True,
      enable_corruption=not play,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "source_spring_jump_privileged": ObservationTermCfg(
          func=mdp.go2_source_spring_jump_privileged_observation,
          params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
          history_length=3,
          flatten_history_dim=True,
          history_fill="zero",
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.rewards["flight"] = RewardTermCfg(
      # Source ``_reward_flight`` returns the command term's persistent
      # ``was_in_flight`` flag, rather than a periodic contact-phase kernel.
      # Passing the command name lets the mjlab term read that state and keeps
      # the reward active until the first landing.
      func=mdp.go2_all_feet_airborne,
      weight=2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
      },
    )
    cfg.terminations["reset_height"] = TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.15},
    )
  elif task in ("handstand", "leggedstand"):
    cfg.episode_length_s = 20.0
    # Stand tasks use the higher-gain controller from the source configs.
    articulation = cfg.scene.entities["robot"].articulation
    assert articulation is not None
    for actuator in articulation.actuators:
      assert isinstance(actuator, BuiltinPositionActuatorCfg)
      actuator.stiffness = 40.0
      actuator.damping = 1.0
    if task == "handstand":
      twist.ranges.lin_vel_x = (-0.2, 0.6)
      twist.ranges.lin_vel_y = (0.0, 0.0)
      twist.ranges.ang_vel_z = (-0.4, 0.4)
      twist.heading_command = True
      twist.ranges.heading = (-math.pi, math.pi)
      twist.rel_heading_envs = 1.0
      twist.rel_standing_envs = 0.0
      twist.rel_forward_envs = 0.0
      twist.source_zero_command_prob = 0.20
      twist.source_zero_xy_prob = 0.10
      twist.source_min_lin_norm = 0.1
      twist.resampling_time_range = (10.0, 10.0)
    else:
      twist.ranges.lin_vel_x = (-0.4, 0.4)
      twist.ranges.lin_vel_y = (0.0, 0.0)
      twist.ranges.ang_vel_z = (-0.4, 0.4)
      twist.heading_command = False
      twist.source_zero_command_prob = 0.20
      twist.source_zero_xy_prob = 0.10
      twist.source_min_lin_norm = 0.1
      twist.resampling_time_range = (5.0, 5.0)
    # Neither stand variant uses the generic template's standing/forward
    # environment mixture.
    twist.rel_standing_envs = 0.0
    twist.rel_forward_envs = 0.0
    cfg.observations["actor"] = ObservationGroupCfg(
      terms={
        "source_stand": ObservationTermCfg(
          func=mdp.go2_source_stand_observation,
          params={"command_name": "twist"},
          noise=_go2_source_noise_cfg(command_first=False),
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=not play,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "source_stand_privileged": ObservationTermCfg(
          func=mdp.go2_source_stand_privileged_observation,
          params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    # Neither source stand config enables the generic stand-still term.
    cfg.rewards.pop("stand_still", None)
    if task == "handstand":
      target_gravity = (-1.0, 0.0, 0.0)
      target_height = 0.52
      airborne_indices = (0, 1)  # FL/FR in the source task.
      target_joint_pos = (
        0.0,
        0.8,
        -1.5,
        0.0,
        0.8,
        -1.5,
        0.0,
        2.25,
        -1.75,
        0.0,
        2.25,
        -1.75,
      )
      target_joint_slice = (0, 6)
    else:
      target_gravity = (1.0, 0.0, 0.0)
      target_height = 0.47
      airborne_indices = (2, 3)  # RL/RR in the source task.
      target_joint_pos = (
        0.0,
        -0.7,
        -1.75,
        0.0,
        -0.7,
        -1.75,
        0.0,
        0.8,
        -1.5,
        0.0,
        0.8,
        -1.5,
      )
      target_joint_slice = (6, 12)
    cfg.rewards["target_gravity"] = RewardTermCfg(
      func=mdp.go2_target_gravity_error,
      weight=-1.0,
      params={"target_gravity": target_gravity},
    )
    cfg.rewards["target_base_height"] = RewardTermCfg(
      func=mdp.go2_target_base_height_reward,
      weight=1.5 if task == "handstand" else 1.0,
      params={"target_height": target_height, "sharpness": 5.0},
    )
    cfg.rewards["handstand_feet_on_air"] = RewardTermCfg(
      func=mdp.go2_selected_feet_airborne,
      weight=0.4,
      params={"sensor_name": "feet_ground_contact", "foot_indices": airborne_indices},
    )
    # ``contact`` and ``default_pos`` below are the source terms.  Avoid
    # registering equivalent aliases here, which previously doubled both
    # contributions in the stand tasks.
    cfg.rewards["desired_pose_reward"] = RewardTermCfg(
      func=mdp.go2_target_joint_position_reward,
      weight=0.5,
      params={
        "target_joint_pos": target_joint_pos,
        "joint_slice": target_joint_slice,
        "sharpness": 1.0,
        "ready_target_height": target_height,
        "ready_threshold": 0.70 if task == "handstand" else 0.78,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    )
    cfg.rewards["default_pos"] = RewardTermCfg(
      func=mdp.go2_target_joint_position_penalty,
      weight=-0.1 if task == "handstand" else -0.05,
      params={
        "target_joint_pos": target_joint_pos,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    )
    cfg.rewards["default_hip_pos"] = RewardTermCfg(
      func=mdp.go2_hip_position_penalty,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["symmetric_joints"] = RewardTermCfg(
      func=mdp.go2_joint_symmetry_penalty,
      weight=-0.1,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        # Leggedstand's source reward compares only the rear/support pair;
        # Handstand keeps the default two-pair comparison.
        "leg_indices": (2, 3) if task == "leggedstand" else (0, 1, 2, 3),
      },
    )
    stand_joint_cfg = SceneEntityCfg("robot", joint_names=(".*",))
    stand_site_cfg = SceneEntityCfg(
      "robot",
      site_names=("FL", "FR", "RL", "RR"),
      preserve_order=True,
    )
    if task == "handstand":
      stand_forward_sign = -1.0
      stand_ready_height = 0.52
      swing_feet = (0, 1)
      support_feet = (2, 3)
      track_threshold = 0.70
    else:
      stand_forward_sign = 1.0
      stand_ready_height = 0.47
      swing_feet = (2, 3)
      support_feet = (0, 1)
      track_threshold = 0.78
    cfg.rewards["tracking_lin_vel_source"] = RewardTermCfg(
      func=mdp.go2_stand_linear_velocity_reward,
      weight=2.5,
      params={
        "command_name": "twist",
        "target_height": stand_ready_height,
        "ready_threshold": track_threshold,
        "forward_sign": stand_forward_sign,
        "sigma": 0.25,
      },
    )
    cfg.rewards["tracking_ang_vel_source"] = RewardTermCfg(
      func=mdp.go2_stand_angular_velocity_reward,
      weight=2.5,
      params={
        "command_name": "twist",
        "target_height": stand_ready_height,
        "ready_threshold": track_threshold,
        "forward_sign": stand_forward_sign,
        "sigma": 0.25,
      },
    )
    if task == "leggedstand":
      cfg.rewards["tracking_lin_vel_zero"] = RewardTermCfg(
        func=mdp.go2_stand_zero_linear_velocity_reward,
        weight=-0.2,
        params={
          "command_name": "twist",
          "target_height": stand_ready_height,
          "ready_threshold": track_threshold,
          "forward_sign": stand_forward_sign,
          "sigma": 0.25,
        },
      )
      cfg.rewards["tracking_ang_vel_zero"] = RewardTermCfg(
        func=mdp.go2_stand_zero_angular_velocity_penalty,
        weight=-0.2,
        params={
          "command_name": "twist",
          "target_height": stand_ready_height,
          "ready_threshold": track_threshold,
        },
      )
    cfg.rewards["lin_vel_z"] = RewardTermCfg(
      func=mdp.go2_stand_linear_z_velocity_reward,
      weight=0.2,
      params={},
    )
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(
      func=mdp.go2_stand_ang_vel_penalty,
      # Source stand configs use a positive ``exp(-||ω_yz||)`` reward.
      weight=0.2,
      params={},
    )
    cfg.rewards["torques"] = RewardTermCfg(
      func=mdp.go2_torque_penalty,
      weight=-0.0002,
      params={"asset_cfg": stand_joint_cfg},
    )
    cfg.rewards["dof_vel"] = RewardTermCfg(
      func=mdp.go2_joint_velocity_penalty,
      weight=0.0,
      params={"asset_cfg": stand_joint_cfg},
    )
    cfg.rewards["dof_acc"] = RewardTermCfg(
      # Both source stand configs retain the small acceleration regularizer
      # used by the original LeggedRobot implementation.
      func=mdp.go2_joint_acceleration_penalty,
      weight=-2.5e-7,
      params={"asset_cfg": stand_joint_cfg},
    )
    # The stand configs define ``max_contact_force`` but leave the matching
    # reward scale unset, so no contact-force term is registered here.
    # Handstand explicitly enables the soft joint-limit term; Leggedstand's
    # source reward scale leaves it disabled.
    cfg.rewards["dof_pos_limits"].weight = -2.0 if task == "handstand" else 0.0
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["handstand_feet_on_air"] = RewardTermCfg(
      func=mdp.go2_selected_feet_airborne,
      weight=0.4,
      params={"sensor_name": "feet_ground_contact", "foot_indices": swing_feet},
    )
    cfg.rewards["contact"] = RewardTermCfg(
      func=mdp.go2_stand_contact_reward,
      weight=0.3,
      params={
        "sensor_name": "feet_ground_contact",
        "foot_indices": support_feet,
        "ready_target_height": stand_ready_height,
        "ready_threshold": track_threshold,
        "asset_cfg": stand_joint_cfg,
      },
    )
    cfg.rewards["ang_xz"] = RewardTermCfg(
      func=mdp.go2_stand_roll_penalty,
      weight=-0.5,
      params={
        "ready_target_height": stand_ready_height,
        "ready_threshold": track_threshold,
        "asset_cfg": stand_joint_cfg,
      },
    )
    cfg.rewards["feet_clearance"] = RewardTermCfg(
      func=mdp.go2_stand_foot_clearance_reward,
      weight=0.4,
      params={
        # The legacy reward reads ``contact_foot_indices`` for this term;
        # these are the two support limbs (the source variable is misleadingly
        # named ``feet_clearance`` even though it tracks that pair).
        "foot_indices": support_feet,
        "cycle_time": 1.6,
        "target_height": 0.06,
        "ready_target_height": stand_ready_height,
        "ready_threshold": track_threshold,
        "asset_cfg": stand_site_cfg,
      },
    )
    if task == "leggedstand":
      cfg.rewards["feet_air_time"] = RewardTermCfg(
        func=mdp.go2_stand_foot_air_time_reward,
        weight=2.0,
        params={
          "sensor_name": "feet_ground_contact",
          "foot_indices": support_feet,
          "ready_target_height": stand_ready_height,
          "ready_threshold": track_threshold,
          "asset_cfg": stand_joint_cfg,
        },
      )
    if task == "handstand":
      # These two symmetry terms exist only in Go2_Handstand_Config.  The
      # Leggedstand source config does not assign either reward scale.
      cfg.rewards["orientation_symmetry"] = RewardTermCfg(
        func=mdp.go2_stand_orientation_symmetry_penalty,
        weight=-0.5,
        params={},
      )
      cfg.rewards["feet_height_symmetry"] = RewardTermCfg(
        func=mdp.go2_stand_feet_height_symmetry_penalty,
        weight=-0.2,
        params={"foot_indices": swing_feet, "asset_cfg": stand_site_cfg},
      )
    cfg.rewards["handstand_feet_height_exp"] = RewardTermCfg(
      func=mdp.go2_stand_hand_height_reward,
      weight=5.0,
      params={
        "target_height": 0.67,
        "sharpness": 10.0,
        # Source ``feet_pos`` is built from ``feet_name_reward`` (the two
        # airborne limbs), not all four foot sites.
        "foot_indices": airborne_indices,
        "asset_cfg": stand_site_cfg,
      },
    )
    if task == "handstand":
      cfg.rewards["alive"] = RewardTermCfg(
        func=mdp.go2_alive_reward,
        weight=1.0,
        params={},
      )
    for name in (
      "track_linear_velocity",
      "track_angular_velocity",
      "pose",
      "upright",
      "foot_clearance",
      "foot_swing_height",
      "foot_slip",
      "soft_landing",
    ):
      cfg.rewards[name].weight = 0.0
  if task == "jump":
    # Source jump uses a positive idle-height kernel around 0.30 m.  The
    # generic velocity template's squared penalty at 0.42 m pulls the policy
    # toward the nominal standing height and is therefore inappropriate here.
    cfg.rewards["base_height"] = RewardTermCfg(
      func=mdp.go2_idle_base_height_reward,
      weight=1.0,
      params={"target_height": 0.30, "sharpness": 10.0, "command_name": "twist"},
    )
    cfg.rewards["vertical_velocity"] = RewardTermCfg(
      func=mdp.go2_jump_vertical_velocity_reward,
      weight=0.05,
      params={},
    )
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(
      func=mdp.go2_jump_angular_velocity_reward,
      weight=0.2,
      params={},
    )
    cfg.rewards["upright"] = RewardTermCfg(
      func=mdp.go2_jump_orientation_reward,
      weight=0.6,
      params={"sharpness": 10.0},
    )
    cfg.rewards["dof_acc"] = RewardTermCfg(
      func=mdp.go2_joint_acceleration_penalty,
      weight=-5.5e-4,
      params={
        "divide_by_dt": False,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    )
    cfg.rewards["default_pos"] = RewardTermCfg(
      func=mdp.go2_default_position_penalty,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["default_hip_pos"] = RewardTermCfg(
      func=mdp.go2_default_hip_position_penalty,
      weight=0.3,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["contact_without_command"] = RewardTermCfg(
      func=mdp.go2_contact_without_command,
      weight=1.0,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )
    cfg.rewards["track_linear_velocity"] = RewardTermCfg(
      func=mdp.go2_jump_tracking_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "sigma": 0.25, "command_threshold": 0.1},
    )
    cfg.rewards["track_angular_velocity"] = RewardTermCfg(
      func=mdp.go2_jump_tracking_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "sigma": 0.25, "command_threshold": 0.1},
    )
    cfg.rewards["air_time"] = RewardTermCfg(
      func=mdp.go2_source_feet_air_time_reward,
      weight=1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "offset": 0.5,
        "command_name": "twist",
        "command_dimensions": 3,
      },
    )
    cfg.rewards["feet_contact_forces"] = RewardTermCfg(
      func=mdp.go2_foot_contact_force_penalty,
      weight=-0.01,
      params={"sensor_name": "feet_ground_contact", "max_contact_force": 100.0},
    )
    cfg.rewards["torques"] = RewardTermCfg(
      # The source Jump override uses an absolute-torque penalty rather than
      # the squared kernel used by the generic velocity task.
      func=mdp.go2_torque_penalty,
      weight=-0.0002,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    # The source Jump scales do not include a joint-position-limit term, and
    # use -0.01 (rather than mjlab's generic -0.1) for action-rate cost.
    cfg.rewards["dof_pos_limits"].weight = 0.0
    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.rewards["stand_still"] = RewardTermCfg(
      func=mdp.go2_stand_still_penalty,
      weight=-1.0,
      params={"command_name": "twist"},
    )
    for name in ("pose", "foot_slip", "soft_landing"):
      cfg.rewards[name].weight = 0.0
  elif task == "spring_jump":
    # Spring-Jump alternates a take-off/flight height target with a lower
    # post-landing stance target.  The contact-derived phase mask is stateless
    # but preserves the source shaping signal in mjlab's reward manager.
    cfg.rewards["line_z"] = RewardTermCfg(
      func=mdp.go2_upward_velocity_reward,
      weight=16.0,
      params={"command_name": "twist"},
    )
    cfg.rewards["flight"] = RewardTermCfg(
      func=mdp.go2_all_feet_airborne,
      weight=2.0,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )
    cfg.rewards["base_height_flight"] = RewardTermCfg(
      func=mdp.go2_base_height_phase_reward,
      weight=3.0,
      params={
        "target_height": 0.47,
        "phase": "airborne",
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "gain": 6.0,
      },
    )
    cfg.rewards["base_height_stance"] = RewardTermCfg(
      func=mdp.go2_base_height_phase_reward,
      weight=-10.0,
      params={
        "target_height": 0.35,
        "phase": "contact",
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
      },
    )
    cfg.rewards["land_pos"] = RewardTermCfg(
      func=mdp.go2_landing_position_reward,
      weight=25.0,
      params={
        "command_name": "twist",
        "target_offset_scale": 1.0,
        "min_jump_height": 0.42,
        "max_orientation_error": 0.6,
        # Preserve the source expression ``euler_xyz.sum() < 0.6`` (without
        # an absolute value), including its signed-angle cancellation.
        "absolute_orientation_error": False,
      },
    )
    cfg.rewards["before_setting"] = RewardTermCfg(
      func=mdp.go2_before_setting_reward,
      weight=5.0,
      params={"command_name": "twist", "denominator": 2.0},
    )
    cfg.rewards["orientation"] = RewardTermCfg(
      func=mdp.go2_orientation_reward,
      weight=2.0,
      params={},
    )
    cfg.rewards["dof_pos"] = RewardTermCfg(
      func=mdp.go2_joint_position_penalty,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["dof_hip_pos"] = RewardTermCfg(
      func=mdp.go2_hip_position_penalty,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(
      func=mdp.go2_ang_vel_xy_penalty,
      weight=-0.2,
      params={"include_pitch": True},
    )
    cfg.rewards["dof_vel"] = RewardTermCfg(
      func=mdp.go2_joint_velocity_penalty,
      weight=-0.001,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["dof_vel_limits"] = RewardTermCfg(
      func=mdp.go2_joint_velocity_limit_penalty,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["torques"] = RewardTermCfg(
      func=mdp.go2_torque_penalty,
      weight=-0.0001,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["feet_contact_forces"] = RewardTermCfg(
      func=mdp.go2_foot_contact_force_penalty,
      weight=-0.1,
      params={"sensor_name": "feet_ground_contact", "max_contact_force": 150.0},
    )
    cfg.rewards["line_vel_stance"] = RewardTermCfg(
      func=mdp.go2_line_velocity_stance_penalty,
      weight=-3.0,
      params={"command_name": "twist"},
    )
    cfg.rewards["foot_clearance"] = RewardTermCfg(
      func=mdp.go2_flip_foot_clearance_penalty,
      weight=-3.0,
      params={
        "command_name": "twist",
        "asset_cfg": SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL")),
      },
    )
    cfg.rewards["foot_swing_height"].weight = 0.0
    cfg.rewards["tracking_lin_vel_source"] = RewardTermCfg(
      func=mdp.go2_flight_linear_velocity_reward,
      weight=5.0,
      params={"command_name": "twist", "gain": 1.6},
    )
    # Terms from the generic velocity task that have no source equivalent are
    # disabled for the flip-specific reward contract.
    for name in (
      "track_linear_velocity",
      "track_angular_velocity",
      "pose",
      "upright",
      "foot_swing_height",
      "foot_slip",
      "soft_landing",
    ):
      cfg.rewards[name].weight = 0.0
    cfg.rewards["dof_pos_limits"].weight = -10.0
    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.rewards.pop("base_height", None)
    cfg.rewards.pop("vertical_velocity", None)
  elif task == "backflip":
    cfg.episode_length_s = 4.0
    # Backflip starts from the neutral hip pose used by the source task.
    backflip_joint_pos = cfg.scene.entities["robot"].init_state.joint_pos
    assert backflip_joint_pos is not None
    backflip_joint_pos.update(
      {
        "FL_hip_joint": 0.0,
        "FR_hip_joint": 0.0,
        "RL_hip_joint": 0.0,
        "RR_hip_joint": 0.0,
      }
    )
    cfg.observations["actor"] = ObservationGroupCfg(
      terms={
        "source_backflip": ObservationTermCfg(
          func=mdp.go2_source_backflip_observation,
          params={"command_name": "twist"},
          noise=_go2_source_47_noise_cfg(),
          history_length=10,
          flatten_history_dim=True,
          history_fill="zero",
        )
      },
      concatenate_terms=True,
      enable_corruption=not play,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "source_backflip_privileged": ObservationTermCfg(
          func=mdp.go2_source_backflip_privileged_observation,
          params={"command_name": "twist"},
          history_length=3,
          flatten_history_dim=True,
          history_fill="zero",
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.rewards["inverted_orientation"] = RewardTermCfg(
      func=mdp.go2_inverted_orientation_reward,
      weight=1.0,
      params={},
    )
    cfg.rewards["line_z"] = RewardTermCfg(
      func=mdp.go2_upward_velocity_reward,
      weight=25.0,
      params={"command_name": "twist"},
    )
    cfg.rewards["angle_y"] = RewardTermCfg(
      func=mdp.go2_pitch_angular_velocity_reward,
      weight=10.0,
      params={"command_name": "twist"},
    )
    cfg.rewards["base_height_flight"] = RewardTermCfg(
      func=mdp.go2_base_height_phase_reward,
      weight=5.0,
      params={
        "target_height": 0.60,
        "phase": "airborne",
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "gain": 6.0,
      },
    )
    cfg.rewards["base_height_stance"] = RewardTermCfg(
      func=mdp.go2_base_height_phase_reward,
      weight=10.0,
      params={
        "target_height": 0.35,
        "phase": "contact",
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
      },
    )
    cfg.rewards["backflip_orientation"] = RewardTermCfg(
      func=mdp.go2_backflip_orientation_reward,
      weight=10.0,
      params={"command_name": "twist"},
    )
    cfg.rewards["symmetric_joints"] = RewardTermCfg(
      func=mdp.go2_joint_symmetry_penalty,
      weight=-0.3,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["default_hip_pos"] = RewardTermCfg(
      func=mdp.go2_default_hip_position_penalty,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["land_pos"] = RewardTermCfg(
      func=mdp.go2_landing_position_reward,
      weight=1.0,
      params={"command_name": "twist", "target_offset_scale": 0.0},
    )
    cfg.rewards["before_setting"] = RewardTermCfg(
      func=mdp.go2_before_setting_reward,
      weight=5.0,
      params={"command_name": "twist", "denominator": 4.0},
    )
    cfg.rewards["orientation_before"] = RewardTermCfg(
      func=mdp.go2_orientation_before_reward,
      weight=2.0,
      params={"command_name": "twist"},
    )
    cfg.rewards["dof_pos"] = RewardTermCfg(
      func=mdp.go2_joint_position_penalty,
      weight=-0.2,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["line_vel_stance"] = RewardTermCfg(
      func=mdp.go2_line_velocity_stance_penalty,
      weight=-1.0,
      # BackFlip penalizes all three body-frame linear velocity components
      # throughout the episode; Spring-Jump uses planar velocity after landing.
      params={
        "command_name": "twist",
        "include_vertical": True,
        "after_landing": False,
      },
    )
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(
      func=mdp.go2_ang_vel_xy_penalty,
      weight=-0.2,
      params={"include_pitch": False},
    )
    cfg.rewards["dof_vel"] = RewardTermCfg(
      func=mdp.go2_joint_velocity_penalty,
      weight=-0.001,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["dof_vel_limits"] = RewardTermCfg(
      func=mdp.go2_joint_velocity_limit_penalty,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["torques"] = RewardTermCfg(
      func=mdp.go2_torque_penalty,
      weight=-0.0001,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["feet_contact_forces"] = RewardTermCfg(
      func=mdp.go2_foot_contact_force_penalty,
      weight=-0.1,
      params={"sensor_name": "feet_ground_contact", "max_contact_force": 150.0},
    )
    # Backflip follows the source scale table; disable inherited generic
    # orientation shaping and restore its stronger joint-limit/action-rate
    # terms.
    cfg.rewards["inverted_orientation"].weight = 0.0
    cfg.rewards["dof_pos_limits"].weight = -10.0
    cfg.rewards["action_rate_l2"].weight = -0.01
    # Backflip has explicit pre-trigger/post-landing shaping instead of the
    # generic velocity-task tracking terms.
    for name in (
      "track_linear_velocity",
      "track_angular_velocity",
      "pose",
      "upright",
      "foot_clearance",
      "foot_swing_height",
      "foot_slip",
      "soft_landing",
    ):
      cfg.rewards[name].weight = 0.0
    cfg.terminations["reset_height"] = TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.1},
    )
  # Task-specific source horizons are training semantics. The shared play
  # contract is intentionally unbounded, so apply it after task overrides.
  if play:
    cfg.episode_length_s = int(1e9)
  _go2_source_observation_clipping(cfg)
  return cfg
