"""Environment variants required by Go2 custom training profiles."""

import math
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import (
  discrete_obstacles,
  hf_pyramid_slope,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from ... import mdp
from ..common import (
  _go2_source_geom_friction,
  _go2_source_noise_cfg,
  _go2_source_observation_clipping,
)
from .velocity import unitree_go2_rough_env_cfg


def _go2_custom_algorithm_env_cfg(
  kind: Literal[
    "dreamwaq",
    "amp_dreamwaq",
    "cts",
    "amp_cts",
    "amp_ts",
    "amp_ts_student",
    "ts",
    "ts_student",
  ],
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Configure source-shaped observations for a custom algorithm task."""
  cfg = unitree_go2_rough_env_cfg(play=play)
  # Public rough velocity terminates on thigh contact.  Every migrated custom
  # source task instead uses ``terminate_after_contacts_on=['base']``.
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "trunk_ground_touch"},
  )
  # The custom source classes use multiplicative joint reset ranges. CTS is
  # narrower (0.9–1.1); DreamWaQ/AMP/TS use 0.5–1.5.
  cfg.events["reset_robot_joints"].func = mdp.reset_joints_by_scale
  cfg.events["reset_robot_joints"].params.pop("position_range", None)
  cfg.events["reset_robot_joints"].params["scale_range"] = (
    (0.9, 1.1) if kind == "cts" else (0.5, 1.5)
  )
  cfg.events["reset_base"].params["pose_range"].update(
    {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
  )
  # Go2_Cts/DreamWaQ and the TS base task all randomize root linear and
  # angular velocity in [-0.5, 0.5] at episode reset.
  cfg.events["reset_base"].params["velocity_range"] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.5, 0.5),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
  }
  # Source custom Go2 tasks use five terrain families with proportions
  # [smooth slope, rough slope, stairs up, stairs down, discrete].  Keep the
  # public mjlab rough preset untouched, but make all eight migrated custom
  # tasks train on the source distribution.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=70.0 if kind in ("cts", "amp_cts") else 25.0,
    num_rows=10,
    num_cols=20,
    curriculum=True,
    sub_terrains={
      "smooth_slope": hf_pyramid_slope(
        proportion=0.15, slope_range=(0.0, 0.4), platform_width=3.0, border_width=1.0
      ),
      "rough_slope": random_rough(
        proportion=0.15, noise_range=(0.02, 0.10), noise_step=0.02, border_width=1.0
      ),
      "stairs_up": pyramid_stairs(
        proportion=0.30,
        step_height_range=(0.0, 0.1),
        step_width=0.31,
        platform_width=3.0,
        border_width=1.0,
      ),
      "stairs_down": pyramid_stairs_inv(
        proportion=0.30,
        step_height_range=(0.0, 0.1),
        step_width=0.31,
        platform_width=3.0,
        border_width=1.0,
      ),
      "discrete": discrete_obstacles(
        proportion=0.10,
        obstacle_width_range=(0.3, 1.0),
        obstacle_height_range=(0.05, 0.25),
        num_obstacles=40,
        border_width=1.0,
      ),
    },
    add_lights=True,
  )
  cfg.clip_rewards_to_positive = kind == "amp_cts"
  # The source CTS/DreamWaQ/TS families all enable HIMLoco-style random
  # action switching inside the four physics substeps.  Keep the public action
  # dimension and default offset unchanged while replacing only the action
  # term implementation.
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
  )
  if kind == "amp_cts":
    # AMP-CTS is the one source variant with the mirrored abduction preload.
    amp_cts_joint_pos = cfg.scene.entities["robot"].init_state.joint_pos
    assert amp_cts_joint_pos is not None
    amp_cts_joint_pos.update(
      {
        "FL_hip_joint": -0.1,
        "FR_hip_joint": 0.1,
        "RL_hip_joint": -0.1,
        "RR_hip_joint": 0.1,
      }
    )
  # Source domain-randomization cadence samples calibration/model fields at
  # environment startup, with an interval push disturbance.  The generic rough
  # config already owns the event terms; these overrides align their ranges.
  # Source samples calibration and model parameters once when each simulator
  # environment is created; only the robot state/command is reset per episode.
  cfg.events["encoder_bias"].mode = "startup"
  cfg.events["encoder_bias"].params["bias_range"] = (-0.035, 0.035)
  custom_friction_range = {
    "cts": (0.2, 1.0),
    "amp_cts": (0.2, 1.0),
    "dreamwaq": (0.2, 1.25),
    "amp_dreamwaq": (0.2, 1.25),
    "amp_ts": (0.05, 3.0),
    "amp_ts_student": (0.05, 3.0),
    "ts": (0.05, 3.0),
    "ts_student": (0.05, 3.0),
  }[kind]
  _go2_source_geom_friction(cfg, custom_friction_range)
  cfg.events["base_com"].mode = "startup"
  is_ts_family = kind in ("ts", "ts_student", "amp_ts", "amp_ts_student")
  com_extent = 0.1 if is_ts_family else 0.05
  cfg.events["base_com"].params["ranges"] = {
    0: (-com_extent, com_extent),
    1: (-com_extent, com_extent),
    2: (-com_extent, com_extent),
  }
  cfg.events["go2_base_mass"] = EventTermCfg(
    func=envs_mdp.dr.body_mass,
    mode="startup",
    params={
      "ranges": (-1.0, 5.0),
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "operation": "add",
    },
  )
  cfg.events["go2_link_mass"] = EventTermCfg(
    func=envs_mdp.dr.body_mass,
    mode="startup",
    params={
      "ranges": (0.8, 1.2),
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=(r"(?:FL|FR|RL|RR)_(?:hip|thigh|calf)",)
      ),
      "operation": "scale",
    },
  )
  if "push_robot" in cfg.events:
    push_interval = 8.0 if is_ts_family else 4.0
    push_linear = 1.0 if is_ts_family else 0.4
    push_angular = 1.0 if is_ts_family else 0.6
    cfg.events["push_robot"].interval_range_s = (push_interval, push_interval)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-push_linear, push_linear),
      "y": (-push_linear, push_linear),
      "z": (0.0, 0.0),
      "roll": (-push_angular, push_angular),
      "pitch": (-push_angular, push_angular),
      "yaw": (-push_angular, push_angular),
    }
  gain_min, gain_max = (0.8, 1.2) if is_ts_family else (0.9, 1.1)
  cfg.events["go2_pd_gains"] = EventTermCfg(
    func=envs_mdp.dr.pd_gains,
    mode="startup",
    params={
      "kp_range": (gain_min, gain_max),
      "kd_range": (gain_min, gain_max),
      "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
      "operation": "scale",
    },
  )
  torque_multiplier_range = (0.8, 1.2) if is_ts_family else (0.9, 1.1)
  cfg.events["go2_torque_multiplier"] = EventTermCfg(
    func=mdp.go2_torque_multiplier,
    mode="startup",
    params={
      "torque_multiplier_range": torque_multiplier_range,
      "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
    },
  )
  # Replace the generic velocity rewards with the source LeggedRobot kernels.
  # The term names stay stable for logging, while the functions now separate
  # planar tracking, vertical velocity, tilt and terrain-relative height.
  tracking_lin_weight = 1.0 if kind.startswith("amp") or kind.startswith("ts") else 1.5
  tracking_ang_weight = 0.5 if kind.startswith("amp") or kind.startswith("ts") else 0.75
  base_height_weight = {
    "dreamwaq": -5.0,
    "amp_dreamwaq": -5.0,
    "amp_ts": -5.0,
    "amp_ts_student": -1.0,
    "ts": -1.0,
    "ts_student": -1.0,
  }.get(kind, -2.0)
  torque_weight = {
    "amp_cts": -1.0e-5,
    "amp_dreamwaq": -1.0e-5,
    "amp_ts_student": -1.0e-5,
    "ts": -1.0e-5,
    "ts_student": -1.0e-5,
  }.get(kind, -1.0e-4)
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.go2_source_tracking_linear_velocity,
    weight=tracking_lin_weight,
    params={"command_name": "twist", "sigma": 0.25},
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.go2_source_tracking_angular_velocity,
    weight=tracking_ang_weight,
    params={"command_name": "twist", "sigma": 0.25},
  )
  cfg.rewards["upright"] = RewardTermCfg(
    func=mdp.go2_orientation_penalty,
    weight=-0.2,
    params={},
  )
  cfg.rewards["body_ang_vel"] = RewardTermCfg(
    func=mdp.go2_angular_velocity_xy_penalty,
    weight=-0.05,
    params={},
  )
  cfg.rewards["lin_vel_z"] = RewardTermCfg(
    func=mdp.go2_linear_velocity_z_penalty,
    # CTS/DreamWaQ use the legacy -1 coefficient; AMP/TS configs retain -2.
    weight=-2.0 if kind.startswith("amp") or kind.startswith("ts") else -1.0,
    params={},
  )
  cfg.rewards["base_height"] = RewardTermCfg(
    func=mdp.go2_base_height_penalty,
    weight=base_height_weight,
    params={
      "target_height": 0.35 if kind == "amp_dreamwaq" else 0.4,
      "sensor_name": "terrain_scan",
    },
  )
  cfg.rewards["torques"] = RewardTermCfg(
    func=envs_mdp.joint_torques_l2,
    weight=torque_weight,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=".*")},
  )
  cfg.rewards["dof_acc"] = RewardTermCfg(
    func=mdp.go2_joint_acceleration_penalty,
    weight=-2.5e-7,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["action_smoothness"] = RewardTermCfg(
    func=mdp.go2_action_smoothness_penalty,
    # Only the student configs explicitly inherit the source base scale
    # ``smoothness=-0.005``.  The two teacher configs replace the nested
    # scales class, so this inherited term is absent there.
    weight=-0.005
    if kind in ("amp_ts_student", "ts_student")
    else (0.0 if kind in ("amp_ts", "ts") else -0.01),
    params={},
  )
  cfg.rewards["dof_pos_limits"].weight = (
    -2.0 if kind in ("cts", "dreamwaq", "amp_cts", "amp_dreamwaq", "amp_ts") else 0.0
  )
  cfg.rewards["action_rate_l2"].weight = -0.01
  # Source custom tasks use one aggregate collision term.  Remove the rough
  # baseline's three per-sensor terms to avoid applying the penalty twice.
  for name in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards.pop(name, None)
  cfg.rewards["collision"] = RewardTermCfg(
    func=mdp.go2_special_collision_penalty,
    weight=-0.5 if kind == "ts" else -1.0,
    params={
      "sensor_names": (
        ("thigh_ground_touch", "shank_ground_touch")
        if kind == "amp_cts"
        else ("thigh_ground_touch", "shank_ground_touch", "trunk_ground_touch")
      ),
    },
  )
  # TS teachers replace the base reward-scale class and omit clearance.  TS
  # students inherit the base ``foot_clearance=-0.01``; the other custom
  # configurations define their own -0.5 scale.
  foot_clearance_weight = (
    -0.01
    if kind in ("amp_ts_student", "ts_student")
    else (0.0 if kind in ("amp_ts", "ts") else -0.5)
  )
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=mdp.go2_source_foot_clearance_penalty,
    weight=foot_clearance_weight,
    params={
      "target_height": (
        0.09
        if kind in ("amp_ts_student", "ts_student")
        else (-0.25 if kind == "amp_cts" else -0.20)
      ),
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("FR", "FL", "RR", "RL"), preserve_order=True
      ),
    },
  )
  cfg.rewards["foot_swing_height"].weight = 0.0
  cfg.rewards["foot_slip"].weight = 0.0
  cfg.rewards["soft_landing"].weight = 0.0
  cfg.rewards["pose"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["stumble"] = RewardTermCfg(
    func=mdp.go2_stumble_penalty,
    # Student scales inherit the source base class, which has no stumble term.
    weight=0.0 if kind in ("amp_ts_student", "ts_student") else -0.5,
    params={"sensor_name": "feet_ground_contact"},
  )
  if kind == "amp_cts":
    cfg.rewards["hip_pos"] = RewardTermCfg(
      func=mdp.go2_hip_position_squared_penalty,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_hip_joint",))},
    )
  if kind == "amp_dreamwaq":
    cfg.rewards["rear_hip_limit"] = RewardTermCfg(
      func=mdp.go2_rear_hip_limit_penalty,
      weight=-1.0,
      params={
        "limit": 0.4,
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(r"(?:RL|RR)_hip_joint",), preserve_order=True
        ),
      },
    )
  if kind in ("amp_cts", "amp_ts_student", "ts_student"):
    cfg.rewards["air_time"] = RewardTermCfg(
      func=mdp.go2_source_feet_air_time_reward,
      weight=1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "offset": 0.3 if kind in ("amp_ts_student", "ts_student") else 0.5,
        "command_name": "twist",
        "command_dimensions": 3,
      },
    )
  else:
    cfg.rewards["air_time"].weight = 0.0
  cfg.rewards["base_height"].params["target_height"] = (
    0.35 if kind == "amp_dreamwaq" else 0.4
  )
  # Keep the per-world contact budget bounded for large parallel runs.  The
  # source Isaac-Gym tasks use a fixed contact buffer; an unbounded MuJoCo
  # Warp heuristic combined with the five rough-terrain variants can allocate
  # several GiB before the first step at 1024 environments.  SAP segmented
  # broadphase preserves the contact candidates while avoiding that startup
  # allocation, and 35 contacts/world matches the standard Go2 velocity
  # budget (users can still override this through ``--env.sim.nconmax``).
  cfg.sim.broadphase = "sap_segmented"
  cfg.sim.nconmax = 35
  custom_noise = _go2_source_noise_cfg(
    command_first=True,
    dof_pos_noise=0.03
    if kind in ("amp_ts_student", "ts_student")
    else (0.02 if kind == "dreamwaq" else 0.01),
    ang_vel_noise=0.3 if kind in ("amp_ts_student", "ts_student") else 0.2,
  )
  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      "source_stand": ObservationTermCfg(
        func=mdp.go2_source_stand_observation,
        params={"command_name": "twist", "command_first": True},
        noise=custom_noise,
        history_length=1,
        flatten_history_dim=True,
      )
    },
    concatenate_terms=True,
    enable_corruption=not play,
  )
  cfg.observations["history"] = ObservationGroupCfg(
    terms={
      "source_history": ObservationTermCfg(
        func=mdp.go2_source_stand_observation,
        params={"command_name": "twist", "command_first": True},
        # Reuse the processed actor frame so the history contains the exact
        # policy noise sample, as in the source ``obs_hist_buf``.
        history_source=("actor", "source_stand"),
        # CTS and DreamWaQ feed the five observations immediately preceding
        # the current actor frame to their history encoders.  mjlab history
        # includes the current frame, so retain six here and drop the newest
        # frame in the matching models.  TS students keep recurrent state in
        # their dedicated runner and do not consume this compatibility group.
        history_length=(
          6 if kind in ("cts", "amp_cts", "dreamwaq", "amp_dreamwaq") else 5
        ),
        flatten_history_dim=True,
        history_fill="zero",
      )
    },
    concatenate_terms=True,
    enable_corruption=not play,
  )
  cfg.observations["terrain"] = ObservationGroupCfg(
    terms={
      "terrain_scan": ObservationTermCfg(
        func=mdp.go2_source_terrain_heights,
        params={"sensor_name": "terrain_scan"},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  if kind.startswith("amp"):
    cfg.observations["amp"] = ObservationGroupCfg(
      terms={
        "amp_state": ObservationTermCfg(
          func=mdp.go2_amp_observation,
          params={"sensor_name": "terrain_scan"},
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )

  command_resample = 8.0 if kind == "cts" else 10.0
  if kind == "amp_cts":
    command_resample = 10.0
  custom_twist = cfg.commands["twist"]
  assert isinstance(custom_twist, UniformVelocityCommandCfg)
  custom_twist.ranges.lin_vel_x = (-1.0, 1.0)
  # Keep the per-variant source command ranges.  TS teacher/student configs
  # intentionally use heading/yaw bounds of ±pi, while the other custom
  # variants use the narrower ±1 rad/s yaw range.
  lin_y_range = {
    "cts": (-1.0, 1.0),
    "dreamwaq": (-1.0, 1.0),
    "amp_cts": (-0.65, 0.65),
    "amp_dreamwaq": (-0.6, 0.6),
    "amp_ts": (-0.6, 0.6),
    "ts": (-0.6, 0.6),
    "amp_ts_student": (-0.5, 0.5),
    "ts_student": (-0.5, 0.5),
  }[kind]
  yaw_range = (-math.pi, math.pi) if is_ts_family else (-1.0, 1.0)
  # Source custom configs set heading_command=True globally; they do not use
  # mjlab's mixed heading/standing/forward environment fractions.
  custom_twist.heading_command = True
  custom_twist.ranges.heading = (-math.pi, math.pi)
  custom_twist.rel_heading_envs = 1.0
  custom_twist.rel_standing_envs = 0.0
  custom_twist.rel_forward_envs = 0.0
  custom_twist.source_min_lin_norm = 0.1 if kind == "cts" else 0.2
  custom_twist.source_zero_command_prob = (
    0.05 if kind in ("amp_ts", "amp_ts_student", "ts", "ts_student") else 0.0
  )
  custom_twist.source_zero_xy_prob = (
    0.05 if kind in ("amp_ts", "amp_ts_student", "ts", "ts_student") else 0.0
  )
  custom_twist.ranges.lin_vel_y = lin_y_range
  custom_twist.ranges.ang_vel_z = yaw_range
  custom_twist.resampling_time_range = (command_resample, command_resample)
  if kind in ("amp_ts_student", "ts_student"):
    # Source student configs set commands.curriculum=False.
    cfg.curriculum.pop("command_vel", None)

  if kind in ("dreamwaq", "amp_dreamwaq"):
    cfg.observations["explicit"] = ObservationGroupCfg(
      terms={
        "dreamwaq_velocity": ObservationTermCfg(
          func=mdp.go2_dreamwaq_velocity_target,
          params={},
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "dreamwaq_privileged": ObservationTermCfg(
          func=mdp.go2_source_dreamwaq_privileged_observation,
          params={"command_name": "twist", "sensor_name": "terrain_scan"},
          history_length=3,
          flatten_history_dim=True,
          history_fill="zero",
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.observations["privileged"] = ObservationGroupCfg(
      terms={
        "dreamwaq_privileged_frame": ObservationTermCfg(
          func=mdp.go2_source_dreamwaq_privileged_observation,
          params={"command_name": "twist", "sensor_name": "terrain_scan"},
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
  elif kind in ("cts", "amp_cts"):
    # Source CTS uses a fixed 3:1 teacher/student environment split.  The mask
    # is kept in its own observation group so it does not alter the 45-D actor
    # contract consumed by checkpoints and deployment wrappers.
    cfg.observations["teacher_mask"] = ObservationGroupCfg(
      terms={"cts_teacher_mask": ObservationTermCfg(func=mdp.go2_cts_teacher_mask)},
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.observations["privileged"] = ObservationGroupCfg(
      terms={
        "cts_privileged": ObservationTermCfg(
          func=mdp.go2_source_cts_privileged_observation,
          params={
            "sensor_name": "terrain_scan",
            "contact_sensor_name": "feet_ground_contact",
          },
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "cts_critic": ObservationTermCfg(
          func=mdp.go2_source_cts_critic_observation,
          params={
            "sensor_name": "terrain_scan",
            "contact_sensor_name": "feet_ground_contact",
            "include_lin_vel": kind == "amp_cts",
          },
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
  else:
    cfg.observations["privileged"] = ObservationGroupCfg(
      terms={
        # The TS teacher encoder consumes the source 74-D
        # domain-randomization/contact buffer.  Keep the full 309-D value
        # observation in the separate ``critic`` group below.
        "ts_teacher_privileged": ObservationTermCfg(
          func=mdp.go2_source_ts_privileged_observation,
          params={"contact_sensor_name": "feet_ground_contact"},
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
      terms={
        "ts_critic": ObservationTermCfg(
          func=mdp.go2_source_ts_critic_observation,
          params={
            "sensor_name": "terrain_scan",
            "contact_sensor_name": "feet_ground_contact",
          },
          history_length=1,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
  _go2_source_observation_clipping(cfg)
  return cfg


def unitree_go2_dreamwaq_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("dreamwaq", play)


def unitree_go2_amp_dreamwaq_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("amp_dreamwaq", play)


def unitree_go2_cts_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("cts", play)


def unitree_go2_amp_cts_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("amp_cts", play)


def unitree_go2_amp_ts_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("amp_ts", play)


def unitree_go2_amp_ts_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("amp_ts_student", play)


def unitree_go2_ts_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("ts", play)


def unitree_go2_ts_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_custom_algorithm_env_cfg("ts_student", play)
