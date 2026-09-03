"""Go2 locomotion environment factory support."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from ... import mdp
from ..common import (
  _go2_source_47_noise_cfg,
  _go2_source_geom_friction,
  _go2_source_observation_clipping,
)
from .velocity import unitree_go2_flat_env_cfg, unitree_go2_rough_env_cfg


def unitree_go2_trot_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the source-compatible Go2 trot task on flat terrain.

  This is the first special-action task in the migration.  It keeps the common
  mjlab velocity MDP for scene plumbing while adding the source task's 47-D
  single-frame observation, ten-frame history, phase gait reward, and standstill
  terms.  Source motor/IMU/action latency is enabled on the action term; the
  policy mirror loss is configured with the task's RL runner configuration.
  """
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_base"].params["pose_range"].update(
    {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
  )
  # Trot's source asset still penalizes thigh/calf/base contacts and resets
  # on base contact even though it trains on a plane.  The public flat
  # baseline removes these sensors, so restore only the three source body
  # groups here instead of weakening the task's collision contract.
  rough_sensor_cfgs = unitree_go2_rough_env_cfg(play=play).scene.sensors or ()
  collision_sensor_names = {"thigh_ground_touch", "shank_ground_touch"}
  required_sensor_names = collision_sensor_names | {"trunk_ground_touch"}
  cfg.scene.sensors = (cfg.scene.sensors or ()) + tuple(
    sensor for sensor in rough_sensor_cfgs if sensor.name in required_sensor_names
  )
  cfg.terminations.pop("fell_over", None)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "trunk_ground_touch"},
  )
  _go2_source_geom_friction(cfg, (0.2, 1.2))
  cfg.events["encoder_bias"].params["bias_range"] = (-0.035, 0.035)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.03, 0.03),
    1: (-0.03, 0.03),
    2: (-0.03, 0.03),
  }
  cfg.events["go2_base_mass"] = EventTermCfg(
    func=envs_mdp.dr.body_mass,
    mode="startup",
    params={
      "ranges": (-1.0, 2.0),
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
  # The source ``go2_trot`` starts from a symmetric trot pose.  The shared Go2
  # asset defaults intentionally follow the jump/stand tasks (rear thigh 1.0
  # and signed hip offsets), so override only this task's reset pose here.
  trot_joint_pos = cfg.scene.entities["robot"].init_state.joint_pos
  assert trot_joint_pos is not None
  trot_joint_pos.update(
    {
      "FL_hip_joint": 0.0,
      "FR_hip_joint": 0.0,
      "RL_hip_joint": 0.0,
      "RR_hip_joint": 0.0,
      "FL_thigh_joint": 0.8,
      "FR_thigh_joint": 0.8,
      "RL_thigh_joint": 0.8,
      "RR_thigh_joint": 0.8,
    }
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
    delay_mode="buffer",
    delay_steps_range=(1, 3),
    observation_latency=True,
    observation_motor_latency_range=(1, 3),
    observation_imu_latency_range=(1, 3),
  )

  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      "source_trot": ObservationTermCfg(
        func=mdp.go2_source_trot_observation,
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
      "source_trot_privileged": ObservationTermCfg(
        func=mdp.go2_source_trot_privileged_observation,
        params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
        history_length=3,
        flatten_history_dim=True,
        history_fill="zero",
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.ranges.ang_vel_z = (-1.0, 1.0)
  twist.source_zero_command_prob = 0.05
  twist.source_zero_xy_prob = 0.05
  twist.source_min_lin_norm = 0.1
  twist.resampling_time_range = (5.0, 5.0)

  cfg.rewards["trot"] = RewardTermCfg(
    func=mdp.go2_trot_phase_reward,
    weight=0.8,
    params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
  )
  cfg.rewards["feet_clearance_source"] = RewardTermCfg(
    func=mdp.go2_trot_feet_clearance_reward,
    weight=0.1,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("FR", "FL", "RR", "RL"), preserve_order=True
      ),
    },
  )
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.go2_source_tracking_linear_velocity,
    weight=2.0,
    params={"command_name": "twist", "sigma": 0.25, "trot_gate": True},
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.go2_source_tracking_angular_velocity,
    weight=2.0,
    params={"command_name": "twist", "sigma": 0.25, "trot_gate": True},
  )
  cfg.rewards["upright"] = RewardTermCfg(
    func=mdp.go2_orientation_penalty,
    weight=-2.0,
    params={},
  )
  cfg.rewards["body_ang_vel"] = RewardTermCfg(
    func=mdp.go2_angular_velocity_xy_penalty,
    weight=-0.05,
    params={},
  )
  cfg.rewards["lin_vel_z"] = RewardTermCfg(
    func=mdp.go2_linear_velocity_z_penalty,
    weight=-2.0,
    params={},
  )
  cfg.rewards["base_height"] = RewardTermCfg(
    func=mdp.go2_base_height_penalty,
    weight=-5.0,
    params={"target_height": 0.29, "sensor_name": "terrain_scan"},
  )
  cfg.rewards["dof_acc"] = RewardTermCfg(
    func=mdp.go2_joint_acceleration_penalty,
    weight=-2.5e-7,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["torques"] = RewardTermCfg(
    # Source Trot uses the base LeggedRobot squared-torque kernel.
    func=envs_mdp.joint_torques_l2,
    weight=-0.0001,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=".*")},
  )
  cfg.rewards["action_smoothness"] = RewardTermCfg(
    func=mdp.go2_action_smoothness_penalty,
    # Source Trot only has first-order action-rate cost; the generic
    # second-order smoothness term is not part of its reward contract.
    weight=0.0,
    params={},
  )
  cfg.rewards["dof_pos_limits"].weight = 0.0
  cfg.rewards["action_rate_l2"].weight = -0.01
  cfg.rewards["pose"].weight = 0.0
  cfg.rewards["foot_clearance"].weight = 0.0
  cfg.rewards["foot_swing_height"].weight = 0.0
  cfg.rewards["foot_slip"].weight = 0.0
  cfg.rewards["soft_landing"].weight = 0.0
  cfg.rewards["default_hip_pos"] = RewardTermCfg(
    func=mdp.go2_default_hip_position_penalty,
    weight=-0.2,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["default_pos"] = RewardTermCfg(
    func=mdp.go2_default_position_penalty,
    weight=-0.1,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=mdp.go2_stand_still_penalty,
    weight=-1.0,
    params={"command_name": "twist"},
  )
  cfg.rewards["contact_without_command"] = RewardTermCfg(
    func=mdp.go2_contact_without_command,
    weight=1.0,
    params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
  )
  cfg.rewards["collision"] = RewardTermCfg(
    func=mdp.go2_special_collision_penalty,
    weight=-1.0,
    params={"sensor_names": tuple(sorted(collision_sensor_names))},
  )
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].interval_range_s = (4.0, 4.0)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.4, 0.4),
      "y": (-0.4, 0.4),
      "z": (0.0, 0.0),
      "roll": (-0.6, 0.6),
      "pitch": (-0.6, 0.6),
      "yaw": (-0.6, 0.6),
    }
  _go2_source_observation_clipping(cfg)
  return cfg
