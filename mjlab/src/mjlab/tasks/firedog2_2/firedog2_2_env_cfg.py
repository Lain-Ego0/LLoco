"""Minimal generic joint-position task binding for FireDog 2.2.

This module reuses MJLab managers and MDP terms. It does not copy a vendor task,
runner, FSM, driver, or policy checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, time_out
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

_MODEL = Path(__file__).resolve().parents[5] / "robots/firedog2.2.SLDASM/model/firedog2_2.xml"
_ROBOT = SceneEntityCfg("robot", joint_names=(".*",))
_JOINTS = tuple(
  f"{leg}{index if index else ''}_joint"
  for leg in ("RF", "RR", "LR", "LF")
  for index in range(4)
)


def _firedog_upright(env, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Flat-ground upright reward using the root projected gravity vector."""
  gravity_xy = env.scene[asset_cfg.name].data.projected_gravity_b[:, :2]
  return torch.exp(-torch.sum(torch.square(gravity_xy), dim=1) / (std**2))


def _spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(_MODEL))
  # Scene merges its own init_state keyframe; use EntityCfg.InitialStateCfg instead.
  for key in list(spec.keys):
    spec.delete(key)
  return spec


def _robot() -> EntityCfg:
  return EntityCfg(
    spec_fn=_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=_JOINTS, command_field="position"),),
    ),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.55),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos={name: 0.0 for name in _JOINTS},
      joint_vel={name: 0.0 for name in _JOINTS},
    ),
  )


def firedog2_2_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Return the generic 50 Hz FireDog joint-state smoke/tracking environment."""
  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": _ROBOT}),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": _ROBOT}),
      },
      enable_corruption=not play,
    ),
  }
  cfg = ManagerBasedRlEnvCfg(
    decimation=10,
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot()},
      num_envs=1,
      env_spacing=2.0,
    ),
    observations=observations,
    actions={
      "joint_position": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=_JOINTS,
        scale=0.5,
        use_default_offset=True,
      )
    },
    terminations={"time_out": TerminationTermCfg(func=time_out, time_out=True)},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=2.5,
      elevation=-20.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),
    episode_length_s=20.0 if not play else 1e10,
  )
  return cfg


def firedog2_2_velocity_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """FireDog binding for the public flat velocity-tracking task.

  The older factory above remains the R2 load/reset/action smoke environment.
  This binding only supplies FireDog's entity, actuator order, feet/contact
  names, default pose, and control timing to generic velocity MDP terms.
  """
  robot = _robot()
  robot_cfg = SceneEntityCfg("robot", joint_names=_JOINTS)
  foot_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom", pattern=".*3_collision", entity="robot"
    ),
    # The flat-plane terrain is already the only non-robot collision target in
    # this scene, so an unfiltered primary contact is deterministic and avoids
    # coupling the public task to terrain namespace details.
    fields=("found", "force"),
    reduce="maxforce",
    track_air_time=True,
  )

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel,
      params={"asset_cfg": SceneEntityCfg("robot")},
      noise=Unoise(n_min=-0.15, n_max=0.15),
      clip=(-5.0, 5.0),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel,
      params={"asset_cfg": SceneEntityCfg("robot")},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      clip=(-5.0, 5.0),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      params={"asset_cfg": SceneEntityCfg("robot")},
      noise=Unoise(n_min=-0.03, n_max=0.03),
      clip=(-1.2, 1.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": robot_cfg, "biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
      clip=(-3.5, 3.5),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": robot_cfg},
      noise=Unoise(n_min=-0.25, n_max=0.25),
      clip=(-20.0, 20.0),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action, clip=(-1.0, 1.0)),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
      clip=(-2.0, 2.0),
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
      clip=(0.0, 1.0),
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
      nan_policy="error",
    ),
    "critic": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=False,
      nan_policy="error",
    ),
  }

  return ManagerBasedRlEnvCfg(
    decimation=10,
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot},
      sensors=(foot_contact,),
      num_envs=64 if not play else 1,
      env_spacing=2.0,
      extent=2.5,
    ),
    observations=observations,
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=_JOINTS,
        scale=0.5,
        use_default_offset=True,
      )
    },
    commands={
      "twist": UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(2.0, 4.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        rel_forward_envs=0.25,
        heading_command=False,
        debug_vis=not play,
        ranges=UniformVelocityCommandCfg.Ranges(
          lin_vel_x=(-0.8, 1.2),
          lin_vel_y=(-0.4, 0.4),
          ang_vel_z=(-0.6, 0.6),
          heading=None,
        ),
      )
    },
    events={
      "reset_base": EventTermCfg(
        func=envs_mdp.reset_root_state_uniform,
        mode="reset",
        params={
          "pose_range": {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "yaw": (-0.15, 0.15),
          },
          "velocity_range": {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.02, 0.02),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
          },
        },
      ),
      "reset_robot_joints": EventTermCfg(
        func=envs_mdp.reset_joints_by_offset,
        mode="reset",
        params={
          "position_range": (0.0, 0.02),
          "velocity_range": (0.0, 0.02),
          "asset_cfg": robot_cfg,
        },
      ),
    },
    rewards={
      "track_linear_velocity": RewardTermCfg(
        func=mdp.track_linear_velocity,
        weight=2.5,
        params={"command_name": "twist", "std": 0.5},
      ),
      "track_angular_velocity": RewardTermCfg(
        func=mdp.track_angular_velocity,
        weight=1.5,
        params={"command_name": "twist", "std": 0.6},
      ),
      "upright": RewardTermCfg(
        func=_firedog_upright,
        weight=1.0,
        params={"std": 0.35, "asset_cfg": SceneEntityCfg("robot")},
      ),
      "posture": RewardTermCfg(
        func=mdp.posture,
        weight=0.35,
        params={"asset_cfg": robot_cfg, "std": {".*": 0.8}},
      ),
      "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.03),
      "joint_limit": RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": robot_cfg},
      ),
      "alive": RewardTermCfg(func=mdp.is_alive, weight=0.1),
    },
    terminations={
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
      "fell_over": TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.9, "asset_cfg": SceneEntityCfg("robot")},
      ),
      "root_too_low": TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.18, "asset_cfg": SceneEntityCfg("robot")},
      ),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=2.6,
      elevation=-15.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      nconmax=64,
      njmax=1000,
      mujoco=MujocoCfg(timestep=0.002, iterations=10, ls_iterations=20),
    ),
    episode_length_s=20.0 if not play else 1e10,
  )
