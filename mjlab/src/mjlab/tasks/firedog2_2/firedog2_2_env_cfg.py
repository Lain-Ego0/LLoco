"""Minimal generic joint-position task binding for FireDog 2.2.

This module reuses MJLab managers and MDP terms. It does not copy a vendor task,
runner, FSM, driver, or policy checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, time_out
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

_MODEL = Path(__file__).resolve().parents[5] / "robots/firedog2.2.SLDASM/model/firedog2_2.xml"
_ROBOT = SceneEntityCfg("robot", joint_names=(".*",))
_JOINTS = tuple(
  f"{leg}{index if index else ''}_joint"
  for leg in ("RF", "RR", "LR", "LF")
  for index in range(4)
)


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
