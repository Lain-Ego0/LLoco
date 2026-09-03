"""Unitree Go2 asset and actuator configuration for mjlab."""

from copy import deepcopy
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

GO2_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_go2" / "xmls" / "go2.xml"
)
assert GO2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  """Load the Go2 MJCF from its package-local mesh directory."""
  return mujoco.MjSpec.from_file(str(GO2_XML))


# These values follow the source Go2 gym configuration.  The XML contains the
# robot structure; position actuators are injected by mjlab.
GO2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_hip_joint",),
  stiffness=20.0,
  damping=0.5,
  effort_limit=23.7,
  armature=0.00448,
)
GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_thigh_joint",),
  stiffness=20.0,
  damping=0.5,
  effort_limit=23.7,
  armature=0.00448,
)
GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_calf_joint",),
  stiffness=20.0,
  damping=0.5,
  effort_limit=45.43,
  armature=0.00448,
)


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.42),
  joint_pos={
    "FL_hip_joint": 0.1,
    "RL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "FR_thigh_joint": 0.8,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
  },
  joint_vel={".*": 0.0},
)

_FOOT_REGEX = r"^[FR][LR]_foot_collision$"
_COLLISION_REGEX = r".*_collision\d*$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_COLLISION_REGEX,),
  contype=1,
  conaffinity=1,
  condim={_FOOT_REGEX: 6, _COLLISION_REGEX: 1},
  priority={_FOOT_REGEX: 1, ".*": 0},
  friction={_FOOT_REGEX: (0.6,)},
)

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GO2_ACTUATOR_HIP, GO2_ACTUATOR_THIGH, GO2_ACTUATOR_CALF),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_robot_cfg() -> EntityCfg:
  """Return a fresh Go2 entity configuration."""
  return EntityCfg(
    # These dataclasses are intentionally copied: task-specific configs may
    # adjust the initial pose or actuator gains without mutating later tasks.
    init_state=deepcopy(INIT_STATE),
    collisions=(deepcopy(FULL_COLLISION),),
    spec_fn=get_spec,
    articulation=deepcopy(GO2_ARTICULATION),
  )


GO2_ACTION_SCALE: dict[str, float] = {}
for actuator in GO2_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  assert actuator.effort_limit is not None
  for pattern in actuator.target_names_expr:
    GO2_ACTION_SCALE[pattern] = 0.25
