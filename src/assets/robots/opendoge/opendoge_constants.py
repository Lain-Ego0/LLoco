"""OpenDoge (OpenDog V1.1) constants."""

from copy import deepcopy
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from src import PACKAGE_ROOT

##
# MJCF and assets.
##

OPENDOGE_XML: Path = (
  PACKAGE_ROOT / "assets" / "robots" / "opendoge" / "xmls" / "opendoge.xml"
)
assert OPENDOGE_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(OPENDOGE_XML))


##
# Actuator config.
##

OPENDOGE_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=12.0,
  damping=0.5,
  effort_limit=6.0,
  armature=0.005,
)
OPENDOGE_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=12.0,
  damping=0.5,
  effort_limit=6.0,
  armature=0.005,
)
OPENDOGE_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=12.0,
  damping=0.5,
  effort_limit=9.0,
  armature=0.005,
)

##
# Keyframes.
##

OPENDOGE_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.158),
  joint_pos={
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.6,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": -0.8,
    "FR_calf_joint": 1.6,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.8,
    "RL_calf_joint": -1.6,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": -0.8,
    "RR_calf_joint": 1.6,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet. Feet self-collisions are also
# disabled; the velocity task only needs foot-ground contacts.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.8,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collision geoms except self-collisions. Feet receive the
# contact parameters needed for stable ground contact.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1, ".*": 0},
  friction={_foot_regex: (0.8,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

OPENDOGE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    OPENDOGE_ACTUATOR_HIP,
    OPENDOGE_ACTUATOR_THIGH,
    OPENDOGE_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_opendoge_robot_cfg() -> EntityCfg:
  """Get an independent OpenDoge robot configuration instance."""
  return deepcopy(
    EntityCfg(
      init_state=OPENDOGE_INIT_STATE,
      collisions=(FULL_COLLISION,),
      spec_fn=get_spec,
      articulation=OPENDOGE_ARTICULATION,
    )
  )
