"""Shared Go2 robot entity factories."""

from copy import deepcopy

import mujoco
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg

from lloco.assets.robots import get_go2_robot_cfg
from lloco.assets.robots.unitree_go2.go2_constants import GO2_XML


def trot_robot_cfg():
  # The public asset factory currently reuses its InitialStateCfg.  Copy the
  # whole entity locally so skill overrides cannot mutate Flat/Rough.
  cfg = deepcopy(get_go2_robot_cfg())
  cfg.init_state.pos = (0.0, 0.0, 0.42)
  cfg.init_state.joint_pos = {
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    ".*thigh_joint": 0.8,
    ".*calf_joint": -1.5,
  }
  cfg.articulation = EntityArticulationInfoCfg(
    actuators=(
      IdealPdActuatorCfg(
        target_names_expr=(".*hip_joint",),
        stiffness=20.0,
        damping=0.5,
        effort_limit=23.7,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*thigh_joint",),
        stiffness=20.0,
        damping=0.5,
        effort_limit=23.7,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*calf_joint",),
        stiffness=20.0,
        damping=0.5,
        effort_limit=35.55,
        armature=0.0,
      ),
    ),
    soft_joint_pos_limit_factor=0.9,
  )
  return cfg


def jump_robot_cfg():
  cfg = deepcopy(get_go2_robot_cfg())
  cfg.init_state.pos = (0.0, 0.0, 0.42)
  cfg.init_state.joint_pos = {
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    ".*calf_joint": -1.5,
  }
  cfg.articulation = EntityArticulationInfoCfg(
    actuators=(
      IdealPdActuatorCfg(
        target_names_expr=(".*hip_joint",),
        stiffness=20.0,
        damping=0.5,
        effort_limit=23.7,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*thigh_joint",),
        stiffness=20.0,
        damping=0.5,
        effort_limit=23.7,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*calf_joint",),
        stiffness=20.0,
        damping=0.5,
        effort_limit=35.55,
        armature=0.0,
      ),
    ),
    soft_joint_pos_limit_factor=0.9,
  )
  return cfg


def spring_jump_robot_cfg():
  """Gym spring-jump has Jump's asymmetric pose but starts at 0.39 m."""
  cfg = jump_robot_cfg()
  cfg.init_state.pos = (0.0, 0.0, 0.39)
  return cfg


def dreamwaq_robot_cfg():
  """DreamWaQ uses Jump's asymmetric nominal stance and 20/.5 PD gains."""
  return jump_robot_cfg()


def _cts_go2_spec() -> mujoco.MjSpec:
  """Load the exact source-URDF collision shapes with Gym capsule conversion."""
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  for geom in spec.geoms:
    if not geom.name or "collision" not in geom.name:
      continue
    # The shared MJCF has a 0.213 m thigh box, whereas the CTS source URDF
    # specifies 0.11 m.  MuJoCo stores box half-sizes.
    if geom.name.endswith("_thigh_collision"):
      geom.size[0] = 0.055
    if geom.name.endswith("_foot_collision"):
      geom.pos[0] = -0.002
    if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
      geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
  # The shared MJCF omits the third lower-calf collision present in the URDF.
  for leg in ("FL", "FR", "RL", "RR"):
    calf = next(body for body in spec.bodies if body.name == f"{leg}_calf")
    calf.add_geom(
      name=f"{leg}_calf3_collision",
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      pos=(0.00801333, 0.0, -0.18745022),
      quat=(0.9650925, 0.0, 0.26190927, 0.0),
      size=(0.0155, 0.015),
    )
  return spec


def cts_robot_cfg():
  """Robot model matching the source CTS asset and actuator parameters."""
  cfg = jump_robot_cfg()
  cfg.spec_fn = _cts_go2_spec
  # Isaac Gym enables self-collision, but directly setting MuJoCo
  # conaffinity=1 is not equivalent: a controlled gate collapses at iteration
  # 30 while the same model without MuJoCo self-contact recovers.  Keep this
  # backend-specific collision filter until pair-level Gym filters are mapped.
  cfg.collisions[0].conaffinity = 0
  cfg.articulation = EntityArticulationInfoCfg(
    actuators=tuple(
      IdealPdActuatorCfg(
        target_names_expr=(joint_expr,),
        stiffness=20.0,
        damping=0.5,
        effort_limit=effort_limit,
        armature=0.00448,
      )
      for joint_expr, effort_limit in (
        (".*hip_joint", 23.7),
        (".*thigh_joint", 23.7),
        (".*calf_joint", 35.55),
      )
    ),
    soft_joint_pos_limit_factor=0.9,
  )
  return cfg


def rear_stand_robot_cfg():
  cfg = deepcopy(get_go2_robot_cfg())
  cfg.init_state.pos = (0.0, 0.0, 0.42)
  cfg.init_state.joint_pos = {
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    ".*calf_joint": -1.5,
  }
  cfg.articulation = EntityArticulationInfoCfg(
    actuators=(
      IdealPdActuatorCfg(
        target_names_expr=(".*hip_joint",),
        stiffness=40.0,
        damping=1.0,
        effort_limit=21.33,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*thigh_joint",),
        stiffness=40.0,
        damping=1.0,
        effort_limit=21.33,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*calf_joint",),
        stiffness=40.0,
        damping=1.0,
        effort_limit=31.995,
        armature=0.0,
      ),
    ),
    soft_joint_pos_limit_factor=0.9,
  )
  return cfg


def handstand_robot_cfg():
  cfg = deepcopy(get_go2_robot_cfg())
  cfg.init_state.pos = (0.0, 0.0, 0.42)
  cfg.init_state.joint_pos = {
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    ".*calf_joint": -1.5,
  }
  cfg.articulation = EntityArticulationInfoCfg(
    actuators=(
      IdealPdActuatorCfg(
        target_names_expr=(".*hip_joint",),
        stiffness=40.0,
        damping=1.0,
        effort_limit=21.33,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*thigh_joint",),
        stiffness=40.0,
        damping=1.0,
        effort_limit=21.33,
        armature=0.0,
      ),
      IdealPdActuatorCfg(
        target_names_expr=(".*calf_joint",),
        stiffness=40.0,
        damping=1.0,
        effort_limit=31.995,
        armature=0.0,
      ),
    ),
    soft_joint_pos_limit_factor=0.9,
  )
  return cfg
