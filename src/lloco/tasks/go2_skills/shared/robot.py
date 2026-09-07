"""Shared Go2 robot entity factories."""

from copy import deepcopy

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg

from lloco.assets.robots import get_go2_robot_cfg


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
