"""Authoritative policy-facing Unitree Go2 robot specification."""

from lainloco.core import RobotSpec

GO2_JOINT_ORDER = (
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
)

GO2 = RobotSpec(
  robot_id="unitree/go2",
  asset_factory="mjlab.asset_zoo.robots.unitree_go2.go2_constants:get_go2_robot_cfg",
  joint_order=GO2_JOINT_ORDER,
  base_body="base_link",
  foot_sites=("FL", "FR", "RL", "RR"),
  collision_geoms=(
    "base1_collision",
    "base2_collision",
    "base3_collision",
    *(
      f"{leg}_{part}_collision"
      for leg in ("FL", "FR", "RL", "RR")
      for part in ("hip", "thigh", "calf1", "calf2", "foot")
    ),
  ),
  default_pose=(
    ("FL_hip_joint", 0.1),
    ("FL_thigh_joint", 0.8),
    ("FL_calf_joint", -1.5),
    ("FR_hip_joint", -0.1),
    ("FR_thigh_joint", 0.8),
    ("FR_calf_joint", -1.5),
    ("RL_hip_joint", 0.1),
    ("RL_thigh_joint", 1.0),
    ("RL_calf_joint", -1.5),
    ("RR_hip_joint", -0.1),
    ("RR_thigh_joint", 1.0),
    ("RR_calf_joint", -1.5),
  ),
  action_scale=(0.25,) * 12,
  physics_dt=0.005,
  control_dt=0.02,
  # Hardware order is deliberately unset until the hardware safety plan
  # validates the SDK mapping. Sim-to-sim must continue using joint_order.
  hardware_joint_mapping=(),
)
