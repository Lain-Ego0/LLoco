"""Read the saved LLoco G1 tracking motion for direct pose playback."""

from pathlib import Path

from .motion_library import LIBRARY
from .services import inside


def load_motion(root: Path, value: str):
  import numpy as np

  path = inside(root, value, ".npz")
  if not path.is_relative_to((root / LIBRARY).resolve()):
    raise ValueError("请选择动作资产库中的 NPZ")
  with np.load(path, allow_pickle=False) as motion:
    required = {"fps", "joint_pos", "body_pos_w", "body_quat_w"}
    if not required.issubset(motion.files):
      raise ValueError("需要 LLoco Tracking 格式的 NPZ")
    joints = motion["joint_pos"]
    positions = motion["body_pos_w"]
    rotations = motion["body_quat_w"]
    rate = np.asarray(motion["fps"])
    if rate.size != 1 or not np.isfinite(rate).all() or float(rate.reshape(-1)[0]) <= 0:
      raise ValueError("NPZ 帧率无效")
    fps = float(rate.reshape(-1)[0])
    if joints.ndim != 2 or joints.shape[1] not in (23, 29) or not len(joints):
      raise ValueError("目前支持 G1 29 / 23-DoF Tracking NPZ")
    count = len(joints)
    if (
      positions.ndim != 3
      or positions.shape[0] != count
      or positions.shape[1] < 1
      or positions.shape[2] != 3
      or rotations.shape != (*positions.shape[:2], 4)
    ):
      raise ValueError("NPZ 根位置 / 四元数形状不匹配")
    # LLoco converter writes native joint order; body 0 is the robot root.
    qpos = np.concatenate((positions[:, 0], rotations[:, 0], joints), axis=1)
    if not np.isfinite(qpos).all() or np.any(
      np.linalg.norm(qpos[:, 3:7], axis=1) < 1e-8
    ):
      raise ValueError("NPZ 包含无效姿态数值")
    robot = "g1" if joints.shape[1] == 29 else "g1_23dof"
    return dict(
      model=f"src/lloco/assets/robots/unitree_g1/xmls/{robot}.xml",
      info=dict(
        stage="complete", processed=count, total=count, fps=fps, source=path.name
      ),
      frames=[
        dict(index=i, time=i / fps, qpos=values.tolist())
        for i, values in enumerate(qpos)
      ],
    )
