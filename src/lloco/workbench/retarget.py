"""Project-local LAFAN1 BVH → GMR IK → LLoco NPZ pipeline."""

import re
import tempfile
from pathlib import Path


def retarget(source: Path, output: Path, preview=None):
  import numpy as np

  from lloco.motion_conversion import G1_JOINT_NAMES, convert_csv_to_npz

  from .gmr.motion_retarget import GeneralMotionRetargeting
  from .gmr.utils.lafan1 import load_lafan1_file

  match = re.search(r"Frame Time:\s*([0-9.eE+-]+)", source.read_text())
  if not match or float(match[1]) <= 0:
    raise ValueError("BVH 缺少有效 Frame Time")
  frames, height = load_lafan1_file(str(source))
  if len(frames) < 2:
    raise ValueError("BVH 至少需要两帧")
  if preview:
    from .gmr.utils.lafan_vendor.extract import read_bvh

    skeleton = read_bvh(str(source))
    preview.update(
      stage="retargeting",
      total=len(frames),
      fps=1 / float(match[1]),
      bones=skeleton.bones,
      parents=[int(p) for p in skeleton.parents],
      source=source.name,
      output=str(output),
    )
  solver = GeneralMotionRetargeting("bvh", "unitree_g1", height, solver="daqp")
  import mujoco

  names = [
    mujoco.mj_id2name(solver.model, mujoco.mjtObj.mjOBJ_JOINT, j)
    for j in range(solver.model.njnt)
    if solver.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
  ]
  if tuple(names) != G1_JOINT_NAMES:
    raise ValueError("GMR 模型关节顺序与 LLoco G1 不匹配")
  poses = []
  for index, frame in enumerate(frames):
    poses.append(solver.retarget(frame).copy())
    if preview and (
      index % max(1, round(1 / float(match[1]) / 15)) == 0 or index == len(frames) - 1
    ):
      preview.frame(
        index,
        1 / float(match[1]),
        poses[-1],
        [frame[name][0].tolist() for name in skeleton.bones],
      )
      preview.update(processed=index + 1)
    if index % 30 == 0:
      print(f"GMR retarget: {index + 1}/{len(frames)}", flush=True)
  if preview:
    preview.update(stage="converting", processed=len(frames))
  qpos = np.asarray(poses)
  # MuJoCo WXYZ → LLoco's CSV XYZW.
  csv = np.concatenate((qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]), axis=1)
  output.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="lloco-gmr-") as temporary:
    path = Path(temporary) / "motion.csv"
    np.savetxt(path, csv, delimiter=",")
    convert_csv_to_npz(
      "g1", str(path), str(output), input_fps=1 / float(match[1]), device="cpu"
    )
  print(f"GMR → LLoco: {output}", flush=True)
