"""Preview the saved motion, preserving root and joint pose conventions."""

import numpy as np
import pytest

from lloco.workbench.motion_library import LIBRARY
from lloco.workbench.npz_preview import load_motion


@pytest.mark.parametrize("dof", [23, 29])
def test_motion_pose_and_rate(tmp_path, dof):
  path = tmp_path / LIBRARY / "motion.npz"
  path.parent.mkdir(parents=True)
  positions = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
  rotations = np.array([[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]])
  joints = np.arange(2 * dof).reshape(2, dof) / 100
  np.savez(
    path, fps=[50], joint_pos=joints, body_pos_w=positions, body_quat_w=rotations
  )
  result = load_motion(tmp_path, str(path))
  assert len(result["frames"]) == 2
  assert result["frames"][1]["time"] == 0.02
  assert result["frames"][0]["qpos"][:7] == [1, 2, 3, 1, 0, 0, 0]
  assert result["frames"][1]["qpos"][7:] == joints[1].tolist()
  assert ("g1_23dof.xml" in result["model"]) == (dof == 23)


def test_missing_npz_fields(tmp_path):
  path = tmp_path / LIBRARY / "wrong.npz"
  path.parent.mkdir(parents=True)
  np.savez(path, data=[1, 2, 3])
  with pytest.raises(ValueError, match="Tracking"):
    load_motion(tmp_path, str(path))
