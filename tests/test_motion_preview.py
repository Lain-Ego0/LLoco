"""Incremental preview framing, progress publication and pose validation."""

import json

import numpy as np
import pytest

from lloco.workbench.motion_preview import Preview, read_preview


def test_incremental_preview_and_partial_record(tmp_path, monkeypatch):
  monkeypatch.setenv("LLOCO_PREVIEW_DIR", str(tmp_path))
  preview = Preview()
  preview.update(stage="retargeting", total=3, fps=30)
  preview.frame(0, 30, np.arange(36, dtype=float), [[0, 0, 1]])
  preview.update(processed=1)
  first = read_preview(tmp_path, 0)
  assert first["info"]["processed"] == 1
  assert first["frames"][0]["qpos"] == list(range(36))
  with (tmp_path / "frames.jsonl").open("ab") as stream:
    stream.write(b'{"index": 1')
  partial = read_preview(tmp_path, first["cursor"])
  assert partial["cursor"] == first["cursor"]
  assert partial["frames"] == []
  with (tmp_path / "frames.jsonl").open("ab") as stream:
    stream.write(b"}\n")
  assert read_preview(tmp_path, first["cursor"])["frames"] == [{"index": 1}]
  preview.update(stage="complete", processed=3)
  assert json.loads((tmp_path / "status.json").read_text())["stage"] == "complete"
  with pytest.raises(ValueError):
    read_preview(tmp_path, -1)


def test_invalid_pose_does_not_change_model(tmp_path):
  from test_workbench_models import URDF, uploaded

  from lloco.workbench.models import Models

  models = Models()
  initial = models.load(
    tmp_path, dict(selected="a.urdf", files=[uploaded("a.urdf", URDF)])
  )
  with pytest.raises(ValueError):
    models.pose(dict(id=initial["id"], qpos=[0, 1]))
  pose = models.pose(dict(id=initial["id"], qpos=[0.5]))
  assert pose["axes"][0]["value"] == pytest.approx(0.5)
