"""Incremental, job-scoped retargeting previews independent of training."""

import json
import os
from pathlib import Path


class Preview:
  def __init__(self):
    value = os.environ.get("LLOCO_PREVIEW_DIR")
    self.directory = Path(value) if value else None
    self.info = dict(stage="loading", processed=0, total=0)
    if self.directory:
      self.directory.mkdir(parents=True, exist_ok=True)
      self.update()

  def update(self, **values):
    self.info.update(values)
    if self.directory:
      temporary = self.directory / "status.tmp"
      temporary.write_text(json.dumps(self.info, allow_nan=False))
      temporary.replace(self.directory / "status.json")

  def frame(self, index, fps, qpos, human=None):
    if not self.directory:
      return
    value = dict(index=index, time=index / fps, qpos=qpos.tolist(), human=human)
    with (self.directory / "frames.jsonl").open("a") as stream:
      stream.write(json.dumps(value, allow_nan=False) + "\n")


def read_preview(directory: Path, cursor: int):
  if cursor < 0:
    raise ValueError("无效预览位置")
  info_path = directory / "status.json"
  info = (
    json.loads(info_path.read_text())
    if info_path.exists()
    else dict(stage="loading", processed=0, total=0)
  )
  frames = []
  path = directory / "frames.jsonl"
  if path.exists():
    with path.open("rb") as stream:
      if cursor > path.stat().st_size:
        raise ValueError("预览位置超出文件范围")
      stream.seek(cursor)
      start = cursor
      while stream.tell() - start < 256 * 1024:
        line = stream.readline()
        if not line.endswith(b"\n"):
          break
        frames.append(json.loads(line))
        cursor = stream.tell()
  return dict(info=info, frames=frames, cursor=cursor)
