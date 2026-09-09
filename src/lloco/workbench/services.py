"""Validated commands and local process lifecycle for the workbench."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .catalog import ALGORITHMS, TASKS


def inside(root: Path, value: str, suffix: str | None = None) -> Path:
  if not isinstance(value, str) or not value.strip():
    raise ValueError("请填写项目内文件路径")
  path = (root / Path(value).expanduser()).resolve()
  if not path.is_relative_to(root.resolve()):
    raise ValueError("文件必须位于 LLoco 项目目录内")
  if suffix and (path.suffix.lower() != suffix or not path.is_file()):
    raise ValueError(f"需要已存在的 {suffix} 文件：{value}")
  return path


def positive(body: dict, key: str, default: int) -> str:
  value = body.get(key, default)
  if isinstance(value, bool) or str(value).isdigit() is False or int(value) < 1:
    raise ValueError(f"{key} 必须是正整数")
  return str(int(value))


def selection(body: dict) -> tuple[str, str]:
  kind = body.get("kind", "velocity")
  group = body.get("group", kind)
  if kind not in TASKS or group not in (kind, "go2_skill"):
    raise ValueError("无效策略分支")
  if group == "go2_skill" and kind != "velocity":
    raise ValueError("Go2 Skill 仅用于 Velocity 四足分支")
  algorithm = body.get("algorithm")
  task = body.get("task")
  if task not in ALGORITHMS.get(group, {}).get(algorithm, []):
    raise ValueError("算法与任务不匹配")
  return kind, task


def command(root: Path, action: str, body: dict) -> list[str]:
  python = str(root / ".venv/bin/python")
  if not Path(python).exists():
    python = sys.executable
  worker = [python, "-u", "-m", "lloco.workbench.worker"]
  if action in ("train", "export", "viser"):
    kind, task = selection(body)
    motion = []
    if kind == "tracking":
      motion = [str(inside(root, body.get("motion", ""), ".npz"))]
    if action == "train":
      result = worker + [
        "train",
        task,
        "--env.scene.num-envs",
        positive(body, "num_envs", 4096),
        "--agent.max-iterations",
        positive(body, "iterations", 10000),
        "--agent.save-interval",
        positive(body, "save_interval", 100),
        "--agent.logger",
        "tensorboard",
        "--log-root",
        str(root / "logs/workbench"),
      ]
      if motion:
        result += ["--env.commands.motion.motion-file", *motion]
      return result
    checkpoint = inside(root, body.get("checkpoint", ""), ".pt")
    result = worker + [action, task, "--checkpoint", str(checkpoint)]
    if motion:
      result += ["--motion", *motion]
    return result
  if action in ("gmr-convert", "gmr-retarget"):
    source = inside(
      root, body.get("source", ""), ".pkl" if action == "gmr-convert" else ".bvh"
    )
    output = inside(root, body.get("output", ""))
    if output.suffix != ".npz" or output.exists():
      raise ValueError("输出必须是尚不存在的 .npz 路径")
    return worker + [action, "--source", str(source), "--output", str(output)]
  if action == "tensorboard":
    return [
      python,
      "-m",
      "tensorboard.main",
      "--logdir",
      str(root / "logs"),
      "--host",
      "127.0.0.1",
      "--port",
      "6006",
    ]
  raise ValueError("未知操作")


class Jobs:
  def __init__(self, root: Path):
    self.root = root
    self.directory = root / ".lloco-workbench"
    self.directory.mkdir(exist_ok=True)
    self.lock = threading.RLock()
    self.records: dict[str, dict] = {}
    for path in self.directory.glob("*.json"):
      try:
        record = json.loads(path.read_text())
        if "id" in record and "command" in record:
          record["running"] = False
          record["interrupted"] = record.get("returncode") is None
          self.records[record["id"]] = record
      except (ValueError, OSError):
        continue

  def _save(self, record):
    (self.directory / f"{record['id']}.json").write_text(
      json.dumps({k: v for k, v in record.items() if k != "process"}, indent=2)
    )

  def start(self, action: str, args: list[str], body: dict) -> dict:
    with self.lock:
      for record in self.records.values():
        if record.get("running") and record.get("action") == action:
          raise ValueError(f"已有 {action} 在运行，请先停止")
      identifier = uuid.uuid4().hex
      with (self.directory / f"{identifier}.log").open("wb") as log:
        process = subprocess.Popen(
          args,
          cwd=self.root,
          stdout=log,
          stderr=subprocess.STDOUT,
          start_new_session=True,
          env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
      record = dict(
        id=identifier,
        action=action,
        command=args,
        selection=body,
        started=time.time(),
        running=True,
        returncode=None,
      )
      self.records[identifier] = {**record, "process": process}
      self._save(record)
      threading.Thread(
        target=self._watch, args=(identifier, process), daemon=True
      ).start()
      return record

  def _watch(self, identifier, process):
    code = process.wait()
    with self.lock:
      record = self.records[identifier]
      record.update(running=False, returncode=code)
      self._save(record)

  def list(self):
    with self.lock:
      return [
        {k: v for k, v in r.items() if k != "process"}
        for r in sorted(self.records.values(), key=lambda r: r["started"], reverse=True)
      ]

  def stop(self, identifier: str):
    with self.lock:
      record = self.records.get(identifier)
      if not record:
        raise ValueError("未知任务")
      process = record.get("process")
      if process and process.poll() is None:
        record["stop_requested"] = True
        self._save(record)
        try:
          os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
          return
        threading.Thread(target=self._kill_later, args=(process,), daemon=True).start()

  @staticmethod
  def _kill_later(process):
    try:
      process.wait(timeout=8)
    except subprocess.TimeoutExpired:
      try:
        os.killpg(process.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass

  def close(self):
    for record in self.list():
      if record["running"]:
        self.stop(record["id"])

  def log(self, identifier):
    if identifier not in self.records:
      raise ValueError("未知任务")
    with (self.directory / f"{identifier}.log").open("rb") as stream:
      stream.seek(max(0, stream.seek(0, 2) - 64000))
      return stream.read().decode(errors="replace")
