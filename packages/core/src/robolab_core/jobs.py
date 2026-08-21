"""The file based ``robolab-job-v1`` protocol and local process Worker.

Executable Skills never import the platform process.  They receive a private
run directory and communicate only through the files defined here.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

JOB_PROTOCOL = "robolab-job-v1"
JOB_STATES = frozenset({"CREATED", "VALIDATING", "QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", "REJECTED"})


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class JobPaths:
    """All paths owned by one run; callers may expose no other writable path."""

    run_dir: Path
    input: Path
    events: Path
    result: Path
    stdout: Path
    stderr: Path
    artifacts: Path


def create_job_run(
    runs_root: str | Path,
    *,
    action: str,
    parameters: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    allowed_paths: Sequence[str | Path] = (),
    platform_version: str = "0.1.0",
    job_id: str | None = None,
) -> JobPaths:
    """Create the immutable input side of a protocol run."""
    identifier = job_id or str(uuid.uuid4())
    if not identifier or Path(identifier).name != identifier:
        raise ValueError("job_id 必须是单个安全路径段")
    run_dir = Path(runs_root).resolve() / identifier
    if run_dir.exists():
        raise FileExistsError(f"Job run 已存在: {run_dir}")
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    paths = JobPaths(run_dir, run_dir / "input.json", run_dir / "events.jsonl", run_dir / "result.json", run_dir / "stdout.log", run_dir / "stderr.log", artifacts)
    _write_json(paths.input, {
        "protocol": JOB_PROTOCOL,
        "jobId": identifier,
        "action": action,
        "parameters": dict(parameters),
        "metadata": dict(metadata or {}),
        "allowedPaths": [str(Path(item).resolve()) for item in allowed_paths],
        "platformVersion": platform_version,
    })
    paths.events.touch()
    return paths


def append_event(paths: JobPaths, event: Mapping[str, Any]) -> None:
    """Append one structured event.  A timestamp is supplied if omitted."""
    payload = {"timestamp": time.time(), **dict(event)}
    if "type" not in payload:
        raise ValueError("事件必须包含 type")
    with paths.events.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_events(paths: JobPaths) -> list[dict[str, Any]]:
    return [json.loads(line) for line in paths.events.read_text(encoding="utf-8").splitlines() if line]


def _artifact_records(artifacts: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in sorted(artifacts.rglob("*")):
        if item.is_file():
            records.append({"path": item.relative_to(artifacts).as_posix(), "sha256": _sha256(item), "size": item.stat().st_size})
    return records


@dataclass
class JobHandle:
    paths: JobPaths
    process: subprocess.Popen[bytes]
    started_at: float
    cancelled: bool = False

    def cancel(self, timeout: float = 5.0) -> None:
        """Terminate the complete process group, escalating after ``timeout``."""
        if self.process.poll() is not None:
            return
        self.cancelled = True
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()

    def finalize(self) -> dict[str, Any] | None:
        code = self.process.poll()
        if code is None:
            return None
        supplied: dict[str, Any] = {}
        if self.paths.result.exists():
            try:
                supplied = json.loads(self.paths.result.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                supplied = {"status": "FAILED", "message": "Skill 写入了无效 result.json"}
        status = "CANCELLED" if self.cancelled else ("SUCCEEDED" if code == 0 and supplied.get("status", "SUCCEEDED") == "SUCCEEDED" else "FAILED")
        result = {
            "protocol": JOB_PROTOCOL,
            "jobId": self.paths.run_dir.name,
            "status": status,
            "exitCode": code,
            "startedAt": self.started_at,
            "finishedAt": time.time(),
            "artifacts": _artifact_records(self.paths.artifacts),
            **{key: value for key, value in supplied.items() if key not in {"protocol", "jobId", "status", "exitCode", "artifacts"}},
        }
        _write_json(self.paths.result, result)
        append_event(self.paths, {"type": "job.finished", "status": status, "exitCode": code})
        return result


class LocalWorker:
    """Starts untrusted executable Skills in a separate process group."""

    def start(self, paths: JobPaths, command: Sequence[str], *, cwd: str | Path | None = None, env: Mapping[str, str] | None = None) -> JobHandle:
        if os.name != "posix":
            raise RuntimeError("B3 LocalWorker 当前只支持 POSIX 进程组")
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            raise PermissionError("拒绝以 root 运行第三方 Skill；请使用普通用户启动 RoboLab")
        if not command:
            raise ValueError("Job command 不能为空")
        child_env = os.environ.copy()
        child_env.update(env or {})
        # Keep the names from SKILL_SPEC and the explicit JOB_* aliases for
        # early clients that adopted the draft protocol.
        child_env.update({
            "ROBOLAB_INPUT_FILE": str(paths.input),
            "ROBOLAB_RUN_DIR": str(paths.run_dir),
            "ROBOLAB_ARTIFACTS_DIR": str(paths.artifacts),
            "ROBOLAB_EVENTS_FILE": str(paths.events),
            "ROBOLAB_JOB_INPUT": str(paths.input),
            "ROBOLAB_JOB_RUN_DIR": str(paths.run_dir),
            "ROBOLAB_JOB_ARTIFACTS": str(paths.artifacts),
            "ROBOLAB_JOB_EVENTS": str(paths.events),
        })
        stdout = paths.stdout.open("wb")
        stderr = paths.stderr.open("wb")
        try:
            process = subprocess.Popen(list(command), cwd=cwd, env=child_env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
        finally:
            stdout.close()
            stderr.close()
        append_event(paths, {"type": "job.started", "pid": process.pid, "command": list(command)})
        return JobHandle(paths, process, time.time())
