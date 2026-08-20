"""Discover vendor task IDs without importing vendor code into the API."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    source: str
    command: tuple[str, ...]


_TASK_RE = re.compile(r"^\s*(?:\d+\s*\|\s*)?([A-Za-z0-9_.:/-]+)\s*$")


def discover_tasks(vendor_root: str | Path, *, keyword: str | None = None, python: str | None = None, timeout: float = 30.0) -> list[TaskInfo]:
    """Run the vendor's read-only list_envs entry point and parse task IDs."""
    root = Path(vendor_root).resolve()
    script = root / "scripts" / "list_envs.py"
    if not script.is_file():
        raise FileNotFoundError(f"MJLab discovery entry point not found: {script}")
    command = (python or sys.executable, str(script), *((["--keyword", keyword] if keyword else [])))
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"MJLab task discovery failed ({completed.returncode}): {completed.stderr.strip()}")
    tasks: list[TaskInfo] = []
    for line in completed.stdout.splitlines():
        # PrettyTable output is normally ``| 1 | Task-ID |``.  Retain a
        # fallback for simple one-task-per-line implementations.
        columns = [part.strip() for part in line.split("|") if part.strip()]
        candidate = columns[-1] if len(columns) >= 2 and columns[0].isdigit() else line
        match = _TASK_RE.match(candidate)
        if not match or match.group(1).lower() in {"task", "task-id", "available", "environments", "info"}:
            continue
        task_id = match.group(1)
        if keyword and keyword.lower() not in task_id.lower():
            continue
        tasks.append(TaskInfo(task_id, "vendor/unitree_rl_mjlab", command))
    return tasks
