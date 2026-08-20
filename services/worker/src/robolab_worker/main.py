"""Minimal CLI worker for a pre-created robolab-job-v1 run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from robolab_core.jobs import JobPaths, LocalWorker


def run_job(run_dir: str | Path, command: list[str]) -> dict[str, object]:
    root = Path(run_dir).resolve()
    paths = JobPaths(root, root / "input.json", root / "events.jsonl", root / "result.json", root / "stdout.log", root / "stderr.log", root / "artifacts")
    if not paths.input.is_file() or not paths.artifacts.is_dir():
        raise ValueError(f"无效 Job run 目录: {root}")
    worker = LocalWorker()
    handle = worker.start(paths, command, cwd=root)
    while handle.process.poll() is None:
        time.sleep(0.05)
    return handle.finalize() or {"status": "FAILED"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robolab-worker")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.command:
        parser.error("需要提供 -- 后的 Skill 命令")
    try:
        result = run_job(args.run_dir, args.command)
    except (OSError, ValueError, PermissionError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "SUCCEEDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
