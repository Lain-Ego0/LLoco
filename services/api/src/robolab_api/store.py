"""Local SQLite metadata and content-addressed artifact storage."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import sqlite3
from pathlib import Path
from typing import Any


class LocalStore:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.artifact_dir = self.data_dir / "artifacts"
        self.runs_dir = self.data_dir / "runs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.data_dir / "robolab.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, action TEXT NOT NULL, status TEXT NOT NULL,
              run_dir TEXT NOT NULL, input_json TEXT NOT NULL, result_json TEXT,
              created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
              sha256 TEXT PRIMARY KEY, size INTEGER NOT NULL, path TEXT NOT NULL,
              media_type TEXT, created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifact_lineage (
              artifact_sha256 TEXT NOT NULL, job_id TEXT NOT NULL,
              relative_path TEXT NOT NULL, PRIMARY KEY (artifact_sha256, job_id, relative_path)
            );
        """)
        self.db.commit()

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for part in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(part)
        return digest.hexdigest()

    def put_artifact(self, source: str | Path, *, media_type: str | None = None) -> dict[str, Any]:
        source_path = Path(source).resolve()
        digest = self._hash(source_path)
        destination = self.artifact_dir / digest[:2] / digest
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copy2(source_path, destination)
        size = source_path.stat().st_size
        import time
        self.db.execute("INSERT OR IGNORE INTO artifacts (sha256, size, path, media_type, created_at) VALUES (?, ?, ?, ?, ?)", (digest, size, str(destination), media_type, time.time()))
        self.db.commit()
        return {"sha256": digest, "size": size, "path": str(destination), "mediaType": media_type}

    def refresh_runs(self) -> None:
        import time
        for input_path in self.runs_dir.glob("*/input.json"):
            try:
                input_data = json.loads(input_path.read_text(encoding="utf-8"))
                job_id = input_data["jobId"]
                result_path = input_path.parent / "result.json"
                result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
                existing = self.db.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
                status = result["status"] if result else (existing["status"] if existing else "RUNNING")
                now = time.time()
                self.db.execute("INSERT INTO jobs (id, action, status, run_dir, input_json, result_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, result_json=excluded.result_json, updated_at=excluded.updated_at", (job_id, input_data["action"], status, str(input_path.parent), json.dumps(input_data, ensure_ascii=False), json.dumps(result, ensure_ascii=False) if result else None, now, now))
                if result:
                    for artifact in result.get("artifacts", []):
                        file_path = input_path.parent / "artifacts" / artifact["path"]
                        if file_path.is_file():
                            registered = self.put_artifact(file_path)
                            self.db.execute("INSERT OR IGNORE INTO artifact_lineage (artifact_sha256, job_id, relative_path) VALUES (?, ?, ?)", (registered["sha256"], job_id, artifact["path"]))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        self.db.commit()

    def register_created(self, job_id: str, action: str, run_dir: Path, input_data: dict[str, Any]) -> None:
        import time
        now = time.time()
        self.db.execute("INSERT INTO jobs (id, action, status, run_dir, input_json, result_json, created_at, updated_at) VALUES (?, ?, 'CREATED', ?, ?, NULL, ?, ?)", (job_id, action, str(run_dir), json.dumps(input_data, ensure_ascii=False), now, now))
        self.db.commit()

    def mark_running(self, job_id: str) -> None:
        import time
        self.db.execute("UPDATE jobs SET status = 'RUNNING', updated_at = ? WHERE id = ?", (time.time(), job_id))
        self.db.commit()

    def list_jobs(self) -> list[dict[str, Any]]:
        self.refresh_runs()
        return [dict(row) for row in self.db.execute("SELECT id, action, status, run_dir, created_at, updated_at FROM jobs ORDER BY created_at DESC")]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        self.refresh_runs()
        row = self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["input"] = json.loads(item.pop("input_json"))
        item["result"] = json.loads(item.pop("result_json")) if item["result_json"] else None
        return item

    def list_artifacts(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.execute("SELECT sha256, size, media_type, created_at FROM artifacts ORDER BY created_at DESC")]

    def artifact_path(self, digest: str) -> Path | None:
        row = self.db.execute("SELECT path FROM artifacts WHERE sha256 = ?", (digest,)).fetchone()
        return Path(row["path"]) if row else None

    def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        """Signal the saved process group and record a terminal cancellation."""
        job = self.get_job(job_id)
        if job is None:
            return None
        if job["status"] in {"SUCCEEDED", "FAILED", "CANCELLED", "REJECTED"}:
            return job
        run_dir = Path(job["run_dir"])
        pid = None
        event_path = run_dir / "events.jsonl"
        if event_path.is_file():
            for line in event_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                    if event.get("type") == "job.started":
                        pid = event.get("pid")
                except json.JSONDecodeError:
                    continue
        if isinstance(pid, int):
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        result = {"protocol": "robolab-job-v1", "jobId": job_id, "status": "CANCELLED", "exitCode": None, "message": "通过本地 API 请求取消"}
        (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"type": "job.cancelled"}, ensure_ascii=False) + "\n")
        self.refresh_runs()
        return self.get_job(job_id)
