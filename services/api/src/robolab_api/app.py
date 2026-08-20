"""Loopback-only FastAPI application for the local RoboLab control plane."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from robolab_core import LocalWorker, create_job_run, default_action_registry
from robolab_api.health import health
from robolab_api.store import LocalStore


def create_app(*, data_dir: str | Path = "var", workspace: str | Path = "skills", vendor_root: str | Path = "vendor/unitree_rl_mjlab") -> FastAPI:
    store = LocalStore(data_dir)
    app = FastAPI(title="RoboLab", version="0.1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.store = store
    app.state.workspace = Path(workspace)
    app.state.vendor_root = Path(vendor_root)
    app.state.actions = default_action_registry()
    # The Worker remains a separate child-process executor; keeping this
    # handle on the control plane makes serve's ownership explicit without
    # embedding any third-party Skill in the API process.
    app.state.worker = LocalWorker()

    @app.get("/api/v1/actions")
    def list_actions() -> list[dict[str, Any]]:
        return app.state.actions.describe()

    @app.get("/api/v1/health")
    def get_health() -> dict[str, object]:
        report = health(store.data_dir, app.state.vendor_root)
        report["worker"] = {"ok": True, "mode": "local-subprocess", "rootExecution": False}
        return report

    @app.get("/api/v1/skills")
    def list_skills() -> list[dict[str, Any]]:
        from robolab_core import scan_skill_workspace
        return [item.to_dict() for item in scan_skill_workspace(app.state.workspace)]

    @app.get("/api/v1/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return store.list_jobs()

    @app.post("/api/v1/jobs", status_code=201)
    def create_job(payload: dict[str, Any]) -> dict[str, Any]:
        action = payload.get("action")
        parameters = payload.get("parameters", {})
        if not isinstance(action, str) or not action or not isinstance(parameters, dict):
            raise HTTPException(422, "需要 action 字符串和 parameters object")
        paths = create_job_run(store.runs_dir, action=action, parameters=parameters, allowed_paths=payload.get("allowedPaths", []))
        import json
        store.register_created(paths.run_dir.name, action, paths.run_dir, json.loads(paths.input.read_text(encoding="utf-8")))
        store.refresh_runs()
        return store.get_job(paths.run_dir.name) or {"id": paths.run_dir.name, "status": "CREATED"}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job 不存在")
        return job

    @app.get("/api/v1/jobs/{job_id}/logs")
    def job_logs(job_id: str) -> dict[str, str]:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job 不存在")
        run_dir = Path(job["run_dir"])
        return {"stdout": (run_dir / "stdout.log").read_text(encoding="utf-8") if (run_dir / "stdout.log").is_file() else "", "stderr": (run_dir / "stderr.log").read_text(encoding="utf-8") if (run_dir / "stderr.log").is_file() else "", "events": (run_dir / "events.jsonl").read_text(encoding="utf-8") if (run_dir / "events.jsonl").is_file() else ""}

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        job = store.cancel_job(job_id)
        if job is None:
            raise HTTPException(404, "Job 不存在")
        return job

    @app.get("/api/v1/artifacts")
    def list_artifacts() -> list[dict[str, Any]]:
        store.refresh_runs()
        return store.list_artifacts()

    @app.get("/api/v1/artifacts/{sha256}")
    def get_artifact(sha256: str) -> FileResponse:
        path = store.artifact_path(sha256)
        if path is None or not path.is_file():
            raise HTTPException(404, "Artifact 不存在")
        return FileResponse(path, filename=sha256)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    return app
