"""Loopback-only FastAPI application for the local RoboLab control plane."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import sys
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from robolab_core import LocalWorker, create_job_run, default_action_registry
from robolab_api.health import health
from robolab_api.store import LocalStore


def _watch_job(handle: Any) -> None:
    while handle.process.poll() is None:
        time.sleep(0.1)
    handle.finalize()


def create_app(*, data_dir: str | Path = "var", workspace: str | Path = "skills", vendor_root: str | Path = "vendor/unitree_rl_mjlab", web_root: str | Path = "apps/web/dist") -> FastAPI:
    store = LocalStore(data_dir)
    app = FastAPI(title="RoboLab", version="0.1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.store = store
    app.state.workspace = Path(workspace)
    app.state.vendor_root = Path(vendor_root)
    app.state.actions = default_action_registry()
    app.state.web_root = Path(web_root).resolve()
    if (app.state.web_root / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=app.state.web_root / "assets"), name="web-assets")
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
        from robolab_core import load_document, scan_skill_workspace
        result = []
        for entry in scan_skill_workspace(app.state.workspace):
            item = entry.to_dict()
            if not entry.error:
                try:
                    document = load_document(entry.path)
                    item["actions"] = document["spec"]["actions"]
                    item["permissions"] = document["spec"]["permissions"]
                    item["license"] = document["metadata"]["license"]
                    item["runtime"] = document["spec"]["runtime"]
                except (OSError, ValueError, KeyError):
                    item["error"] = "无法读取 manifest 详情"
            result.append(item)
        return result

    @app.post("/api/v1/skills/install", status_code=201)
    def install_skill(payload: dict[str, Any]) -> dict[str, Any]:
        from robolab_core import install_skill as install
        source = payload.get("source")
        if not isinstance(source, str):
            raise HTTPException(422, "source 必须是本地 catalog checkout 内的路径")
        source_path = Path(source).resolve()
        catalog_root = Path("../RoboLab-Skill").resolve()
        if catalog_root not in source_path.parents:
            raise HTTPException(403, "只允许从本地 RoboLab-Skill catalog 安装")
        try:
            return install(source_path, app.state.workspace / "installed").to_dict()
        except (OSError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/skills/{skill_id}/actions/{action}:invoke", status_code=201)
    def invoke_skill_action(skill_id: str, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Invoke a PlatformSkill through the same isolated LocalWorker as CLI."""
        from robolab_core import load_document, scan_skill_workspace
        match = next((item for item in scan_skill_workspace(app.state.workspace) if item.skill_id == skill_id and not item.error), None)
        if match is None:
            raise HTTPException(404, "Skill 未安装")
        package = Path(match.path).parent.resolve()
        document = load_document(package / "skill.yaml")
        if document["kind"] != "PlatformSkill" or action not in document["spec"]["actions"]:
            raise HTTPException(422, "当前仅支持已声明的 PlatformSkill action")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise HTTPException(422, "parameters 必须是 object")
        entrypoint = document["spec"]["runtime"]["entrypoint"]
        module = entrypoint.get("module")
        command = [sys.executable, "-m", module] if module else entrypoint["command"].split()
        paths = create_job_run(store.runs_dir, action=f"{skill_id}.{action}", parameters=parameters, allowed_paths=[package])
        input_data = json.loads(paths.input.read_text(encoding="utf-8"))
        store.register_created(paths.run_dir.name, input_data["action"], paths.run_dir, input_data)
        try:
            handle = app.state.worker.start(paths, command, cwd=package, env={"PYTHONPATH": str(package / "src") + ":" + os.environ.get("PYTHONPATH", "")})
            threading.Thread(target=_watch_job, args=(handle,), daemon=True).start()
            store.mark_running(paths.run_dir.name)
        except (OSError, ValueError, PermissionError) as exc:
            raise HTTPException(422, f"无法启动 Skill: {exc}") from exc
        return store.get_job(paths.run_dir.name) or {"id": paths.run_dir.name, "status": "RUNNING"}

    @app.get("/api/v1/robots")
    def list_robots() -> list[dict[str, Any]]:
        import yaml
        from robolab_core import check_compatibility, load_document, scan_skill_workspace, validate_document
        robots_root = Path("robots")
        result: list[dict[str, Any]] = []
        for profile_path in sorted(robots_root.glob("*/profile.yaml")):
            try:
                document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
                metadata = document.get("metadata", {})
                description = document.get("description", {})
                compatibility = []
                for skill in scan_skill_workspace(app.state.workspace):
                    if skill.error:
                        continue
                    try:
                        skill_doc = load_document(skill.path)
                        profile_doc = load_document(profile_path)
                        if not validate_document(skill_doc, package_dir=Path(skill.path).parent).ok:
                            continue
                        verdict = check_compatibility(skill_doc, profile_doc)
                        compatibility.append({"skillId": skill.skill_id, "version": skill.version, "compatible": verdict.ok, "reasons": [issue.message for issue in verdict.issues]})
                    except (OSError, ValueError, KeyError):
                        continue
                result.append({"id": metadata.get("id"), "name": metadata.get("name") or metadata.get("id"), "version": metadata.get("version"), "maturity": "simulation-only" if not document.get("capabilities", {}).get("physicalDeployment", False) else "physical-capable", "capabilities": [name for name, enabled in document.get("capabilities", {}).items() if enabled], "path": str(profile_path), "compatibility": compatibility})
            except (OSError, ValueError, yaml.YAMLError):
                continue
        return result

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
        built = app.state.web_root / "index.html"
        if built.is_file():
            return built.read_text(encoding="utf-8")
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    return app
