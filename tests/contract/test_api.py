from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from robolab_api import create_app
from robolab_api.store import LocalStore


def test_content_addressed_store_deduplicates(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")
    store = LocalStore(tmp_path / "var")
    first = store.put_artifact(source, media_type="text/plain")
    second = store.put_artifact(source, media_type="text/plain")
    assert first["sha256"] == second["sha256"]
    assert len(store.list_artifacts()) == 1
    assert Path(first["path"]).read_text() == "hello"


def test_api_health_jobs_and_artifact_download(tmp_path):
    runs = tmp_path / "var/runs/job-1"
    runs.mkdir(parents=True)
    (runs / "input.json").write_text(json.dumps({"protocol": "robolab-job-v1", "jobId": "job-1", "action": "test", "parameters": {}}), encoding="utf-8")
    (runs / "events.jsonl").write_text("", encoding="utf-8")
    (runs / "artifacts").mkdir()
    (runs / "artifacts/out.txt").write_text("artifact", encoding="utf-8")
    (runs / "result.json").write_text(json.dumps({"protocol": "robolab-job-v1", "jobId": "job-1", "status": "SUCCEEDED", "artifacts": [{"path": "out.txt"}]}), encoding="utf-8")
    app = create_app(data_dir=tmp_path / "var", workspace=tmp_path / "skills")
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/motion/tasks").json()[0]["id"] == "robolab.motion.smoke.cartpole"
        robots = client.get("/api/v1/motion/robots").json()
        assert [robot["id"] for robot in robots] == ["community.firedog2_2"]
        assert client.get("/api/v1/motion/toolchain").status_code == 200
        assert len(client.get("/api/v1/actions").json()) >= 4
        robots = client.get("/api/v1/robots")
        assert robots.status_code == 200
        assert client.get("/api/v1/jobs").json()[0]["id"] == "job-1"
        artifacts = client.get("/api/v1/artifacts").json()
        assert len(artifacts) == 1
        assert client.get(f"/api/v1/artifacts/{artifacts[0]['sha256']}").text == "artifact"
        assert client.get("/").status_code == 200


def test_api_serves_built_webui_assets(tmp_path):
    web_root = Path(__file__).resolve().parents[2] / "apps/web/dist"
    if not (web_root / "index.html").is_file():
        pytest.skip("WebUI build not present")
    app = create_app(data_dir=tmp_path / "var", web_root=web_root)
    with TestClient(app) as client:
        assert "RoboLab" in client.get("/").text
        index = (web_root / "index.html").read_text(encoding="utf-8")
        asset = next(item.split('src="')[1].split('"')[0] for item in index.split() if 'src="/assets/' in item)
        assert client.get(asset).status_code == 200


def test_api_creates_reproducible_job_run(tmp_path):
    app = create_app(data_dir=tmp_path / "var")
    with TestClient(app) as client:
        response = client.post("/api/v1/jobs", json={"action": "robolab.jobs.create", "parameters": {"x": 1}})
        assert response.status_code == 201
        job = response.json()
        assert job["status"] == "CREATED"
        assert (tmp_path / "var/runs" / job["id"] / "input.json").is_file()


def test_api_installs_and_invokes_platform_skill(tmp_path):
    catalog = Path(__file__).resolve().parents[3] / "RoboLab-Skill/skills/platform/mjcf_inspector"
    xml = tmp_path / "model.xml"
    xml.write_text("<mujoco><worldbody><body name='base'><joint name='hinge' range='-1 1'/></body></worldbody></mujoco>", encoding="utf-8")
    app = create_app(data_dir=tmp_path / "var", workspace=tmp_path / "skills")
    with TestClient(app) as client:
        installed = client.post("/api/v1/skills/install", json={"source": str(catalog)})
        assert installed.status_code == 201
        invoke = client.post("/api/v1/skills/tools.mjcf-inspector/actions/inspect:invoke", json={"parameters": {"mjcf_path": str(xml)}})
        assert invoke.status_code == 201
        job_id = invoke.json()["id"]
        for _ in range(20):
            result = client.get(f"/api/v1/jobs/{job_id}").json()
            if result["status"] != "RUNNING":
                break
            time.sleep(0.05)
        assert result["status"] == "SUCCEEDED"
