"""CPU-only contract tests for the B3 file protocol and process isolation."""

from __future__ import annotations

import json
import sys
import time

from robolab_core import Action, ActionRegistry, LocalWorker, create_job_run, read_events
import robolab_core.jobs as jobs


def test_worker_creates_reproducible_run_and_result(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs.os, "geteuid", lambda: 1000)
    paths = create_job_run(tmp_path, action="test.echo", parameters={"value": 7}, allowed_paths=[tmp_path])
    script = "import os, pathlib; pathlib.Path(os.environ['ROBOLAB_ARTIFACTS_DIR'], 'out.txt').write_text('ok')"
    handle = LocalWorker().start(paths, [sys.executable, "-c", script])
    handle.process.wait(timeout=5)
    result = handle.finalize()

    assert result and result["status"] == "SUCCEEDED"
    assert json.loads(paths.input.read_text())["protocol"] == "robolab-job-v1"
    assert result["artifacts"][0]["path"] == "out.txt"
    assert {event["type"] for event in read_events(paths)} == {"job.started", "job.finished"}


def test_worker_cancel_ends_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs.os, "geteuid", lambda: 1000)
    paths = create_job_run(tmp_path, action="test.wait", parameters={})
    handle = LocalWorker().start(paths, [sys.executable, "-c", "import time; time.sleep(30)"])
    time.sleep(0.05)
    handle.cancel(timeout=1)
    result = handle.finalize()
    assert result and result["status"] == "CANCELLED"


def test_action_registry_is_single_explicit_dispatch_path():
    registry = ActionRegistry()
    registry.register(Action("robolab.test", "test action", lambda values: values["x"] + 1, {"type": "object"}))
    assert registry.invoke("robolab.test", {"x": 2}) == 3
    assert registry.describe()[0]["id"] == "robolab.test"
