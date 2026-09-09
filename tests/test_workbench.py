"""Workbench boundaries and command contracts; no training runtime required."""

import functools
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from lloco.workbench.catalog import ALGORITHMS
from lloco.workbench.server import Handler
from lloco.workbench.services import Jobs, command, inside


@pytest.fixture
def project(tmp_path):
  (tmp_path / "logs").mkdir()
  (tmp_path / "logs/model_1.pt").write_bytes(b"checkpoint")
  (tmp_path / "motion.npz").write_bytes(b"motion")
  return tmp_path


def body(kind="velocity"):
  group = ALGORITHMS[kind]
  algorithm = next(iter(group))
  return dict(kind=kind, group=kind, algorithm=algorithm, task=group[algorithm][0])


def test_train_and_tracking_contract(project):
  args = command(
    project, "train", {**body(), "iterations": 2, "save_interval": 1, "num_envs": 4}
  )
  assert args[args.index("--agent.max-iterations") + 1] == "2"
  assert args[args.index("--agent.logger") + 1] == "tensorboard"
  with pytest.raises(ValueError):
    command(project, "train", body("tracking"))
  args = command(project, "train", {**body("tracking"), "motion": "motion.npz"})
  assert "--env.commands.motion.motion-file" in args
  args = command(project, "viser", {**body(), "checkpoint": "logs/model_1.pt"})
  assert args[args.index("--checkpoint") + 1] == str(project / "logs/model_1.pt")


@pytest.mark.parametrize("value", [0, -1, True, "1.5", "--help"])
def test_reject_bad_numbers(project, value):
  with pytest.raises(ValueError):
    command(project, "train", {**body(), "num_envs": value})


def test_selection_and_paths(project, tmp_path):
  with pytest.raises(ValueError):
    command(project, "train", {**body("tracking"), "group": "go2_skill"})
  with pytest.raises(ValueError):
    inside(project, "../outside.pt")
  (project / "escape").symlink_to("/tmp")
  with pytest.raises(ValueError):
    inside(project, "escape/file.pt")


def test_job_exit_log_and_history(project):
  jobs = Jobs(project)
  record = jobs.start(
    "test", [sys.executable, "-c", 'print("finished"); raise SystemExit(3)'], {}
  )
  deadline = time.monotonic() + 5
  while jobs.list()[0]["running"] and time.monotonic() < deadline:
    time.sleep(0.02)
  assert jobs.list()[0]["returncode"] == 3
  assert "finished" in jobs.log(record["id"])
  assert Jobs(project).list()[0]["returncode"] == 3


def test_http_static_and_invalid_requests(project):
  jobs = Jobs(project)
  server = ThreadingHTTPServer(
    ("127.0.0.1", 0), functools.partial(Handler, root=project, jobs=jobs)
  )
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  base = f"http://127.0.0.1:{server.server_port}"
  try:
    with urllib.request.urlopen(base + "/api/state") as response:
      assert json.load(response)["artifacts"][0]["kind"] == "pt"
    with urllib.request.urlopen(base + "/app.js") as response:
      assert "javascript" in response.headers["Content-Type"]
    request = urllib.request.Request(
      base + "/api/train",
      data=b"{}",
      headers={"Content-Type": "application/json", "Origin": "https://other.example"},
    )
    with pytest.raises(urllib.error.HTTPError) as error:
      urllib.request.urlopen(request)
    assert error.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as error:
      urllib.request.urlopen(base + "/missing.js")
    assert error.value.code == 404
  finally:
    server.shutdown()
    server.server_close()
    jobs.close()


def test_stop_process_group(project):
  jobs = Jobs(project)
  record = jobs.start("test", [sys.executable, "-c", "import time; time.sleep(60)"], {})
  jobs.stop(record["id"])
  deadline = time.monotonic() + 5
  while jobs.list()[0]["running"] and time.monotonic() < deadline:
    time.sleep(0.02)
  assert not jobs.list()[0]["running"]
  assert jobs.list()[0]["stop_requested"]
  assert jobs.list()[0]["returncode"] != 0
