from __future__ import annotations

import sys
from pathlib import Path

from robolab_mjlab_adapter import build_play_command, discover_tasks


def test_build_play_command_uses_vendor_entrypoint(tmp_path):
    script = tmp_path / "scripts" / "play.py"
    script.parent.mkdir()
    script.write_text("", encoding="utf-8")
    command = build_play_command(tmp_path, "Unitree-G1-Flat", {"num_envs": 2, "video": True})
    assert command[2:] == ["Unitree-G1-Flat", "--num-envs", "2", "--video"]


def test_discover_tasks_parses_vendor_prettytable_output(tmp_path):
    script = tmp_path / "scripts" / "list_envs.py"
    script.parent.mkdir()
    script.write_text("print('| 1 | Unitree-G1-Flat |')\nprint('| 2 | Unitree-Go2-Flat |')", encoding="utf-8")
    assert [item.task_id for item in discover_tasks(tmp_path, python=sys.executable)] == ["Unitree-G1-Flat", "Unitree-Go2-Flat"]


def test_discover_tasks_exposes_vendor_root_on_pythonpath(tmp_path):
    script = tmp_path / "scripts" / "list_envs.py"
    script.parent.mkdir()
    script.write_text("import os\nassert os.environ['PYTHONPATH'].split(os.pathsep)[0] == os.getcwd()\nprint('| 1 | Unitree-G1-Flat |')", encoding="utf-8")
    assert [item.task_id for item in discover_tasks(tmp_path, python=sys.executable)] == ["Unitree-G1-Flat"]
