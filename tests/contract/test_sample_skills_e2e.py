"""B4 sample Skills: install, validate, execute and Codex export."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from robolab_cli.main import main


REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO.parent / "RoboLab-Skill" / "skills"


def test_three_sample_skills_install_and_motion_is_compatible(tmp_path, capsys):
    installed = tmp_path / "installed"
    packages = [CATALOG / "motion/g1_velocity", CATALOG / "platform/mjcf_inspector", CATALOG / "agent/robot_onboarding"]
    for package in packages:
        assert main(["skill", "install", str(package), "--installed-root", str(installed)]) == 0
    assert main(["check", "--skill", str(packages[0] / "skill.yaml"), "--profile", str(REPO / "robots/unitree.g1.29dof/profile.yaml"), "--package-dir", str(packages[0])]) == 0
    assert len(json.loads((installed / "registry.json").read_text())) == 3


def test_inspector_job_and_agent_export(tmp_path, monkeypatch):
    inspector = CATALOG / "platform/mjcf_inspector" / "skill.yaml"
    model = REPO / "vendor/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml"
    assert main(["skill", "run", str(inspector), "--params", json.dumps({"mjcf_path": str(model)}), "--runs-root", str(tmp_path / "runs"), "--wait"]) == 0
    result_path = next((tmp_path / "runs").glob("*/result.json"))
    result = json.loads(result_path.read_text())
    assert result["status"] == "SUCCEEDED"
    assert {entry["path"] for entry in result["artifacts"]} == {"report.json", "report.md", "robot_profile.draft.yaml"}
    assert main(["agent", "export", str(CATALOG / "agent/robot_onboarding"), "--target-root", str(tmp_path / "agent-skills")]) == 0
    exported = tmp_path / "agent-skills/robot-onboarding"
    assert (exported / "SKILL.md").is_file()
    assert "activate_real" in (exported / "agents/openai.yaml").read_text()
