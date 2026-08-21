"""B1.5 ``robolab check`` exit codes and explainable output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from robolab_cli.main import EXIT_CHECK_FAILED, EXIT_OK, EXIT_USAGE_ERROR, main
from conftest import FIXTURES


def _make_motion_skill_package(root: Path) -> Path:
    """Build a minimal complete MotionSkill package with real hashes."""
    policy = b"fake-onnx-policy-bytes"
    deploy = b"fake-deploy-params"
    manifest = {
        "apiVersion": "robolab.dev/v1alpha1",
        "kind": "MotionSkill",
        "metadata": {
            "id": "motion.test-velocity",
            "name": "Test Velocity",
            "version": "0.1.0",
            "description": "Synthetic test skill.",
            "license": "MIT",
            "source": {
                "repository": "https://example.invalid/catalog",
                "revision": "0123456789abcdef0123456789abcdef01234567",
            },
        },
        "spec": {
            "compatibility": {
                "platform": ">=0.1.0 <0.2.0",
                "skillApi": "v1alpha1",
                "robots": [{"profile": "test.reference_biped", "version": ">=1.0.0 <2.0.0"}],
                "controlMode": "joint_position",
                "controlHz": 50,
                "jointSet": "test.reference_biped.joints.v1",
                "observationSchema": "test.observation.v1",
                "actionSchema": "test.action.v1",
            },
            "runtime": {"type": "onnx", "runner": "rl_policy", "protocol": "robolab-motion-v1"},
            "artifacts": [
                {
                    "name": "policy",
                    "path": "artifacts/policy.onnx",
                    "mediaType": "application/onnx",
                    "size": len(policy),
                    "sha256": hashlib.sha256(policy).hexdigest(),
                },
                {
                    "name": "deploy-params",
                    "path": "params/deploy.yaml",
                    "mediaType": "application/yaml",
                    "sha256": hashlib.sha256(deploy).hexdigest(),
                },
            ],
            "actions": {
                "play": {
                    "title": "Play in MJLab",
                    "inputSchema": "schemas/command.json",
                    "task": "test.reference_biped.velocity.flat",
                }
            },
            "permissions": {
                "filesystem": {"read": ["skill", "robot_profile"], "write": ["run"]},
                "network": False,
                "subprocess": False,
                "robotState": True,
                "robotCommand": True,
            },
            "safety": {
                "maturity": "experimental",
                "defaultTarget": "simulation",
                "requiredGates": ["offline", "mjlab_play", "sim_to_sim"],
                "fallbackState": "damping",
                "realRobotRequiresExplicitConfirmation": True,
            },
            "validation": {"smoke": "tests/smoke.yaml"},
        },
    }
    (root / "artifacts").mkdir(parents=True)
    (root / "params").mkdir()
    (root / "schemas").mkdir()
    (root / "tests").mkdir()
    (root / "artifacts/policy.onnx").write_bytes(policy)
    (root / "params/deploy.yaml").write_bytes(deploy)
    (root / "schemas/command.json").write_text("{}")
    (root / "tests/smoke.yaml").write_text("smoke: true")
    (root / "LICENSE").write_text("MIT License")
    (root / "README.md").write_text("# Test Velocity")
    skill_path = root / "skill.yaml"
    skill_path.write_text(yaml.safe_dump(manifest, allow_unicode=True))
    return skill_path


@pytest.fixture()
def motion_skill_package(tmp_path) -> Path:
    return _make_motion_skill_package(tmp_path / "pkg")


class TestSingleDocumentCheck:
    def test_valid_profile_exit_ok(self, capsys):
        assert main(["check", str(FIXTURES / "robot_profile.reference_biped.yaml")]) == EXIT_OK
        assert "通过" in capsys.readouterr().out

    def test_valid_joint_set_exit_ok(self, capsys):
        assert main(["check", str(FIXTURES / "joint_set.reference_biped.yaml")]) == EXIT_OK
        assert "无问题" in capsys.readouterr().out

    def test_broken_document_exit_1(self, tmp_path, capsys):
        broken = tmp_path / "broken.yaml"
        doc = yaml.safe_load((FIXTURES / "robot_profile.reference_biped.yaml").read_text())
        del doc["capabilities"]
        broken.write_text(yaml.safe_dump(doc))
        assert main(["check", str(broken)]) == EXIT_CHECK_FAILED
        assert "capabilities" in capsys.readouterr().out

    def test_missing_file_exit_2(self, capsys):
        assert main(["check", "does-not-exist.yaml"]) == EXIT_USAGE_ERROR
        assert "无法读取" in capsys.readouterr().out

    def test_no_args_exit_2(self, capsys):
        assert main(["check"]) == EXIT_USAGE_ERROR
        assert "错误" in capsys.readouterr().err


class TestCompatibilityCheck:
    def test_compatible_pair(self, motion_skill_package, capsys):
        code = main([
            "check",
            "--skill", str(motion_skill_package),
            "--profile", str(FIXTURES / "robot_profile.reference_biped.yaml"),
        ])
        assert code == EXIT_OK
        out = capsys.readouterr().out
        assert "jointSet 一致" in out
        assert "SHA-256 校验通过" in out

    def test_incompatible_pair_exit_1(self, motion_skill_package, tmp_path, capsys):
        profile = yaml.safe_load((FIXTURES / "robot_profile.reference_biped.yaml").read_text())
        profile["control"]["frequencyHz"] = 200
        modified = tmp_path / "profile.yaml"
        modified.write_text(yaml.safe_dump(profile))
        code = main([
            "check",
            "--skill", str(motion_skill_package),
            "--profile", str(modified),
        ])
        assert code == EXIT_CHECK_FAILED
        out = capsys.readouterr().out
        assert "compat.control-hz" in out

    def test_skill_without_profile_exit_2(self, motion_skill_package, capsys):
        code = main(["check", "--skill", str(motion_skill_package)])
        assert code == EXIT_USAGE_ERROR

    def test_json_output_machine_readable(self, motion_skill_package, capsys):
        code = main([
            "check",
            "--skill", str(motion_skill_package),
            "--profile", str(FIXTURES / "robot_profile.reference_biped.yaml"),
            "--json",
        ])
        assert code == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert all("issues" in r for r in payload["reports"])

    def test_artifacts_are_reported_once(self, motion_skill_package, capsys):
        assert main([
            "check",
            "--skill", str(motion_skill_package),
            "--profile", str(FIXTURES / "robot_profile.reference_biped.yaml"),
        ]) == EXIT_OK
        assert capsys.readouterr().out.count("lint.artifact-ok") == 2
