"""B1.6 lint tests for licenses and artifact SHA-256 checks."""

from __future__ import annotations

import copy

from robolab_core import lint_skill_package, validate_document
from robolab_core.lint import verify_artifacts


def _rules(issues) -> set[str]:
    return {i.rule for i in issues}


class TestLicenseRules:
    def test_known_spdx_passes(self, motion_skill):
        issues = lint_skill_package(motion_skill)
        assert not [i for i in issues if i.rule == "lint.license-unknown"]

    def test_unknown_spdx_warns(self, motion_skill):
        doc = copy.deepcopy(motion_skill)
        doc["metadata"]["license"] = "Custom-1.0"
        assert "lint.license-unknown" in _rules(lint_skill_package(doc))

    def test_license_file_required(self, motion_skill, tmp_path):
        issues = lint_skill_package(motion_skill, package_dir=tmp_path)
        assert "lint.license-file-missing" in _rules(issues)


class TestRevisionPinning:
    def test_floating_main_rejected(self, motion_skill):
        doc = copy.deepcopy(motion_skill)
        doc["metadata"]["source"]["revision"] = "main"
        issues = lint_skill_package(doc)
        error = next(i for i in issues if i.rule == "lint.revision-floating")
        assert error.severity == "error"

    def test_branch_ref_rejected(self, motion_skill):
        doc = copy.deepcopy(motion_skill)
        doc["metadata"]["source"]["revision"] = "refs/heads/feature-x"
        assert "lint.revision-floating" in _rules(lint_skill_package(doc))

    def test_commit_sha_accepted(self, platform_skill):
        assert lint_skill_package(platform_skill) == []

    def test_tag_warns_but_passes(self, motion_skill):
        # The real g1_velocity package uses refs/tags/... and should warn only.
        issues = lint_skill_package(motion_skill)
        warnings = [i for i in issues if i.rule == "lint.revision-tag"]
        assert warnings and warnings[0].severity == "warning"
        assert not [i for i in issues if i.severity == "error"]


class TestArtifactVerification:
    def test_missing_file(self, motion_skill, tmp_path):
        issues = verify_artifacts(motion_skill["spec"]["artifacts"], tmp_path)
        assert "lint.artifact-missing" in _rules(issues)

    def test_hash_mismatch(self, motion_skill, tmp_path):
        artifact = motion_skill["spec"]["artifacts"][0]
        (tmp_path / "artifacts").mkdir()
        (tmp_path / artifact["path"]).write_bytes(b"corrupted")
        issues = verify_artifacts([artifact], tmp_path)
        assert "lint.artifact-hash" in _rules(issues)

    def test_size_mismatch(self, motion_skill, tmp_path):
        artifact = copy.deepcopy(motion_skill["spec"]["artifacts"][0])
        artifact["size"] = 999
        (tmp_path / "artifacts").mkdir()
        (tmp_path / artifact["path"]).write_bytes(b"0123456789")
        issues = verify_artifacts([artifact], tmp_path)
        assert "lint.artifact-size" in _rules(issues)


class TestReferencedFiles:
    def test_missing_action_schema(self, platform_skill, tmp_path):
        (tmp_path / "LICENSE").write_text("MIT")
        (tmp_path / "README.md").write_text("x")
        issues = lint_skill_package(platform_skill, package_dir=tmp_path)
        assert "lint.action-schema-missing" in _rules(issues)

    def test_missing_readme(self, platform_skill, tmp_path):
        (tmp_path / "LICENSE").write_text("MIT")
        issues = lint_skill_package(platform_skill, package_dir=tmp_path)
        assert "lint.readme-missing" in _rules(issues)

    def test_conda_environment_file_required(self, platform_skill, tmp_path):
        doc = copy.deepcopy(platform_skill)
        doc["spec"]["runtime"]["environment"] = {"mode": "conda", "file": "environment.yml"}
        (tmp_path / "LICENSE").write_text("MIT")
        (tmp_path / "README.md").write_text("x")
        (tmp_path / "schemas").mkdir()
        (tmp_path / "schemas/inspect.input.json").write_text("{}")
        (tmp_path / "schemas/inspect.output.json").write_text("{}")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests/smoke.yaml").write_text("smoke: true")
        issues = lint_skill_package(doc, package_dir=tmp_path)
        assert "lint.environment-missing" in _rules(issues)


class TestRealCatalogPackage:
    """Run full validation for the real g1_velocity package when present."""

    def test_g1_velocity_package_passes(self, g1_velocity_package_dir):
        from robolab_core import load_document

        document = load_document(g1_velocity_package_dir / "skill.yaml")
        report = validate_document(document, package_dir=g1_velocity_package_dir)
        assert report.ok, report.render()
        artifact_notes = [i for i in report.issues if i.rule == "lint.artifact-ok"]
        assert len(artifact_notes) == 2
