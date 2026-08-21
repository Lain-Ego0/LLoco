"""B2.3 and B2.4 immutable Skill installation tests."""

import json
import shutil
import subprocess

import pytest

from robolab_core import install_skill, review_skill, uninstall_skill


def _git_package(source, destination):
    shutil.copytree(source, destination)
    subprocess.run(["git", "init", "-q", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(destination), "config", "user.name", "Contract Test"], check=True)
    subprocess.run(["git", "-C", str(destination), "add", "."], check=True)
    subprocess.run(["git", "-C", str(destination), "commit", "-qm", "fixture"], check=True)


def test_install_records_revision_and_content_hash(tmp_path, platform_skill):
    import yaml

    source_manifest = tmp_path / "fixture" / "skill.yaml"
    source_manifest.parent.mkdir()
    source_manifest.write_text(yaml.safe_dump(platform_skill), encoding="utf-8")
    (source_manifest.parent / "LICENSE").write_text("MIT", encoding="utf-8")
    (source_manifest.parent / "README.md").write_text("fixture", encoding="utf-8")
    source = tmp_path / "source"
    _git_package(source_manifest.parent, source)
    result = install_skill(source, tmp_path / "installed")
    assert len(result.revision) == 40
    assert result.destination.is_dir()
    registry = json.loads((tmp_path / "installed/registry.json").read_text())
    assert registry[0]["content_sha256"] == result.content_sha256


def test_same_identity_with_different_content_is_rejected(tmp_path, platform_skill):
    import yaml

    package = tmp_path / "fixture"
    package.mkdir()
    (package / "skill.yaml").write_text(yaml.safe_dump(platform_skill), encoding="utf-8")
    (package / "LICENSE").write_text("MIT", encoding="utf-8")
    (package / "README.md").write_text("fixture", encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"
    _git_package(package, first)
    _git_package(package, second)
    (second / "README.md").write_text("changed\n")
    subprocess.run(["git", "-C", str(second), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(second), "commit", "-qm", "changed"], check=True)
    installed = tmp_path / "installed"
    install_skill(first, installed)
    with pytest.raises(ValueError, match="Immutable install conflict"):
        install_skill(second, installed)


def test_uninstall_preserves_referenced_content(tmp_path, platform_skill):
    import yaml

    package = tmp_path / "fixture"
    package.mkdir()
    (package / "skill.yaml").write_text(yaml.safe_dump(platform_skill), encoding="utf-8")
    (package / "LICENSE").write_text("MIT", encoding="utf-8")
    (package / "README.md").write_text("fixture", encoding="utf-8")
    source = tmp_path / "source"
    _git_package(package, source)
    installed = tmp_path / "installed"
    result = install_skill(source, installed)
    registry_path = installed / "registry.json"
    registry = json.loads(registry_path.read_text())
    registry[0]["referenced_by"] = ["job-123"]
    registry_path.write_text(json.dumps(registry))
    assert uninstall_skill(result.skill_id, result.version, installed) == []
    assert result.destination.is_dir()


def test_review_never_executes_environment_commands(tmp_path, platform_skill):
    manifest = tmp_path / "skill.yaml"
    manifest.write_text(__import__("yaml").safe_dump(platform_skill))
    (tmp_path / "LICENSE").write_text("MIT")
    (tmp_path / "README.md").write_text("x")
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas/inspect.input.json").write_text("{}")
    (tmp_path / "schemas/inspect.output.json").write_text("{}")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/smoke.yaml").write_text("x")
    assert review_skill(manifest)["executed"] is False
