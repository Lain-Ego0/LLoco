"""Local Skill installation with content-addressed immutable copies."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robolab_core.documents import load_document, validate_document


@dataclass(frozen=True)
class InstallResult:
    skill_id: str
    version: str
    content_sha256: str
    revision: str
    destination: Path
    already_installed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "content_sha256": self.content_sha256,
            "revision": self.revision,
            "destination": str(self.destination),
            "already_installed": self.already_installed,
        }


def uninstall_skill(
    skill_id: str,
    version: str,
    installed_root: str | Path,
    *,
    content_sha256: str | None = None,
) -> list[str]:
    """Remove unreferenced installed copies and preserve referenced history."""
    root = Path(installed_root).resolve()
    registry_path = root / "registry.json"
    registry = _load_registry(registry_path)
    removed: list[str] = []
    kept: list[dict[str, Any]] = []
    matched = False
    for entry in registry:
        same_identity = entry.get("skill_id") == skill_id and entry.get("version") == version
        same_hash = content_sha256 is None or entry.get("content_sha256") == content_sha256
        if not same_identity or not same_hash:
            kept.append(entry)
            continue
        matched = True
        if entry.get("referenced_by"):
            kept.append(entry)
            continue
        destination = Path(entry["destination"]).resolve()
        if root not in destination.parents:
            raise ValueError(f"Registry destination escapes installed root: {destination}")
        if destination.exists():
            shutil.rmtree(destination)
        removed.append(entry["content_sha256"])
    if not matched:
        raise KeyError(f"Skill is not installed: {skill_id}@{version}")
    registry_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return removed


def _git_root(path: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"Unable to resolve a full Git revision for {root}")
    return revision


def _content_hash(package_dir: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in package_dir.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(package_dir).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Skill registry must be a JSON array: {path}")
    return data


def install_skill(source: str | Path, installed_root: str | Path) -> InstallResult:
    """Validate and install one local Skill package immutably."""
    source_path = Path(source).resolve()
    package_dir = source_path.parent if source_path.is_file() else source_path
    manifest_path = package_dir / "skill.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Skill manifest not found: {manifest_path}")
    document = load_document(manifest_path)
    report = validate_document(document, package_dir=package_dir, source=str(manifest_path))
    if not report.ok:
        raise ValueError(report.render())
    repository_root = _git_root(package_dir)
    revision = _git_revision(repository_root)
    metadata = document["metadata"]
    skill_id = metadata["id"]
    version = metadata["version"]
    content_sha256 = _content_hash(package_dir)
    root = Path(installed_root).resolve()
    destination = root / skill_id / version / content_sha256
    registry_path = root / "registry.json"
    registry = _load_registry(registry_path)
    matches = [
        entry for entry in registry
        if entry.get("skill_id") == skill_id and entry.get("version") == version
    ]
    if matches and any(entry.get("content_sha256") != content_sha256 for entry in matches):
        raise ValueError(f"Immutable install conflict for {skill_id}@{version}")
    if destination.exists():
        return InstallResult(skill_id, version, content_sha256, revision, destination, True)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package_dir, destination, symlinks=True)
    entry = InstallResult(skill_id, version, content_sha256, revision, destination, False).to_dict()
    registry.append(entry)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return InstallResult(skill_id, version, content_sha256, revision, destination, False)
