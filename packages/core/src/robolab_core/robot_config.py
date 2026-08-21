"""Profile + task binding resolution and reproducible snapshot generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from robolab_core.documents import load_document, validate_document
from robolab_core.motion import MJLAB_UPSTREAM_REVISION, resolve_toolchain_identity


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_robot_config(profile_path: str | Path, repo_root: str | Path, task_id: str | None = None) -> dict[str, Any]:
    profile_file = Path(profile_path).resolve()
    repo = Path(repo_root).resolve()
    profile = load_document(profile_file)
    report = validate_document(profile, package_dir=profile_file.parent, source=str(profile_file))
    if not report.ok:
        raise ValueError(report.render())
    selected = task_id or profile["bindings"]["tasks"][0]["id"]
    bindings = {item["id"]: item for item in profile["bindings"]["tasks"]}
    if selected not in bindings:
        raise ValueError(f"unknown task binding: {selected}")
    binding = bindings[selected]
    mjcf = profile_file.parent / profile["description"]["mjcf"]
    mapping = profile_file.parent / profile["description"]["mappings"]
    for label, path in (("mjcf", mjcf), ("mappings", mapping)):
        if not path.is_file():
            raise ValueError(f"{label} path does not exist: {path}")
    identity = resolve_toolchain_identity(repo)
    snapshot: dict[str, Any] = {
        "schema": "robolab.robot-config-snapshot.v1",
        "toolchain": {"robolab": identity.to_dict(), "mjlabUpstreamRevision": MJLAB_UPSTREAM_REVISION},
        "profile": {"id": profile["metadata"]["id"], "version": profile["metadata"]["version"], "sha256": _sha256(profile_file)},
        "task": binding,
        "model": {"path": str(mjcf.relative_to(repo)), "sha256": _sha256(mjcf)},
        "mapping": {"path": str(mapping.relative_to(repo)), "sha256": _sha256(mapping)},
        "resolved": {"control": profile["control"], "rootBody": profile["description"]["rootBody"], "feet": profile["description"]["feet"], "contactFrames": profile["description"]["contactFrames"]},
    }
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    snapshot["snapshotSha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return snapshot


def write_snapshot(snapshot: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path
