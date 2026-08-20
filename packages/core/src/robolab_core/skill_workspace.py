"""Discovery of local Skill workspace entries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SkillEntry:
    """One discoverable Skill manifest in a local workspace."""

    path: str
    source: str
    mutable: bool
    skill_id: str | None
    version: str | None
    kind: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SOURCES = (("builtin", False), ("installed", False), ("dev", True))


def scan_skill_workspace(workspace: str | Path) -> list[SkillEntry]:
    """Discover manifests under builtin, installed, and dev workspace roots."""
    root = Path(workspace)
    entries: list[SkillEntry] = []
    for source, mutable in _SOURCES:
        source_root = root / source
        if not source_root.is_dir():
            continue
        for manifest in sorted(source_root.rglob("skill.yaml")):
            try:
                document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
                metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
                entries.append(SkillEntry(
                    path=str(manifest), source=source, mutable=mutable,
                    skill_id=metadata.get("id"), version=metadata.get("version"),
                    kind=document.get("kind") if isinstance(document, dict) else None,
                ))
            except (OSError, ValueError, yaml.YAMLError) as exc:
                entries.append(SkillEntry(
                    path=str(manifest), source=source, mutable=mutable,
                    skill_id=None, version=None, kind=None, error=str(exc),
                ))
    return entries
