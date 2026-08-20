"""Export a validated AgentSkill into an external Agent discovery directory."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from robolab_core.documents import load_document, validate_document


def _content_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def export_agent_skill(source: str | Path, target_root: str | Path = ".agents/skills") -> dict[str, Any]:
    """Copy the portable AgentSkill surface to Codex's `.agents/skills` tree."""
    package = Path(source).resolve()
    manifest = package / "skill.yaml"
    document = load_document(manifest)
    report = validate_document(document, package_dir=package, source=str(manifest))
    if not report.ok:
        raise ValueError(report.render())
    if document["kind"] != "AgentSkill":
        raise ValueError("仅 AgentSkill 可以导出给外部 Agent")
    name = document["metadata"]["id"].split(".")[-1].replace("_", "-")
    destination = Path(target_root).resolve() / name
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for relative in ("SKILL.md", "README.md", "LICENSE", "references", "scripts", "assets"):
        item = package / relative
        if item.is_file():
            shutil.copy2(item, destination / relative)
        elif item.is_dir():
            shutil.copytree(item, destination / relative)
    metadata = {"name": name, "description": document["metadata"]["description"], "robolab": {"skillId": document["metadata"]["id"], "version": document["metadata"]["version"], "contentSha256": _content_hash(package), "allowedActions": document["spec"]["tools"]["allow"], "deniedActions": document["spec"]["tools"].get("deny", [])}}
    agent_dir = destination / "agents"
    agent_dir.mkdir()
    (agent_dir / "openai.yaml").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"source": str(package), "destination": str(destination), "skill_id": document["metadata"]["id"], "content_sha256": metadata["robolab"]["contentSha256"]}
