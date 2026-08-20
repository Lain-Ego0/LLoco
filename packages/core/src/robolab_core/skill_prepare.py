"""Explicit review and Conda preparation planning for Skills."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robolab_core.documents import load_document, validate_document


def review_skill(path: str | Path) -> dict[str, Any]:
    """Return permissions and a non-executing environment preparation plan."""
    manifest = Path(path)
    document = load_document(manifest)
    report = validate_document(document, package_dir=manifest.parent, source=str(manifest))
    if not report.ok:
        raise ValueError(report.render())
    runtime = document["spec"]["runtime"]
    environment = runtime.get("environment", {"mode": "inherit"})
    mode = environment["mode"]
    plan: list[str] = []
    if mode == "conda":
        plan.append(f"Review {environment.get('file', 'environment.yml')} before creating an environment")
        plan.append("Run the approved Conda command manually; no command was executed")
    else:
        plan.append("No separate Conda environment is requested")
    return {
        "skill_id": document["metadata"]["id"],
        "permissions": document["spec"]["permissions"],
        "environment": environment,
        "prepare_plan": plan,
        "executed": False,
    }
