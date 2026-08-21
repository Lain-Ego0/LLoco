"""Manifest loading and validation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jsonschema.exceptions import ValidationError
from robolab_schemas import KNOWN_KINDS, UnknownKindError, detect_kind, validate_schema

from robolab_core.issues import (
    SEVERITY_ERROR,
    Issue,
    Report,
)

SUPPORTED_KINDS = KNOWN_KINDS

_KIND_LABEL = {
    "RobotProfile": "RobotProfile",
    "JointSet": "JointSet",
    "ActuatorSensorMapping": "ActuatorSensorMapping",
    "TaskBinding": "TaskBinding",
    "MotionSkill": "MotionSkill",
    "PlatformSkill": "PlatformSkill",
    "AgentSkill": "AgentSkill",
}


def load_document(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON manifest, raising file and parse errors."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise UnknownKindError(f"{path}: 文档不是 YAML 对象")
    return data


def _schema_error_to_issue(error: ValidationError) -> Issue:
    path = "/".join(str(p) for p in error.absolute_path)
    return Issue(
        severity=SEVERITY_ERROR,
        rule="schema",
        message=error.message,
        path=path,
    )


def subject_of(document: dict[str, Any]) -> str:
    """Return a readable subject name, such as a MotionSkill identifier."""
    kind = document.get("kind", "?")
    metadata = document.get("metadata") or {}
    skill_id = metadata.get("id", "<未命名>")
    version = metadata.get("version", "?")
    return f"{_KIND_LABEL.get(kind, kind)} {skill_id}@{version}"


def validate_document(
    document: dict[str, Any],
    *,
    package_dir: str | Path | None = None,
    source: str = "<memory>",
) -> Report:
    """Run structural and semantic validation for a manifest."""
    from robolab_core.joint_mapping import check_joint_set
    from robolab_core.lint import lint_skill_package
    from robolab_core.profile_checks import check_robot_profile

    try:
        kind = detect_kind(document)
    except UnknownKindError as exc:
        return Report(
            subject=str(source),
            issues=[Issue(SEVERITY_ERROR, "kind", str(exc))],
        )

    report = Report(subject=subject_of(document))
    schema_errors = validate_schema(document, kind)
    report.extend(_schema_error_to_issue(e) for e in schema_errors)
    if schema_errors:
        report.issues.append(
            Issue(
                SEVERITY_ERROR,
                "schema.aborted",
                "结构校验未通过，跳过语义检查；请先修复以上 schema 错误",
            )
        )
        return report

    if kind == "RobotProfile":
        report.extend(check_robot_profile(document))
    elif kind == "JointSet":
        report.extend(check_joint_set(document))
    elif kind == "ActuatorSensorMapping":
        report.extend(check_actuator_sensor_mapping(document))
    elif kind in ("MotionSkill", "PlatformSkill", "AgentSkill"):
        report.extend(
            lint_skill_package(
                document,
                package_dir=Path(package_dir) if package_dir else None,
            )
        )
    return report


def check_actuator_sensor_mapping(document: dict[str, Any]) -> list[Issue]:
    """Cross-field checks for the R2 actuator/sensor mapping manifest."""
    issues: list[Issue] = []
    actuators = document["actuators"]
    for field in ("canonicalJoint",):
        values = [item[field] for item in actuators]
        for value in sorted({item for item in values if values.count(item) > 1}):
            issues.append(Issue(SEVERITY_ERROR, "mapping.duplicate-name", f"actuator {field} 重复: {value!r}", f"actuators/{field}"))
    for item in actuators:
        low, high = item["limits"]["position"]
        if low >= high:
            issues.append(Issue(SEVERITY_ERROR, "mapping.limit-direction", f"{item['canonicalJoint']!r} position limit lower 必须小于 upper", f"actuators/{item['canonicalJoint']}/limits/position"))
    indices = sorted(item["policy"]["actionIndex"] for item in actuators)
    if indices != list(range(len(indices))):
        issues.append(Issue(SEVERITY_ERROR, "mapping.action-index-not-contiguous", f"actionIndex 必须连续从 0 开始，实际 {indices}", "actuators/policy/actionIndex"))
    sensor_indices = sorted(item["policy"]["observationIndex"] for item in document["sensors"])
    if sensor_indices and sensor_indices != list(range(len(sensor_indices))):
        issues.append(Issue(SEVERITY_ERROR, "mapping.observation-index-not-contiguous", f"observationIndex 必须连续从 0 开始，实际 {sensor_indices}", "sensors/policy/observationIndex"))
    return issues
