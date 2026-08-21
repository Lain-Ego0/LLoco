"""RoboLab v1alpha1 JSON Schema loading and structural validation."""

from __future__ import annotations

import json
from functools import cache, lru_cache
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

API_VERSION = "robolab.dev/v1alpha1"
PROFILE_VNEXT_API_VERSION = "robolab.dev/v1beta1"

_SCHEMA_FILES = {
    "RobotProfile": "robot_profile.v1alpha1.schema.json",
    "JointSet": "joint_set.v1alpha1.schema.json",
    "ActuatorSensorMapping": "actuator_sensor_mapping.v1alpha1.schema.json",
    "TaskBinding": "task_binding.v1alpha1.schema.json",
    "MotionSkill": "skill_package.v1alpha1.schema.json",
    "PlatformSkill": "skill_package.v1alpha1.schema.json",
    "AgentSkill": "skill_package.v1alpha1.schema.json",
}

_VERSIONED_SCHEMA_FILES = {
    ("RobotProfile", PROFILE_VNEXT_API_VERSION): "robot_profile.v1beta1.schema.json",
}

KNOWN_KINDS = tuple(_SCHEMA_FILES)

_MOTION_SCHEMA_FILE = "motion_command.v1.schema.json"


class UnknownKindError(ValueError):
    """Raised when a document has no recognized kind or apiVersion."""


def detect_kind(document: dict[str, Any]) -> str:
    """Identify a manifest kind or raise ``UnknownKindError``."""
    if not isinstance(document, dict):
        raise UnknownKindError("文档不是 YAML/JSON 对象")
    kind = document.get("kind")
    if kind not in _SCHEMA_FILES:
        raise UnknownKindError(
            f"无法识别的 kind={kind!r}；支持 {', '.join(KNOWN_KINDS)}"
        )
    api_version = document.get("apiVersion")
    if api_version not in {API_VERSION, PROFILE_VNEXT_API_VERSION}:
        raise UnknownKindError(
            f"无法识别的 apiVersion={api_version!r}；当前仅支持 {API_VERSION} 和 {PROFILE_VNEXT_API_VERSION}"
        )
    if api_version == PROFILE_VNEXT_API_VERSION and kind != "RobotProfile":
        raise UnknownKindError(f"{kind} 不支持 apiVersion={api_version}")
    return kind


@cache
def load_schema(kind: str) -> dict[str, Any]:
    """Load the JSON Schema for a kind, with caching."""
    if kind not in _SCHEMA_FILES:
        raise UnknownKindError(f"没有 {kind!r} 对应的 schema")
    resource = resources.files(__package__).joinpath(f"data/{_SCHEMA_FILES[kind]}")
    return json.loads(resource.read_text(encoding="utf-8"))


@cache
def load_schema_for_document(kind: str, api_version: str) -> dict[str, Any]:
    filename = _VERSIONED_SCHEMA_FILES.get((kind, api_version))
    if filename is None:
        return load_schema(kind)
    resource = resources.files(__package__).joinpath(f"data/{filename}")
    return json.loads(resource.read_text(encoding="utf-8"))


@cache
def get_validator(kind: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema(kind))


def validate_schema(
    document: dict[str, Any], kind: str | None = None
) -> list[ValidationError]:
    """Return structural validation errors sorted by document path."""
    resolved = kind or detect_kind(document)
    schema = load_schema_for_document(resolved, str(document.get("apiVersion", API_VERSION)))
    errors = list(Draft202012Validator(schema).iter_errors(document))
    return sorted(errors, key=lambda e: list(e.absolute_path))


@lru_cache(maxsize=1)
def load_motion_schema() -> dict[str, Any]:
    """Load the stable RoboLab motion JobCommand schema."""
    resource = resources.files(__package__).joinpath(f"data/{_MOTION_SCHEMA_FILE}")
    return json.loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_motion_validator() -> Draft202012Validator:
    return Draft202012Validator(load_motion_schema())


def validate_motion_command(document: dict[str, Any]) -> list[ValidationError]:
    """Return structural errors for a ``robolab-motion-v1`` JobCommand."""
    return sorted(
        get_motion_validator().iter_errors(document),
        key=lambda e: list(e.absolute_path),
    )
