"""RoboLab v1alpha1 JSON Schema loading and structural validation."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

API_VERSION = "robolab.dev/v1alpha1"

_SCHEMA_FILES = {
    "RobotProfile": "robot_profile.v1alpha1.schema.json",
    "JointSet": "joint_set.v1alpha1.schema.json",
    "MotionSkill": "skill_package.v1alpha1.schema.json",
    "PlatformSkill": "skill_package.v1alpha1.schema.json",
    "AgentSkill": "skill_package.v1alpha1.schema.json",
}

KNOWN_KINDS = tuple(_SCHEMA_FILES)


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
    if api_version != API_VERSION:
        raise UnknownKindError(
            f"无法识别的 apiVersion={api_version!r}；当前仅支持 {API_VERSION}"
        )
    return kind


@lru_cache(maxsize=None)
def load_schema(kind: str) -> dict[str, Any]:
    """Load the JSON Schema for a kind, with caching."""
    if kind not in _SCHEMA_FILES:
        raise UnknownKindError(f"没有 {kind!r} 对应的 schema")
    resource = resources.files(__package__).joinpath(f"data/{_SCHEMA_FILES[kind]}")
    return json.loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def get_validator(kind: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema(kind))


def validate_schema(document: dict[str, Any], kind: str | None = None) -> list[ValidationError]:
    """Return structural validation errors sorted by document path."""
    resolved = kind or detect_kind(document)
    errors = list(get_validator(resolved).iter_errors(document))
    return sorted(errors, key=lambda e: list(e.absolute_path))
