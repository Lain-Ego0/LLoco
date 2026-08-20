"""Baseline B1.6 lint checks for Skill packages."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from robolab_core.issues import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    Issue,
)

_KNOWN_SPDX = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Zlib",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "MPL-2.0",
    "CC-BY-4.0",
    "CC0-1.0",
    "Unlicense",
}

_FLOATING_REFS = {"main", "master", "HEAD", "latest", "develop", "trunk", "stable", "release"}
_LICENSE_FILE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")


def _check_license(document: dict[str, Any], package_dir: Path | None) -> list[Issue]:
    issues: list[Issue] = []
    license_id = document["metadata"]["license"]
    if license_id not in _KNOWN_SPDX:
        issues.append(
            Issue(
                SEVERITY_WARNING,
                "lint.license-unknown",
                f"许可证 {license_id!r} 不在常见 SPDX 集合内；请确认其允许公开分发（D-009）",
                "metadata/license",
            )
        )
    if package_dir is not None and not any(
        (package_dir / name).is_file() for name in _LICENSE_FILE_NAMES
    ):
        issues.append(
            Issue(
                SEVERITY_ERROR,
                "lint.license-file-missing",
                f"包目录缺少许可证文件（{'/'.join(_LICENSE_FILE_NAMES)} 之一）",
            )
        )
    return issues


def _check_revision(document: dict[str, Any]) -> list[Issue]:
    source = document["metadata"].get("source")
    if source is None:
        return [
            Issue(
                SEVERITY_INFO,
                "lint.source-missing",
                "未声明 metadata.source；发布到 catalog 前必须补齐仓库与固定 revision",
                "metadata",
            )
        ]
    revision = source["revision"]
    if revision in _FLOATING_REFS or revision.startswith("refs/heads/"):
        return [
            Issue(
                SEVERITY_ERROR,
                "lint.revision-floating",
                f"source.revision={revision!r} 是浮动引用；发布版本必须固定 commit SHA 或 tag（D-016）",
                "metadata/source/revision",
            )
        ]
    if all(c in "0123456789abcdef" for c in revision) and len(revision) == 40:
        return []
    if revision.startswith("refs/tags/"):
        return [
            Issue(
                SEVERITY_WARNING,
                "lint.revision-tag",
                f"source.revision={revision!r} 是 tag，tag 可被移动；建议改为完整 commit SHA",
                "metadata/source/revision",
            )
        ]
    return [
        Issue(
            SEVERITY_WARNING,
            "lint.revision-unrecognized",
            f"无法确认 source.revision={revision!r} 是否固定；推荐完整 commit SHA",
            "metadata/source/revision",
        )
    ]


def verify_artifacts(artifacts: list[dict[str, Any]], package_dir: Path) -> list[Issue]:
    """Verify artifact existence, size, and SHA-256 against the manifest."""
    issues: list[Issue] = []
    for artifact in artifacts:
        path = package_dir / artifact["path"]
        label = f"artifact {artifact['name']!r} ({artifact['path']})"
        if not path.is_file():
            issues.append(
                Issue(SEVERITY_ERROR, "lint.artifact-missing", f"{label} 文件不存在")
            )
            continue
        data = path.read_bytes()
        declared_size = artifact.get("size")
        if declared_size is not None and declared_size != len(data):
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "lint.artifact-size",
                    f"{label} size 声明 {declared_size}，实际 {len(data)}",
                    artifact["path"],
                )
            )
        actual = hashlib.sha256(data).hexdigest()
        if actual != artifact["sha256"]:
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "lint.artifact-hash",
                    f"{label} SHA-256 不匹配: 声明 {artifact['sha256'][:16]}…，实际 {actual[:16]}…",
                    artifact["path"],
                )
            )
        else:
            issues.append(
                Issue(
                    SEVERITY_INFO,
                    "lint.artifact-ok",
                    f"{label} SHA-256 校验通过",
                    artifact["path"],
                )
            )
    return issues


def _check_referenced_files(document: dict[str, Any], package_dir: Path) -> list[Issue]:
    issues: list[Issue] = []
    spec = document["spec"]

    def require(rel_path: str, rule: str, label: str) -> None:
        if not (package_dir / rel_path).is_file():
            issues.append(
                Issue(SEVERITY_ERROR, rule, f"{label}引用的文件不存在: {rel_path}", rel_path)
            )

    for action_name, action in spec["actions"].items():
        for key in ("inputSchema", "outputSchema"):
            if key in action:
                require(action[key], "lint.action-schema-missing", f"action {action_name!r} {key} ")

    validation = spec.get("validation")
    if validation and "smoke" in validation:
        require(validation["smoke"], "lint.smoke-missing", "validation.smoke ")
    elif validation is None:
        issues.append(
            Issue(
                SEVERITY_WARNING,
                "lint.smoke-missing",
                "未声明 validation.smoke；发布检查清单要求至少一个 smoke test（SKILL_SPEC §12）",
                "spec/validation",
            )
        )

    environment = spec["runtime"].get("environment")
    if environment and environment.get("mode") == "conda":
        require(
            environment.get("file", "environment.yml"),
            "lint.environment-missing",
            "runtime.environment ",
        )

    if document["kind"] == "AgentSkill":
        require(
            spec["runtime"]["instructions"],
            "lint.instructions-missing",
            "AgentSkill instructions ",
        )

    if not (package_dir / "README.md").is_file():
        issues.append(
            Issue(SEVERITY_ERROR, "lint.readme-missing", "包目录缺少 README.md（SKILL_SPEC §4 必需）")
        )
    return issues


def lint_skill_package(
    document: dict[str, Any],
    *,
    package_dir: Path | None = None,
) -> list[Issue]:
    """Lint a Skill package; filesystem checks require ``package_dir``."""
    issues: list[Issue] = []
    issues.extend(_check_license(document, package_dir))
    issues.extend(_check_revision(document))
    if package_dir is not None:
        artifacts = document["spec"].get("artifacts")
        if artifacts:
            issues.extend(verify_artifacts(artifacts, package_dir))
        issues.extend(_check_referenced_files(document, package_dir))
    return issues
