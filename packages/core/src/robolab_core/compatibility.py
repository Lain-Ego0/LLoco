"""SkillPackage × RobotProfile compatibility checks for B1.4."""

from __future__ import annotations

from typing import Any

from robolab_core.issues import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    Issue,
    Report,
)
from robolab_core.versioning import VersionRange


def _subject(skill: dict[str, Any], profile: dict[str, Any]) -> str:
    skill_meta = skill["metadata"]
    profile_meta = profile["metadata"]
    return (
        f"{skill['kind']} {skill_meta['id']}@{skill_meta['version']}"
        f" × RobotProfile {profile_meta['id']}@{profile_meta['version']}"
    )


def check_compatibility(
    skill: dict[str, Any],
    profile: dict[str, Any],
) -> Report:
    """Check a structurally valid Skill manifest against a RobotProfile."""
    report = Report(subject=_subject(skill, profile))
    kind = skill["kind"]

    if kind != "MotionSkill":
        report.issues.append(
            Issue(
                SEVERITY_INFO,
                "compat.not-robot-bound",
                f"{kind} 不绑定具体机器人；兼容性由平台版本与权限审查决定，与 Robot Profile 无关",
            )
        )
        return report

    compat = skill["spec"]["compatibility"]
    profile_meta = profile["metadata"]
    description = profile["description"]
    control = profile["control"]

    # Declared Profile identifier.
    robots = compat["robots"]
    entry = next((r for r in robots if r["profile"] == profile_meta["id"]), None)
    if entry is None:
        declared = ", ".join(r["profile"] for r in robots)
        report.issues.append(
            Issue(
                SEVERITY_ERROR,
                "compat.profile-not-declared",
                f"Skill 未声明适配 {profile_meta['id']!r}（已声明: {declared}）",
                "spec/compatibility/robots",
            )
        )
        return report

    # Profile version range.
    version_range = VersionRange.parse(entry["version"])
    if version_range.contains(profile_meta["version"]):
        report.issues.append(
            Issue(
                SEVERITY_INFO,
                "compat.profile-version",
                f"Profile 版本 {profile_meta['version']} 命中范围 {entry['version']!r}",
            )
        )
    else:
        report.issues.append(
            Issue(
                SEVERITY_ERROR,
                "compat.profile-version",
                f"Profile 版本 {profile_meta['version']} 不在 Skill 声明范围 {entry['version']!r} 内",
                "spec/compatibility/robots",
            )
        )

    # Canonical joint set.
    if compat["jointSet"] == description["jointSet"]:
        report.issues.append(
            Issue(
                SEVERITY_INFO,
                "compat.joint-set",
                f"jointSet 一致: {compat['jointSet']}",
            )
        )
    else:
        report.issues.append(
            Issue(
                SEVERITY_ERROR,
                "compat.joint-set",
                f"Skill 要求 jointSet {compat['jointSet']!r}，Profile 提供 {description['jointSet']!r}",
                "spec/compatibility/jointSet",
            )
        )

    # Control mode and frequency.
    if compat["controlMode"] == control["mode"]:
        report.issues.append(
            Issue(SEVERITY_INFO, "compat.control-mode", f"控制模式一致: {control['mode']}")
        )
    else:
        report.issues.append(
            Issue(
                SEVERITY_ERROR,
                "compat.control-mode",
                f"Skill 要求 controlMode {compat['controlMode']!r}，Profile 为 {control['mode']!r}",
                "spec/compatibility/controlMode",
            )
        )
    if compat["controlHz"] == control["frequencyHz"]:
        report.issues.append(
            Issue(SEVERITY_INFO, "compat.control-hz", f"控制频率一致: {control['frequencyHz']} Hz")
        )
    else:
        report.issues.append(
            Issue(
                SEVERITY_ERROR,
                "compat.control-hz",
                f"Skill 要求 {compat['controlHz']} Hz，Profile 为 {control['frequencyHz']} Hz",
                "spec/compatibility/controlHz",
            )
        )

    # Simulation target availability.
    capabilities = profile["capabilities"]
    simulation = profile["targets"]["simulation"]
    if simulation["enabled"] and capabilities["simulation"]:
        report.issues.append(
            Issue(SEVERITY_INFO, "compat.simulation", "仿真 target 可用")
        )
    else:
        report.issues.append(
            Issue(
                SEVERITY_ERROR,
                "compat.simulation",
                "Skill 需要仿真 target，但 Profile 未启用 simulation",
                "targets/simulation",
            )
        )

    # Physical deployment gate information.
    if "deploy" in skill["spec"]["actions"]:
        if capabilities["physicalDeployment"]:
            report.issues.append(
                Issue(
                    SEVERITY_INFO,
                    "compat.deploy",
                    "Skill 声明 deploy action；Profile 具备 physicalDeployment capability，仍需通过全部安全门禁",
                )
            )
        else:
            report.issues.append(
                Issue(
                    SEVERITY_INFO,
                    "compat.deploy",
                    "Skill 声明 deploy action，但 Profile 未具备 physicalDeployment capability；"
                    "实机入口保持禁用（docs/ROBOT_ADAPTATION.md §7）",
                )
            )

    # I/O schema comparison is deferred until B2 task bindings exist.
    report.issues.append(
        Issue(
            SEVERITY_INFO,
            "compat.io-schema",
            f"Skill 声明 observationSchema={compat['observationSchema']!r}、actionSchema={compat['actionSchema']!r}；"
            "Profile 侧 schema id 由任务绑定导出（B2），当前记录待比对",
        )
    )

    return report
