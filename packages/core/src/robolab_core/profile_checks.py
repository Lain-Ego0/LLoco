"""Cross-field RobotProfile checks that JSON Schema cannot express."""

from __future__ import annotations

from typing import Any

from robolab_core.issues import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    Issue,
)

_PHYSICAL_PREREQUISITES = ("motorCommunication", "sensorStreaming", "calibration")


def check_robot_profile(profile: dict[str, Any]) -> list[Issue]:
    if profile.get("apiVersion") == "robolab.dev/v1beta1":
        return check_robot_profile_vnext(profile)
    issues: list[Issue] = []
    capabilities = profile["capabilities"]
    targets = profile["targets"]
    physical = targets["physical"]
    simulation = targets["simulation"]

    if capabilities["simulation"] and not simulation["enabled"]:
        issues.append(
            Issue(
                SEVERITY_ERROR,
                "profile.simulation-inconsistent",
                "capabilities.simulation 为 true，但 targets.simulation.enabled 为 false",
                "targets/simulation",
            )
        )

    if physical["enabled"] and not physical.get("driver"):
        issues.append(
            Issue(
                SEVERITY_ERROR,
                "profile.physical-driver-missing",
                "targets.physical.enabled 为 true，但未指定 driver",
                "targets/physical/driver",
            )
        )
    if not physical["enabled"] and physical.get("driver"):
        issues.append(
            Issue(
                SEVERITY_WARNING,
                "profile.physical-driver-unused",
                f"physical target 未启用，但声明了 driver={physical['driver']!r}；启用前不会生效",
                "targets/physical/driver",
            )
        )

    if capabilities["physicalDeployment"]:
        missing = [c for c in _PHYSICAL_PREREQUISITES if not capabilities[c]]
        if missing:
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "profile.physical-prerequisites",
                    "physicalDeployment 要求 "
                    + "/".join(_PHYSICAL_PREREQUISITES)
                    + f" 均为 true，缺失: {', '.join(missing)}",
                    "capabilities",
                )
            )
        if not physical["enabled"]:
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "profile.physical-target-disabled",
                    "capabilities.physicalDeployment 为 true，但 targets.physical.enabled 为 false",
                    "targets/physical",
                )
            )
    elif physical["enabled"]:
        issues.append(
            Issue(
                SEVERITY_WARNING,
                "profile.physical-without-capability",
                "physical target 已启用但 physicalDeployment capability 为 false，WebUI 不会激活实机入口",
                "capabilities/physicalDeployment",
            )
        )

    if not capabilities["simulation"] and not capabilities["physicalDeployment"]:
        issues.append(
            Issue(
                SEVERITY_INFO,
                "profile.no-target",
                "该 Profile 当前没有任何可用 target（simulation 与 physicalDeployment 均为 false）",
                "capabilities",
            )
        )
    return issues


def check_robot_profile_vnext(profile: dict[str, Any]) -> list[Issue]:
    """Semantic checks for the explicit R2 RobotProfile v1beta1 contract."""
    issues: list[Issue] = []
    capabilities = profile["capabilities"]
    targets = profile["targets"]
    if not targets["simulation"]["enabled"]:
        issues.append(Issue(SEVERITY_ERROR, "profile.simulation-disabled", "v1beta1 simulation-first Profile 必须启用 simulation target", "targets/simulation/enabled"))
    if targets["physical"]["enabled"] or targets["physical"]["driver"] is not None:
        issues.append(Issue(SEVERITY_ERROR, "profile.physical-not-simulation-only", "v1beta1 首版禁止 physical target 和 driver", "targets/physical"))
    required_false = ("motorCommunication", "sensorStreaming", "calibration", "physicalDeployment")
    for capability in required_false:
        if capabilities[capability] is not False:
            issues.append(Issue(SEVERITY_ERROR, "profile.simulation-only-capability", f"simulation-only Profile 的 capabilities/{capability} 必须为 false", f"capabilities/{capability}"))
    if not capabilities["simulation"] or not capabilities["training"]:
        issues.append(Issue(SEVERITY_ERROR, "profile.simulation-training-required", "simulation-only Profile 必须同时声明 simulation=true 和 training=true", "capabilities"))
    if not profile["bindings"]["tasks"]:
        issues.append(Issue(SEVERITY_ERROR, "profile.task-binding-missing", "至少需要一个通用 task binding", "bindings/tasks"))
    if profile["metadata"]["id"] != "community.firedog2_2":
        issues.append(Issue(SEVERITY_WARNING, "profile.reference-id", "FireDog 参考 Profile 的推荐稳定 ID 是 community.firedog2_2", "metadata/id"))
    return issues
