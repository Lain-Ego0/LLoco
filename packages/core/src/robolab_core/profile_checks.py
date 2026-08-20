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
