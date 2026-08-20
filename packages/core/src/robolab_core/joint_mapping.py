"""Machine validation for the B1.3 joint-mapping contract."""

from __future__ import annotations

from typing import Any

from robolab_core.issues import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    Issue,
)


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    dup: list[str] = []
    for value in values:
        if value in seen and value not in dup:
            dup.append(value)
        seen.add(value)
    return dup


def _check_unique_names(joints: list[dict[str, Any]]) -> list[Issue]:
    issues: list[Issue] = []
    for field, extract, label in (
        ("name", lambda j: j["name"], "canonical 关节名"),
        ("mjcf.joint", lambda j: j["mjcf"]["joint"], "MJCF joint 名"),
    ):
        for dup in _duplicates([extract(j) for j in joints]):
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "joint.duplicate-name",
                    f"{label}重复: {dup!r}；名称必须唯一，禁止仅靠数组位置推断",
                )
            )
    actuators = [j["mjcf"]["actuator"] for j in joints if j["mjcf"]["actuator"] is not None]
    for dup in _duplicates(actuators):
        issues.append(
            Issue(
                SEVERITY_ERROR,
                "joint.duplicate-actuator",
                f"MJCF actuator 名重复: {dup!r}；actuator 与关节必须一一对应",
            )
        )
    return issues


def _check_bijection_and_size(joints: list[dict[str, Any]], dof: int) -> list[Issue]:
    issues: list[Issue] = []
    if len(joints) != dof:
        issues.append(
            Issue(
                SEVERITY_ERROR,
                "joint.count-mismatch",
                f"metadata.dof={dof}，但 joints 数组长度为 {len(joints)}；数组长度必须与自由度一致",
                "joints",
            )
    )

    canonical = {j["name"] for j in joints}
    mjcf_joints = {j["mjcf"]["joint"] for j in joints}
    if len(canonical) == len(joints) and len(mjcf_joints) != len(joints):
        issues.append(
            Issue(
                SEVERITY_ERROR,
                "joint.not-bijective",
                "多个 canonical 关节映射到同一个 MJCF joint；canonical↔MJCF 必须是双射",
            )
        )

    missing_actuator = [j["name"] for j in joints if j["mjcf"]["actuator"] is None]
    if missing_actuator:
        issues.append(
            Issue(
                SEVERITY_WARNING,
                "joint.unactuated",
                "以下关节未声明 actuator（欠驱动）: " + ", ".join(missing_actuator),
            )
        )
    for joint in joints:
        actuator = joint["mjcf"]["actuator"]
        index = joint["mjcf"]["actuatorIndex"]
        if (actuator is None) != (index is None):
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "joint.actuator-index-inconsistent",
                    f"关节 {joint['name']!r} 的 actuator 与 actuatorIndex 必须同时为 null 或同时有值",
                    f"joints/{joint['name']}/mjcf",
                )
            )
    return issues


def _check_index_continuity(joints: list[dict[str, Any]], base: dict[str, Any] | None) -> list[Issue]:
    issues: list[Issue] = []
    fields = (
        ("mjcf.qposIndex", lambda j: j["mjcf"]["qposIndex"], "qpos"),
        ("mjcf.qvelIndex", lambda j: j["mjcf"]["qvelIndex"], "qvel"),
        ("mjcf.actuatorIndex", lambda j: j["mjcf"]["actuatorIndex"], "actuator"),
        ("policy.observationIndex", lambda j: j["policy"]["observationIndex"], "policy 观测"),
        ("policy.actionIndex", lambda j: j["policy"]["actionIndex"], "policy 动作"),
    )
    for field, extract, label in fields:
        values = [extract(j) for j in joints if extract(j) is not None]
        if not values:
            continue
        if len(set(values)) != len(values):
            dupes = sorted({v for v in values if values.count(v) > 1})
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "joint.index-duplicate",
                    f"{label}索引重复: {dupes}；同一索引不能服务多个关节",
                    field,
                )
            )
            continue
        span = max(values) - min(values) + 1
        if span != len(values):
            missing = sorted(set(range(min(values), max(values) + 1)) - set(values))
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "joint.index-not-contiguous",
                    f"{label}索引不连续: 覆盖 {min(values)}..{max(values)}，缺少 {missing}",
                    field,
                )
            )

    if base and base.get("type") == "free":
        qpos_dim = base.get("qposDim", 7)
        qvel_dim = base.get("qvelDim", 6)
        low_qpos = min(j["mjcf"]["qposIndex"] for j in joints)
        low_qvel = min(j["mjcf"]["qvelIndex"] for j in joints)
        if low_qpos < qpos_dim:
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "joint.index-overlaps-base",
                    f"自由根关节占用 qpos 0..{qpos_dim - 1}，但关节 qposIndex 从 {low_qpos} 开始",
                    "base",
                )
            )
        if low_qvel < qvel_dim:
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "joint.index-overlaps-base",
                    f"自由根关节占用 qvel 0..{qvel_dim - 1}，但关节 qvelIndex 从 {low_qvel} 开始",
                    "base",
                )
            )
    return issues


def _check_limits_and_pose(joints: list[dict[str, Any]]) -> list[Issue]:
    issues: list[Issue] = []
    for joint in joints:
        name = joint["name"]
        limits = joint["limits"]
        lower = limits["position"]["lower"]
        upper = limits["position"]["upper"]
        if not lower < upper:
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "joint.limit-direction",
                    f"关节 {name!r} 限位方向错误: lower({lower}) 必须小于 upper({upper})",
                    f"joints/{name}/limits/position",
                )
            )
        default = joint["defaultPosition"]
        if not (lower <= default <= upper):
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "joint.default-out-of-range",
                    f"关节 {name!r} 默认姿态 {default} rad 超出限位 [{lower}, {upper}]",
                    f"joints/{name}/defaultPosition",
                )
            )
        recommended = joint["gains"]["recommended"]
        maximum = joint["gains"]["maximum"]
        if recommended["kp"] > maximum["kp"] or recommended["kd"] > maximum["kd"]:
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "joint.gains-order",
                    f"关节 {name!r} 推荐增益 (kp={recommended['kp']}, kd={recommended['kd']}) "
                    f"超过绝对最大 (kp={maximum['kp']}, kd={maximum['kd']})",
                    f"joints/{name}/gains",
                )
            )
        if joint["gearRatio"] <= 0:
            issues.append(
                Issue(
                    SEVERITY_ERROR,
                    "joint.gear-ratio",
                    f"关节 {name!r} 减速比必须为正，当前 {joint['gearRatio']}",
                    f"joints/{name}/gearRatio",
                )
            )
    return issues


def check_joint_set(document: dict[str, Any]) -> list[Issue]:
    """Run every machine validation for a JointSet v1alpha1 document."""
    joints = document["joints"]
    dof = document["metadata"]["dof"]
    base = document.get("base")

    issues: list[Issue] = []
    issues.extend(_check_unique_names(joints))
    issues.extend(_check_bijection_and_size(joints, dof))
    issues.extend(_check_index_continuity(joints, base))
    issues.extend(_check_limits_and_pose(joints))
    return issues
