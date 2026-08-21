"""Reusable MJCF/URDF inspection and deterministic URDF-to-MJCF conversion."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INSPECTOR_VERSION = "robolab-robot-inspector/1"


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    rule: str
    message: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "rule": self.rule,
            "message": self.message,
            "path": self.path,
        }


def _attr(element: ET.Element, name: str, default: str = "") -> str:
    return element.attrib.get(name, default)


def _floats(value: str, count: int) -> list[float]:
    parts = value.split()
    if len(parts) != count:
        raise ValueError(f"expected {count} values, got {len(parts)}")
    return [float(p) for p in parts]


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_names(elements: Iterable[ET.Element], kind: str, path: str) -> list[Diagnostic]:
    seen: dict[str, int] = {}
    result: list[Diagnostic] = []
    for element in elements:
        name = _attr(element, "name")
        if not name:
            result.append(
                Diagnostic("error", "INSPECT_NAME_MISSING", f"{kind} 缺少 name", path)
            )
            continue
        seen[name] = seen.get(name, 0) + 1
    for name, count in sorted(seen.items()):
        if count > 1:
            result.append(
                Diagnostic(
                    "error",
                    "INSPECT_DUPLICATE_NAME",
                    f"{kind} 名称重复: {name!r} ({count} 次)",
                    f"{path}/{kind}[name={name}]",
                )
            )
    return result


def _resolve_mesh(model_path: Path, reference: str, meshdir: str = "") -> tuple[str, Path]:
    candidate = Path(reference)
    if not candidate.is_absolute():
        candidate = model_path.parent / meshdir / candidate
    return reference, candidate.resolve()


def _inspect_mjcf(path: Path, root: ET.Element) -> dict[str, Any]:
    diagnostics: list[Diagnostic] = []
    worldbody = root.find("worldbody")
    if worldbody is None:
        diagnostics.append(
            Diagnostic("error", "INSPECT_ROOT_BODY_MISSING", "MJCF 缺少 worldbody", "worldbody")
        )
        root_bodies: list[ET.Element] = []
    else:
        root_bodies = list(worldbody.findall("body"))
        if len(root_bodies) != 1:
            diagnostics.append(
                Diagnostic(
                    "error",
                    "INSPECT_ROOT_BODY_INVALID",
                    f"MJCF 必须有且只有一个 worldbody 根 body，实际 {len(root_bodies)} 个",
                    "worldbody/body",
                )
            )

    compiler = root.find("compiler")
    meshdir = _attr(compiler, "meshdir") if compiler is not None else ""
    meshes = list(root.findall("./asset/mesh"))
    bodies = list(root.iter("body"))
    joints = list(root.iter("joint"))
    actuators = [item for actuator_group in [root.find("actuator")] if actuator_group is not None for item in list(actuator_group)]
    sensors = [item for sensor_group in [root.find("sensor")] if sensor_group is not None for item in list(sensor_group)]
    sites = list(root.iter("site"))
    geoms = list(root.iter("geom"))
    keyframes = list(root.findall("./keyframe/key"))

    diagnostics.extend(_unique_names(bodies, "body", "worldbody"))
    diagnostics.extend(_unique_names(joints, "joint", "worldbody"))
    diagnostics.extend(_unique_names(actuators, "actuator", "actuator"))
    diagnostics.extend(_unique_names(sensors, "sensor", "sensor"))
    diagnostics.extend(_unique_names(sites, "site", "worldbody"))
    diagnostics.extend(_unique_names(meshes, "mesh", "asset"))
    diagnostics.extend(_unique_names(keyframes, "keyframe", "keyframe"))

    mesh_records = []
    for mesh in meshes:
        reference = _attr(mesh, "file")
        _, resolved = _resolve_mesh(path, reference, meshdir)
        exists = resolved.is_file()
        mesh_records.append({"name": _attr(mesh, "name"), "file": reference, "resolved": str(resolved), "exists": exists})
        if not exists:
            diagnostics.append(
                Diagnostic("error", "INSPECT_MESH_MISSING", f"mesh 文件不存在: {reference!r}", f"asset/mesh[name={_attr(mesh, 'name')}]@file")
            )

    joint_names = {_attr(item, "name") for item in joints}
    for actuator in actuators:
        target = _attr(actuator, "joint") or _attr(actuator, "tendon") or _attr(actuator, "site")
        if _attr(actuator, "joint") and target not in joint_names:
            diagnostics.append(
                Diagnostic("error", "INSPECT_REFERENCE_MISSING", f"actuator { _attr(actuator, 'name')!r} 引用了不存在的 joint {target!r}", f"actuator/{_attr(actuator, 'name')}@joint")
            )
    for joint in joints:
        if _attr(joint, "limited") != "false" and _attr(joint, "range"):
            try:
                lower, upper = _floats(_attr(joint, "range"), 2)
                if not lower < upper:
                    diagnostics.append(Diagnostic("error", "INSPECT_LIMIT_INVALID", f"joint { _attr(joint, 'name')!r} range lower 必须小于 upper", f"worldbody/joint[name={_attr(joint, 'name')}]/@range"))
            except ValueError:
                diagnostics.append(Diagnostic("error", "INSPECT_LIMIT_INVALID", f"joint { _attr(joint, 'name')!r} range 不是两个数", f"worldbody/joint[name={_attr(joint, 'name')}]/@range"))

    site_names = {_attr(item, "name") for item in sites}
    for sensor in sensors:
        site_ref = _attr(sensor, "site")
        if site_ref and site_ref not in site_names:
            diagnostics.append(Diagnostic("error", "INSPECT_REFERENCE_MISSING", f"sensor { _attr(sensor, 'name')!r} 引用了不存在的 site {site_ref!r}", f"sensor/{_attr(sensor, 'name')}@site"))

    return {
        "format": "mjcf",
        "path": str(path),
        "counts": {"body": len(bodies), "joint": len(joints), "actuator": len(actuators), "sensor": len(sensors), "site": len(sites), "mesh": len(meshes), "keyframe": len(keyframes), "geom": len(geoms)},
        "rootBodies": [_attr(item, "name") for item in root_bodies],
        "bodies": [_attr(item, "name") for item in bodies],
        "joints": [{"name": _attr(item, "name"), "type": _attr(item, "type"), "range": _attr(item, "range")} for item in joints],
        "actuators": [{"name": _attr(item, "name"), "type": item.tag, "joint": _attr(item, "joint")} for item in actuators],
        "sensors": [{"name": _attr(item, "name"), "type": item.tag, "site": _attr(item, "site"), "joint": _attr(item, "joint")} for item in sensors],
        "sites": [_attr(item, "name") for item in sites],
        "meshes": mesh_records,
        "keyframes": [_attr(item, "name") for item in keyframes],
        "geoms": [{"name": _attr(item, "name"), "type": _attr(item, "type"), "mesh": _attr(item, "mesh")} for item in geoms],
        "diagnostics": [item.to_dict() for item in sorted(diagnostics, key=lambda i: (i.path, i.rule, i.message))],
    }


def _inspect_urdf(path: Path, root: ET.Element) -> dict[str, Any]:
    diagnostics: list[Diagnostic] = []
    links = list(root.findall("link"))
    joints = list(root.findall("joint"))
    link_names = {_attr(item, "name") for item in links}
    child_names = {_attr(item.find("child"), "link") for item in joints if item.find("child") is not None}
    roots = sorted(link_names - child_names)
    if len(roots) != 1:
        diagnostics.append(Diagnostic("error", "INSPECT_ROOT_BODY_INVALID", f"URDF 必须解析出一个 root link，实际 {len(roots)} 个", "robot/link"))
    diagnostics.extend(_unique_names(links, "link", "robot"))
    diagnostics.extend(_unique_names(joints, "joint", "robot"))
    mesh_records = []
    for link in links:
        for mesh in link.findall(".//mesh"):
            reference = _attr(mesh, "filename")
            clean = reference.split("/meshes/", 1)[-1] if "/meshes/" in reference else Path(reference).name
            resolved = (path.parent.parent / "meshes" / clean).resolve()
            exists = resolved.is_file()
            mesh_records.append({"link": _attr(link, "name"), "file": reference, "resolved": str(resolved), "exists": exists})
            if not exists:
                diagnostics.append(Diagnostic("error", "INSPECT_MESH_MISSING", f"link { _attr(link, 'name')!r} 引用的 mesh 不存在: {clean!r}", f"link/{_attr(link, 'name')}/mesh@filename"))
    joint_records = []
    for joint in joints:
        limit = joint.find("limit")
        lower = _attr(limit, "lower") if limit is not None else ""
        upper = _attr(limit, "upper") if limit is not None else ""
        if _attr(joint, "type") not in {"continuous", "fixed", "floating", "planar"} and (not lower or not upper):
            diagnostics.append(Diagnostic("error", "INSPECT_LIMIT_MISSING", f"joint { _attr(joint, 'name')!r} 缺少 position limit", f"joint/{_attr(joint, 'name')}/limit"))
        if lower and upper:
            try:
                if not float(lower) < float(upper):
                    diagnostics.append(Diagnostic("error", "INSPECT_LIMIT_INVALID", f"joint { _attr(joint, 'name')!r} lower 必须小于 upper", f"joint/{_attr(joint, 'name')}/limit"))
            except ValueError:
                diagnostics.append(Diagnostic("error", "INSPECT_LIMIT_INVALID", f"joint { _attr(joint, 'name')!r} limit 不是数字", f"joint/{_attr(joint, 'name')}/limit"))
        joint_records.append({"name": _attr(joint, "name"), "type": _attr(joint, "type"), "parent": _attr(joint.find("parent"), "link"), "child": _attr(joint.find("child"), "link"), "lower": lower, "upper": upper, "effort": _attr(limit, "effort") if limit is not None else "", "velocity": _attr(limit, "velocity") if limit is not None else ""})
    return {
        "format": "urdf",
        "path": str(path),
        "counts": {"body": len(links), "joint": len(joints), "actuator": 0, "sensor": 0, "site": 0, "mesh": len(mesh_records), "keyframe": 0, "geom": len(mesh_records)},
        "rootBodies": roots,
        "bodies": sorted(link_names),
        "joints": joint_records,
        "actuators": [],
        "sensors": [],
        "sites": [],
        "meshes": mesh_records,
        "keyframes": [],
        "geoms": [],
        "diagnostics": [item.to_dict() for item in sorted(diagnostics, key=lambda i: (i.path, i.rule, i.message))],
    }


def inspect_model(path: str | Path) -> dict[str, Any]:
    """Inspect an MJCF or URDF and return a stable, JSON-serializable report."""
    model_path = Path(path).resolve()
    try:
        root = ET.parse(model_path).getroot()
    except (OSError, ET.ParseError) as exc:
        return {"inspector": INSPECTOR_VERSION, "format": "unknown", "path": str(model_path), "counts": {}, "diagnostics": [Diagnostic("error", "INSPECT_PARSE_ERROR", str(exc), str(model_path)).to_dict()]}
    result = _inspect_mjcf(model_path, root) if root.tag == "mujoco" else _inspect_urdf(model_path, root)
    result["inspector"] = INSPECTOR_VERSION
    result["ok"] = not any(item["severity"] == "error" for item in result["diagnostics"])
    return result


def _xml_child(parent: ET.Element, tag: str, **attrs: str) -> ET.Element:
    return ET.SubElement(parent, tag, {key: value for key, value in attrs.items() if value != ""})


def _parse_urdf(path: Path) -> tuple[ET.Element, dict[str, ET.Element], dict[str, ET.Element]]:
    root = ET.parse(path).getroot()
    links = {_attr(item, "name"): item for item in root.findall("link")}
    joints = {_attr(item, "name"): item for item in root.findall("joint")}
    if not links or not joints:
        raise ValueError("URDF 必须至少包含 link 和 joint")
    return root, links, joints


def convert_urdf_to_mjcf(urdf_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Convert the supported URDF subset used by the FireDog export.

    The converter is deterministic and deliberately records every simulation-only
    addition in the returned metadata. It does not depend on ROS or a vendor SDK.
    """
    source = Path(urdf_path).resolve()
    output = Path(output_path).resolve()
    _, links, joints = _parse_urdf(source)
    parent_of = {_attr(j.find("child"), "link"): _attr(j.find("parent"), "link") for j in joints.values()}
    roots = sorted(set(links) - set(parent_of))
    if len(roots) != 1:
        raise ValueError(f"URDF root link 必须唯一，实际 {roots}")
    children: dict[str, list[ET.Element]] = {name: [] for name in links}
    for joint in joints.values():
        children[_attr(joint.find("parent"), "link")].append(joint)
    leg_order = {name: index for index, name in enumerate(("RF", "RR", "LR", "LF"))}
    for items in children.values():
        items.sort(key=lambda item: (leg_order.get(_attr(item.find("child"), "link"), 99), _attr(item, "name")))

    mj = ET.Element("mujoco", {"model": "firedog2_2"})
    _xml_child(mj, "compiler", angle="radian", autolimits="true")
    _xml_child(mj, "option", timestep="0.002", gravity="0 0 -9.81")
    asset = _xml_child(mj, "asset")
    mesh_files: dict[str, str] = {}
    for link_name in sorted(links):
        mesh = links[link_name].find("visual/geometry/mesh")
        if mesh is None:
            continue
        reference = _attr(mesh, "filename")
        filename = Path(reference.split("/meshes/", 1)[-1]).name
        mesh_path = (source.parent.parent / "meshes" / filename).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"mesh missing: {mesh_path}")
        mesh_files[filename] = f"../meshes/{filename}"
    for filename in sorted(mesh_files):
        _xml_child(asset, "mesh", name=Path(filename).stem, file=mesh_files[filename])

    worldbody = _xml_child(mj, "worldbody")
    foot_sites: list[str] = []
    joint_specs: list[tuple[str, str, str, float | None, float | None, float, float]] = []

    def add_link(parent: ET.Element, link_name: str, joint: ET.Element | None = None) -> None:
        link = links[link_name]
        attrs: dict[str, str] = {"name": link_name}
        if joint is not None:
            origin = joint.find("origin")
            if origin is not None:
                attrs["pos"] = _attr(origin, "xyz", "0 0 0")
                if _attr(origin, "rpy"):
                    attrs["euler"] = _attr(origin, "rpy")
        body = _xml_child(parent, "body", **attrs)
        if link_name == roots[0]:
            _xml_child(body, "freejoint", name="base_free")
        inertial = link.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            mass = inertial.find("mass")
            inertia = inertial.find("inertia")
            if mass is not None and inertia is not None:
                i = [_attr(inertia, key, "0") for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
                _xml_child(body, "inertial", pos=_attr(origin, "xyz", "0 0 0") if origin is not None else "0 0 0", mass=_attr(mass, "value"), fullinertia=" ".join(i))
        visual_mesh = link.find("visual/geometry/mesh")
        if visual_mesh is not None:
            filename = Path(_attr(visual_mesh, "filename").split("/meshes/", 1)[-1]).stem
            material = link.find("visual/material/color")
            rgba = _attr(material, "rgba", "0.65 0.7 0.8 1") if material is not None else "0.65 0.7 0.8 1"
            _xml_child(body, "geom", name=f"{link_name}_visual", type="mesh", mesh=filename, contype="0", conaffinity="0", rgba=rgba)
            _xml_child(body, "geom", name=f"{link_name}_collision", type="mesh", mesh=filename, group="0", contype="1", conaffinity="1", rgba=rgba)
        if link_name.endswith("3"):
            foot = f"{link_name}_foot"
            _xml_child(body, "site", name=foot, pos="0 0 0", size="0.02", type="sphere", rgba="0.2 0.8 0.2 0.8")
            foot_sites.append(foot)
        if link_name == roots[0]:
            _xml_child(body, "site", name="imu", pos="0 0 0.06", size="0.01", type="sphere")
        if joint is not None:
            joint_name = _attr(joint, "name")
            joint_type = _attr(joint, "type")
            axis = _attr(joint.find("axis"), "xyz", "0 0 1")
            limit = joint.find("limit")
            effort = float(_attr(limit, "effort", "40")) if limit is not None else 40.0
            velocity = float(_attr(limit, "velocity", "12.5664")) if limit is not None else 12.5664
            lower = float(_attr(limit, "lower")) if limit is not None and _attr(limit, "lower") else None
            upper = float(_attr(limit, "upper")) if limit is not None and _attr(limit, "upper") else None
            jattrs = {"name": joint_name, "type": "hinge", "axis": axis, "damping": "0.2", "frictionloss": "0.01"}
            if lower is not None and upper is not None:
                jattrs["range"] = f"{lower:.9g} {upper:.9g}"
            else:
                jattrs["limited"] = "false"
                lower, upper = -math.pi, math.pi
            _xml_child(body, "joint", **jattrs)
            joint_specs.append((joint_name, joint_type, axis, lower, upper, effort, velocity))
        for child_joint in children.get(link_name, []):
            add_link(body, _attr(child_joint.find("child"), "link"), child_joint)

    add_link(worldbody, roots[0])
    actuator = _xml_child(mj, "actuator")
    for name, _, _, lower, upper, effort, _ in joint_specs:
        _xml_child(actuator, "position", name=f"act_{name}", joint=name, kp="40", kv="4", ctrlrange=f"{lower:.9g} {upper:.9g}", forcerange=f"{-effort:.9g} {effort:.9g}")
    sensor = _xml_child(mj, "sensor")
    for name, *_ in joint_specs:
        _xml_child(sensor, "jointpos", name=f"sense_{name}_position", joint=name)
        _xml_child(sensor, "jointvel", name=f"sense_{name}_velocity", joint=name)
    _xml_child(sensor, "accelerometer", name="imu_accel", site="imu")
    _xml_child(sensor, "gyro", name="imu_gyro", site="imu")
    keyframe = _xml_child(mj, "keyframe")
    qpos = [0.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0] + [0.0] * len(joint_specs)
    _xml_child(keyframe, "key", name="default", qpos=_fmt(qpos), ctrl=_fmt([0.0] * len(joint_specs)))

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(mj, space="  ")
    xml = ET.tostring(mj, encoding="unicode") + "\n"
    output.write_text(xml, encoding="utf-8")
    return {
        "converter": INSPECTOR_VERSION,
        "source": str(source),
        "output": str(output),
        "sourceSha256": _sha256(source),
        "outputSha256": _sha256(output),
        "command": f"robolab robot convert {source} --output {output}",
        "manualModifications": [
            "ROS package:// mesh references normalized to ../meshes/*.STL",
            "position actuators with kp=40, kv=4 and force ranges derived from URDF effort limits",
            "jointpos/jointvel, IMU, foot sites and collision geoms added for RoboLab simulation",
            "continuous wheel position sweep represented by [-pi, pi] in JointSet; MJCF hinge remains unlimited",
        ],
        "reproducible": True,
    }


def json_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
