"""R2 Inspector, Profile v1beta1, mapping, conversion and snapshot contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from robolab_core import (
    check_joint_set,
    convert_urdf_to_mjcf,
    inspect_model,
    resolve_robot_config,
    validate_document,
)
from robolab_schemas import validate_schema

ROOT = Path(__file__).resolve().parents[2]
FIRE = ROOT / "robots/firedog2.2.SLDASM"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_firedog_profile_and_mapping_are_valid() -> None:
    profile = _load(FIRE / "profile.yaml")
    mapping = _load(FIRE / "bindings/actuator_sensor_mapping.yaml")
    assert validate_schema(profile) == []
    assert validate_document(profile, package_dir=FIRE).ok
    assert validate_document(mapping, package_dir=FIRE).ok
    assert len(_load(FIRE / "joint_set.yaml")["joints"]) == 16
    assert len(mapping["actuators"]) == 16
    assert len(mapping["sensors"]) == 34


def test_profile_vnext_rejects_physical_capability() -> None:
    profile = _load(FIRE / "profile.yaml")
    profile["capabilities"]["motorCommunication"] = True
    assert validate_schema(profile)


def test_invalid_vnext_fixture_rejects_absolute_path() -> None:
    invalid = _load(Path(__file__).parent / "fixtures/robot_profile.firedog2_2.invalid.yaml")
    assert validate_schema(invalid)


def test_firedog_mjcf_inspector_has_stable_inventory() -> None:
    report = inspect_model(FIRE / "model/firedog2_2.xml")
    assert report["ok"] is True
    assert report["counts"] == {"body": 17, "joint": 16, "actuator": 16, "sensor": 34, "site": 5, "mesh": 17, "keyframe": 1, "geom": 34}
    assert report["rootBodies"] == ["base_link"]


def test_inspector_reports_duplicate_missing_and_bad_root(tmp_path: Path) -> None:
    broken = tmp_path / "broken.xml"
    broken.write_text("""<mujoco><asset><mesh name='m' file='missing.stl'/><mesh name='m' file='missing.stl'/></asset><worldbody><body name='a'/><body name='a'/></worldbody><actuator><position name='a' joint='nope'/><position name='a' joint='nope'/></actuator></mujoco>""", encoding="utf-8")
    report = inspect_model(broken)
    rules = {item["rule"] for item in report["diagnostics"]}
    assert {"INSPECT_DUPLICATE_NAME", "INSPECT_MESH_MISSING", "INSPECT_ROOT_BODY_INVALID", "INSPECT_REFERENCE_MISSING"} <= rules
    assert all(item["path"] for item in report["diagnostics"])


def test_joint_mapping_rejects_duplicate_and_bad_limit() -> None:
    joint_set = _load(FIRE / "joint_set.yaml")
    broken = copy.deepcopy(joint_set)
    broken["joints"][1]["name"] = broken["joints"][0]["name"]
    broken["joints"][2]["limits"]["position"] = {"lower": 1.0, "upper": -1.0}
    rules = {issue.rule for issue in check_joint_set(broken)}
    assert "joint.duplicate-name" in rules
    assert "joint.limit-direction" in rules


def test_snapshot_is_sorted_and_reproducible() -> None:
    first = resolve_robot_config(FIRE / "profile.yaml", ROOT, "community.firedog2_2.velocity.flat")
    second = resolve_robot_config(FIRE / "profile.yaml", ROOT, "community.firedog2_2.velocity.flat")
    assert first == second
    assert first["snapshotSha256"] == "9fd3a33b4168035cba0d4d10c491b7ae4f08078bd361b0c5cfbb253289fd9f35"
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(second, ensure_ascii=False, sort_keys=True)


def test_snapshot_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="unknown task binding"):
        resolve_robot_config(FIRE / "profile.yaml", ROOT, "community.firedog2_2.unknown")


def test_urdf_to_mjcf_conversion_is_reproducible(tmp_path: Path) -> None:
    first = convert_urdf_to_mjcf(FIRE / "urdf/火狗2.2.SLDASM.urdf", tmp_path / "first.xml")
    second = convert_urdf_to_mjcf(FIRE / "urdf/火狗2.2.SLDASM.urdf", tmp_path / "second.xml")
    assert first["outputSha256"] == second["outputSha256"]
    assert first["sourceSha256"] == "9a98cffad4c2559e881bc9b36a596aea58f1ab0658ee386a5f44e1ccf1119ba1"
