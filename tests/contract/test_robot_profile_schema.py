"""B1.1 RobotProfile schema and semantic checks."""

from __future__ import annotations

import copy

from robolab_core import check_robot_profile, validate_document
from robolab_schemas import validate_schema


class TestRobotProfileSchema:
    def test_g1_fixture_passes(self, g1_profile):
        assert validate_schema(g1_profile) == []

    def test_wrong_api_version_explained(self, g1_profile):
        doc = copy.deepcopy(g1_profile)
        doc["apiVersion"] = "robolab.dev/v2"
        errors = validate_schema(doc, "RobotProfile")
        assert errors, "unknown apiVersion must be rejected"
        assert any("apiVersion" in e.message or "v1alpha1" in e.message for e in errors)

    def test_missing_capabilities_fails(self, g1_profile):
        doc = copy.deepcopy(g1_profile)
        del doc["capabilities"]
        errors = validate_schema(doc, "RobotProfile")
        assert any("capabilities" in e.message for e in errors)

    def test_unknown_field_rejected(self, g1_profile):
        doc = copy.deepcopy(g1_profile)
        doc["metadata"]["ip"] = "192.168.1.10"
        errors = validate_schema(doc, "RobotProfile")
        assert errors

    def test_non_semver_version_rejected(self, g1_profile):
        doc = copy.deepcopy(g1_profile)
        doc["metadata"]["version"] = "1.0"
        assert validate_schema(doc, "RobotProfile")

    def test_absolute_mjcf_path_rejected(self, g1_profile):
        doc = copy.deepcopy(g1_profile)
        doc["description"]["mjcf"] = "/opt/models/g1.xml"
        assert validate_schema(doc, "RobotProfile")

    def test_parent_escape_rejected(self, g1_profile):
        doc = copy.deepcopy(g1_profile)
        doc["description"]["mjcf"] = "../g1.xml"
        assert validate_schema(doc, "RobotProfile")

    def test_fallback_state_enum(self, g1_profile):
        doc = copy.deepcopy(g1_profile)
        doc["safety"]["fallbackState"] = "estop"
        assert validate_schema(doc, "RobotProfile")


class TestRobotProfileSemantics:
    def test_g1_fixture_passes(self, g1_profile):
        report = validate_document(g1_profile)
        assert report.ok, report.render()

    def test_physical_deployment_requires_prerequisites(self, g1_profile):
        doc = copy.deepcopy(g1_profile)
        doc["capabilities"]["physicalDeployment"] = True
        doc["targets"]["physical"] = {"enabled": True, "driver": "unitree_sdk2"}
        issues = check_robot_profile(doc)
        assert any(i.rule == "profile.physical-prerequisites" for i in issues)

    def test_physical_enabled_requires_driver(self, g1_profile):
        doc = copy.deepcopy(g1_profile)
        doc["targets"]["physical"] = {"enabled": True, "driver": None}
        issues = check_robot_profile(doc)
        assert any(i.rule == "profile.physical-driver-missing" for i in issues)

    def test_simulation_capability_consistency(self, g1_profile):
        doc = copy.deepcopy(g1_profile)
        doc["targets"]["simulation"]["enabled"] = False
        issues = check_robot_profile(doc)
        assert any(i.rule == "profile.simulation-inconsistent" for i in issues)
