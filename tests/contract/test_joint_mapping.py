"""B1.3 joint-mapping validation tests."""

from __future__ import annotations

import copy

from robolab_core import check_joint_set, validate_document


def _rules(issues) -> set[str]:
    return {i.rule for i in issues}


class TestJointSetValid:
    def test_reference_fixture_passes(self, reference_joint_set):
        report = validate_document(reference_joint_set)
        assert report.ok, report.render()
        assert report.warnings == [], report.render()


class TestNameUniqueness:
    def test_duplicate_canonical_name(self, reference_joint_set):
        doc = copy.deepcopy(reference_joint_set)
        doc["joints"][1]["name"] = doc["joints"][0]["name"]
        assert "joint.duplicate-name" in _rules(check_joint_set(doc))

    def test_duplicate_mjcf_joint(self, reference_joint_set):
        doc = copy.deepcopy(reference_joint_set)
        doc["joints"][2]["mjcf"]["joint"] = doc["joints"][0]["mjcf"]["joint"]
        rules = _rules(check_joint_set(doc))
        assert "joint.duplicate-name" in rules or "joint.not-bijective" in rules

    def test_duplicate_actuator(self, reference_joint_set):
        doc = copy.deepcopy(reference_joint_set)
        doc["joints"][3]["mjcf"]["actuator"] = doc["joints"][0]["mjcf"]["actuator"]
        assert "joint.duplicate-actuator" in _rules(check_joint_set(doc))


class TestBijectionAndSize:
    def test_dof_mismatch(self, reference_joint_set):
        doc = copy.deepcopy(reference_joint_set)
        doc["joints"] = doc["joints"][:-1]
        assert "joint.count-mismatch" in _rules(check_joint_set(doc))

    def test_actuator_index_consistency(self, reference_joint_set):
        doc = copy.deepcopy(reference_joint_set)
        doc["joints"][0]["mjcf"]["actuatorIndex"] = None
        assert "joint.actuator-index-inconsistent" in _rules(check_joint_set(doc))

    def test_unactuated_joint_warns(self, reference_joint_set):
        doc = copy.deepcopy(reference_joint_set)
        doc["joints"][0]["mjcf"]["actuator"] = None
        doc["joints"][0]["mjcf"]["actuatorIndex"] = None
        issues = check_joint_set(doc)
        assert "joint.unactuated" in _rules(issues)
        assert all(i.severity != "error" or i.rule != "joint.actuator-index-inconsistent" for i in issues)


class TestIndexContinuity:
    def test_action_index_gap(self, reference_joint_set):
        doc = copy.deepcopy(reference_joint_set)
        doc["joints"][5]["policy"]["actionIndex"] = 29
        assert "joint.index-not-contiguous" in _rules(check_joint_set(doc))

    def test_action_index_duplicate(self, reference_joint_set):
        doc = copy.deepcopy(reference_joint_set)
        doc["joints"][1]["policy"]["actionIndex"] = doc["joints"][0]["policy"]["actionIndex"]
        assert "joint.index-duplicate" in _rules(check_joint_set(doc))

    def test_qvel_overlaps_free_base(self, reference_joint_set):
        doc = copy.deepcopy(reference_joint_set)
        doc["joints"][0]["mjcf"]["qvelIndex"] = 5
        assert "joint.index-overlaps-base" in _rules(check_joint_set(doc))


class TestLimitsAndPose:
    def test_inverted_position_limit(self, reference_joint_set):
        doc = copy.deepcopy(reference_joint_set)
        pos = doc["joints"][0]["limits"]["position"]
        pos["lower"], pos["upper"] = pos["upper"], pos["lower"]
        assert "joint.limit-direction" in _rules(check_joint_set(doc))

    def test_default_pose_out_of_range(self, reference_joint_set):
        doc = copy.deepcopy(reference_joint_set)
        doc["joints"][0]["defaultPosition"] = 99.0
        assert "joint.default-out-of-range" in _rules(check_joint_set(doc))

    def test_recommended_gains_exceed_maximum(self, reference_joint_set):
        doc = copy.deepcopy(reference_joint_set)
        doc["joints"][0]["gains"]["maximum"]["kp"] = 1.0
        assert "joint.gains-order" in _rules(check_joint_set(doc))
