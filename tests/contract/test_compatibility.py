"""B1.4 compatibility tests with explainable outcomes."""

from __future__ import annotations

import copy

from robolab_core import check_compatibility


def _rules(report) -> set[str]:
    return {i.rule for i in report.issues}


def _error_rules(report) -> set[str]:
    return {i.rule for i in report.errors}


class TestMotionSkillCompatibility:
    def test_reference_motion_matches_reference_profile(self, motion_skill, reference_profile):
        report = check_compatibility(motion_skill, reference_profile)
        assert report.ok, report.render()
        # Every key successful rule should be reported as informational.
        for rule in (
            "compat.profile-version",
            "compat.joint-set",
            "compat.control-mode",
            "compat.control-hz",
            "compat.simulation",
        ):
            assert rule in _rules(report)

    def test_undeclared_profile(self, motion_skill, reference_profile):
        profile = copy.deepcopy(reference_profile)
        profile["metadata"]["id"] = "test.other_biped"
        report = check_compatibility(motion_skill, profile)
        assert not report.ok
        assert "compat.profile-not-declared" in _error_rules(report)
        # The error should enumerate the Skill's declared Profiles.
        message = next(i for i in report.errors if i.rule == "compat.profile-not-declared").message
        assert "test.reference_biped" in message

    def test_profile_version_out_of_range(self, motion_skill, reference_profile):
        profile = copy.deepcopy(reference_profile)
        profile["metadata"]["version"] = "2.1.0"
        report = check_compatibility(motion_skill, profile)
        assert "compat.profile-version" in _error_rules(report)

    def test_joint_set_mismatch(self, motion_skill, reference_profile):
        profile = copy.deepcopy(reference_profile)
        profile["description"]["jointSet"] = "test.other_biped.joints.v1"
        report = check_compatibility(motion_skill, profile)
        assert "compat.joint-set" in _error_rules(report)

    def test_control_mode_mismatch(self, motion_skill, reference_profile):
        profile = copy.deepcopy(reference_profile)
        profile["control"]["mode"] = "joint_torque"
        report = check_compatibility(motion_skill, profile)
        assert "compat.control-mode" in _error_rules(report)

    def test_control_hz_mismatch(self, motion_skill, reference_profile):
        profile = copy.deepcopy(reference_profile)
        profile["control"]["frequencyHz"] = 200
        report = check_compatibility(motion_skill, profile)
        assert "compat.control-hz" in _error_rules(report)

    def test_simulation_target_required(self, motion_skill, reference_profile):
        profile = copy.deepcopy(reference_profile)
        profile["targets"]["simulation"]["enabled"] = False
        profile["capabilities"]["simulation"] = False
        report = check_compatibility(motion_skill, profile)
        assert "compat.simulation" in _error_rules(report)

    def test_deploy_action_explains_physical_gate(self, motion_skill, reference_profile):
        report = check_compatibility(motion_skill, reference_profile)
        deploy_notes = [i for i in report.issues if i.rule == "compat.deploy"]
        assert deploy_notes and deploy_notes[0].severity == "info"
        assert "physicalDeployment" in deploy_notes[0].message


class TestNonMotionSkill:
    def test_platform_skill_not_robot_bound(self, platform_skill, reference_profile):
        report = check_compatibility(platform_skill, reference_profile)
        assert report.ok
        assert "compat.not-robot-bound" in _rules(report)

    def test_agent_skill_not_robot_bound(self, agent_skill, reference_profile):
        report = check_compatibility(agent_skill, reference_profile)
        assert report.ok
