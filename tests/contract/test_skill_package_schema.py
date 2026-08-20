"""B1.2 structural tests for all SkillPackage kinds."""

from __future__ import annotations

import copy

from robolab_schemas import detect_kind, validate_schema


class TestSkillKinds:
    def test_motion_fixture_passes(self, motion_skill):
        assert validate_schema(motion_skill) == []

    def test_platform_fixture_passes(self, platform_skill):
        assert validate_schema(platform_skill) == []

    def test_agent_fixture_passes(self, agent_skill):
        assert validate_schema(agent_skill) == []

    def test_detect_kind(self, motion_skill, platform_skill, agent_skill):
        assert detect_kind(motion_skill) == "MotionSkill"
        assert detect_kind(platform_skill) == "PlatformSkill"
        assert detect_kind(agent_skill) == "AgentSkill"


class TestMotionSkillRequirements:
    def test_missing_safety_rejected(self, motion_skill):
        doc = copy.deepcopy(motion_skill)
        del doc["spec"]["safety"]
        assert validate_schema(doc)

    def test_missing_artifacts_rejected(self, motion_skill):
        doc = copy.deepcopy(motion_skill)
        del doc["spec"]["artifacts"]
        assert validate_schema(doc)

    def test_missing_robot_compatibility_rejected(self, motion_skill):
        doc = copy.deepcopy(motion_skill)
        del doc["spec"]["compatibility"]["robots"]
        assert validate_schema(doc)

    def test_runtime_type_must_be_onnx(self, motion_skill):
        doc = copy.deepcopy(motion_skill)
        doc["spec"]["runtime"]["type"] = "python"
        assert validate_schema(doc)

    def test_bad_artifact_hash_format(self, motion_skill):
        doc = copy.deepcopy(motion_skill)
        doc["spec"]["artifacts"][0]["sha256"] = "not-a-hash"
        assert validate_schema(doc)

    def test_uppercase_hash_rejected(self, motion_skill):
        doc = copy.deepcopy(motion_skill)
        doc["spec"]["artifacts"][0]["sha256"] = "A" * 64
        assert validate_schema(doc)

    def test_or_version_range_rejected(self, motion_skill):
        doc = copy.deepcopy(motion_skill)
        doc["spec"]["compatibility"]["robots"][0]["version"] = ">=1.0.0 | <2.0.0"
        assert validate_schema(doc)


class TestPlatformSkillRequirements:
    def test_protocol_must_be_job_v1(self, platform_skill):
        doc = copy.deepcopy(platform_skill)
        doc["spec"]["runtime"]["protocol"] = "robolab-motion-v1"
        assert validate_schema(doc)

    def test_missing_entrypoint_rejected(self, platform_skill):
        doc = copy.deepcopy(platform_skill)
        del doc["spec"]["runtime"]["entrypoint"]
        assert validate_schema(doc)

    def test_environment_mode_enum(self, platform_skill):
        doc = copy.deepcopy(platform_skill)
        doc["spec"]["runtime"]["environment"]["mode"] = "docker"
        assert validate_schema(doc)


class TestAgentSkillRequirements:
    def test_missing_instructions_rejected(self, agent_skill):
        doc = copy.deepcopy(agent_skill)
        del doc["spec"]["runtime"]["instructions"]
        assert validate_schema(doc)

    def test_missing_tools_rejected(self, agent_skill):
        doc = copy.deepcopy(agent_skill)
        del doc["spec"]["tools"]
        assert validate_schema(doc)

    def test_tool_name_must_be_platform_tool(self, agent_skill):
        doc = copy.deepcopy(agent_skill)
        doc["spec"]["tools"]["allow"] = ["shell.exec"]
        assert validate_schema(doc)


class TestCommonManifest:
    def test_skill_api_const(self, motion_skill):
        doc = copy.deepcopy(motion_skill)
        doc["spec"]["compatibility"]["skillApi"] = "v2"
        assert validate_schema(doc)

    def test_permission_write_outside_run_rejected(self, platform_skill):
        doc = copy.deepcopy(platform_skill)
        doc["spec"]["permissions"]["filesystem"]["write"] = ["workspace"]
        assert validate_schema(doc)

    def test_unknown_top_level_field_rejected(self, platform_skill):
        doc = copy.deepcopy(platform_skill)
        doc["extra"] = True
        assert validate_schema(doc)
