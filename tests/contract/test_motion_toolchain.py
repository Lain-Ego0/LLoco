from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from robolab_core import (
    ArtifactRef,
    ConfigurationValidationError,
    DuplicateRegistryEntryError,
    ResourceConfig,
    RobotBinding,
    RobotRegistry,
    ToolchainIdentity,
    UnknownRegistryEntryError,
    build_evaluate_command,
    build_export_command,
    build_play_command,
    build_train_command,
    default_robot_registry,
    default_task_registry,
    make_evaluation_result,
    persist_motion_job,
)
from robolab_schemas import validate_motion_command


def _identity() -> ToolchainIdentity:
    return ToolchainIdentity("1" * 40)


def _artifact(tmp_path: Path, name: str = "checkpoint.pt") -> ArtifactRef:
    path = tmp_path / name
    path.write_bytes(b"checkpoint")
    from robolab_core import artifact_ref_from_path

    return artifact_ref_from_path(path, media_type="application/octet-stream")


def test_toolchain_identity_is_stable_and_embedded() -> None:
    identity = _identity()
    assert identity.to_dict() == {
        "id": "robolab_mjlab@1",
        "mjlabUpstreamRevision": "0fb8a681136be94ffc636a3dd423cabb97d91f10",
        "robolabRevision": "1" * 40,
    }


def test_robot_registry_is_versioned_and_not_vendor_path_based() -> None:
    entry = RobotBinding(
        "robolab.sim.example",
        "1.0.0",
        "robolab_mjlab.entities.ExampleEntity",
        "robolab_mjlab.configs.example",
    )
    registry = RobotRegistry([entry])
    assert registry.resolve("robolab.sim.example", "1.0.0") == entry
    with pytest.raises(UnknownRegistryEntryError):
        registry.resolve("robolab.sim.example", "2.0.0")
    with pytest.raises(DuplicateRegistryEntryError):
        registry.register(entry)
    assert default_robot_registry().list() == []


def test_task_registry_validates_config_and_keeps_robot_separate() -> None:
    registry = default_task_registry()
    task = registry.resolve("robolab.motion.smoke.cartpole", "1.0.0")
    assert task.capability == "simulation.cpu_smoke"
    assert (
        registry.validate_config(task.id, {"numEnvs": 1, "device": "cpu"})["device"]
        == "cpu"
    )
    with pytest.raises(ConfigurationValidationError):
        registry.validate_config(task.id, {"device": "cuda"})
    assert task.to_dict()["requiresRobot"] is False


def test_train_command_contains_reproducible_cpu_config(tmp_path: Path) -> None:
    command = build_train_command(
        task_id="robolab.motion.smoke.cartpole",
        resolved_config={"numEnvs": 1, "device": "cpu"},
        seed=7,
        resources=ResourceConfig(),
        output_dir=tmp_path / "train",
        toolchain=_identity(),
    )
    document = command.to_dict()
    assert validate_motion_command(document) == []
    assert document["spec"]["seed"] == 7
    assert document["runtime"]["backend"] == "customized_mjlab_1_6"


def test_play_command_rejects_unhashed_placeholder_and_records_determinism(
    tmp_path: Path,
) -> None:
    checkpoint = _artifact(tmp_path)
    command = build_play_command(
        task_id="robolab.motion.smoke.cartpole",
        resolved_config={"device": "cpu"},
        checkpoint=checkpoint,
        viewer="none",
        recording={"enabled": False},
        deterministic=True,
        toolchain=_identity(),
    )
    assert command.to_dict()["spec"]["deterministic"] is True
    with pytest.raises(ValueError):
        ArtifactRef("checkpoint.pt", "0" * 63, "application/octet-stream")


def test_evaluate_command_has_machine_readable_evidence() -> None:
    command = build_evaluate_command(
        task_id="robolab.motion.smoke.cartpole",
        resolved_config={"device": "cpu"},
        scene="flat",
        episodes=3,
        metrics=("mean_reward", "success_rate"),
        thresholds={"mean_reward": 0.5},
        evidence_dir="var/evidence/run-1",
        toolchain=_identity(),
    )
    result = make_evaluation_result(command, {"mean_reward": 0.2, "success_rate": 0.9})
    assert result["status"] == "passed"
    assert result["metrics"]["mean_reward"]["passed"] is True
    assert result["evidence"]["toolchain"]["id"] == "robolab_mjlab@1"


def test_export_command_contains_policy_artifact_metadata(tmp_path: Path) -> None:
    source = _artifact(tmp_path, "policy.onnx")
    command = build_export_command(
        task_id="robolab.motion.smoke.cartpole",
        resolved_config={"device": "cpu"},
        source=source,
        observation_schema="robolab.obs.cartpole@1",
        action_schema="robolab.action.cartpole@1",
        control_frequency_hz=50,
        action_scale=0.25,
        joint_order=("hinge",),
        metadata={"producer": "robolab.export"},
        toolchain=_identity(),
        output=tmp_path / "artifact",
    )
    artifact = command.to_dict()["result"]["artifact"]
    assert artifact["source"]["sha256"] == source.sha256
    assert artifact["jointOrder"] == ["hinge"]
    assert artifact["deployment"]["controlFrequencyHz"] == 50


def test_persist_motion_job_writes_command_and_metadata(tmp_path: Path) -> None:
    command = build_train_command(
        task_id="robolab.motion.smoke.cartpole",
        resolved_config={"device": "cpu"},
        seed=1,
        resources=ResourceConfig(),
        output_dir="var/runs/train",
        toolchain=_identity(),
    )
    paths = persist_motion_job(tmp_path / "runs", command, job_id="r1-test")
    input_data = json.loads(paths.input.read_text())
    command_data = json.loads((paths.run_dir / "job_command.json").read_text())
    assert input_data["metadata"]["commandProtocol"] == "robolab-motion-v1"
    assert command_data == command.to_dict()


def test_motion_schema_rejects_upstream_environment_id() -> None:
    document = build_train_command(
        task_id="robolab.motion.smoke.cartpole",
        resolved_config={"device": "cpu"},
        seed=1,
        resources=ResourceConfig(),
        output_dir="out",
        toolchain=_identity(),
    ).to_dict()
    broken = copy.deepcopy(document)
    broken["task"]["id"] = "Mjlab-Cartpole-Balance"
    assert validate_motion_command(broken)
