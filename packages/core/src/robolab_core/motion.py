"""R1 RoboLab motion-toolchain contracts.

This module deliberately stops at command construction.  It does not start a
training process, create a viewer, or claim that a policy exists.  The
resulting ``JobCommand`` is the stable object shared by future CLI, API, and
Worker integrations.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator
from robolab_schemas import validate_motion_command

from robolab_core.jobs import JobPaths, create_job_run
from robolab_core.versioning import SemVer

MOTION_PROTOCOL = "robolab-motion-v1"
TOOLCHAIN_ID = "robolab_mjlab@1"
MJLAB_UPSTREAM_REVISION = "0fb8a681136be94ffc636a3dd423cabb97d91f10"

_ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$")
_PYTHON_REF_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

Operation = Literal["train", "play", "evaluate", "export"]


class MotionContractError(ValueError):
    """Base error for invalid R1 motion contracts."""


class RegistryError(MotionContractError):
    """Base error for registry lookup and registration failures."""


class DuplicateRegistryEntryError(RegistryError):
    """Raised when an id/version pair is registered twice."""


class UnknownRegistryEntryError(RegistryError):
    """Raised when a requested id/version pair is not registered."""


class ConfigurationValidationError(MotionContractError):
    """Raised when task configuration does not match its declared schema."""


def _require_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise MotionContractError(
            f"{field_name} 必须是稳定的小写标识（例如 robolab.motion.smoke）"
        )
    return value


def _require_version(value: str, field_name: str = "version") -> str:
    try:
        SemVer.parse(value)
    except ValueError as exc:
        raise MotionContractError(f"{field_name} 必须是合法 SemVer: {value!r}") from exc
    return value


def _require_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MotionContractError(f"{field_name} 必须是 object")
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise MotionContractError(f"{field_name} 必须可 JSON 序列化") from exc
    return dict(value)


def _require_sha256(value: str, field_name: str = "sha256") -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise MotionContractError(f"{field_name} 必须是 64 位小写 SHA-256")
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ToolchainIdentity:
    """Reproducible identity of the RoboLab/MJLab execution base."""

    robolab_revision: str
    toolchain_id: str = TOOLCHAIN_ID
    mjlab_upstream_revision: str = MJLAB_UPSTREAM_REVISION

    def __post_init__(self) -> None:
        if self.toolchain_id != TOOLCHAIN_ID:
            raise MotionContractError(
                f"不支持的 toolchain identity: {self.toolchain_id!r}"
            )
        if not _REVISION_RE.fullmatch(self.mjlab_upstream_revision):
            raise MotionContractError("MJLab upstream revision 必须是完整 commit SHA")
        if not _REVISION_RE.fullmatch(self.robolab_revision):
            raise MotionContractError("RoboLab revision 必须是完整 commit SHA")

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.toolchain_id,
            "mjlabUpstreamRevision": self.mjlab_upstream_revision,
            "robolabRevision": self.robolab_revision,
        }


def resolve_toolchain_identity(
    repo_root: str | Path | None = None,
) -> ToolchainIdentity:
    """Read the current full RoboLab revision without mutating the repository."""

    root = Path(repo_root or Path(__file__).resolve().parents[4])
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MotionContractError(f"无法解析 RoboLab revision: {root}") from exc
    return ToolchainIdentity(robolab_revision=revision)


@dataclass(frozen=True)
class RobotBinding:
    """Stable RoboLab profile -> MJLab entity/config binding.

    ``entity`` and ``config`` are import-style implementation references, not
    filesystem paths and not upstream environment identifiers.
    """

    id: str
    version: str
    entity: str
    config: str
    capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.id, "robot id")
        _require_version(self.version)
        if not _PYTHON_REF_RE.fullmatch(self.entity):
            raise MotionContractError("robot entity 必须是 import-style reference")
        if not _PYTHON_REF_RE.fullmatch(self.config):
            raise MotionContractError("robot config 必须是 import-style reference")
        _require_mapping(self.metadata, "robot metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "mjlab": {"entity": self.entity, "config": self.config},
            "capabilities": list(self.capabilities),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TaskDefinition:
    """Stable task contract independent from a concrete RobotBinding."""

    id: str
    version: str
    capability: str
    config_schema: Mapping[str, Any]
    entrypoint: str
    requires_robot: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.id, "task id")
        _require_version(self.version)
        if not isinstance(self.capability, str) or not self.capability:
            raise MotionContractError("task capability 不能为空")
        if not _PYTHON_REF_RE.fullmatch(self.entrypoint):
            raise MotionContractError("task entrypoint 必须是 import-style reference")
        schema = _require_mapping(self.config_schema, "task config_schema")
        if not isinstance(schema.get("type"), str):
            raise MotionContractError("task config_schema 必须声明 JSON Schema type")
        _require_mapping(self.metadata, "task metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "capability": self.capability,
            "configSchema": dict(self.config_schema),
            "entrypoint": self.entrypoint,
            "requiresRobot": self.requires_robot,
            "metadata": dict(self.metadata),
        }


class RobotRegistry:
    """In-process registry keyed by stable ``id@version`` pairs."""

    def __init__(self, entries: Sequence[RobotBinding] = ()) -> None:
        self._entries: dict[tuple[str, str], RobotBinding] = {}
        for entry in entries:
            self.register(entry)

    def register(self, entry: RobotBinding) -> None:
        key = (entry.id, entry.version)
        if key in self._entries:
            raise DuplicateRegistryEntryError(
                f"Robot 已注册: {entry.id}@{entry.version}"
            )
        self._entries[key] = entry

    def resolve(self, robot_id: str, version: str | None = None) -> RobotBinding:
        _require_id(robot_id, "robot id")
        matches = [
            entry
            for (entry_id, _), entry in self._entries.items()
            if entry_id == robot_id
        ]
        if version is not None:
            _require_version(version)
            matches = [entry for entry in matches if entry.version == version]
        if len(matches) != 1:
            suffix = f"@{version}" if version else ""
            raise UnknownRegistryEntryError(f"未知或不唯一的 Robot: {robot_id}{suffix}")
        return matches[0]

    def list(self) -> list[RobotBinding]:
        return sorted(
            self._entries.values(), key=lambda entry: (entry.id, entry.version)
        )


class TaskRegistry:
    """Registry with declared JSON-schema configuration validation."""

    def __init__(self, entries: Sequence[TaskDefinition] = ()) -> None:
        self._entries: dict[tuple[str, str], TaskDefinition] = {}
        for entry in entries:
            self.register(entry)

    def register(self, entry: TaskDefinition) -> None:
        key = (entry.id, entry.version)
        if key in self._entries:
            raise DuplicateRegistryEntryError(
                f"Task 已注册: {entry.id}@{entry.version}"
            )
        self._entries[key] = entry

    def resolve(self, task_id: str, version: str | None = None) -> TaskDefinition:
        _require_id(task_id, "task id")
        matches = [
            entry
            for (entry_id, _), entry in self._entries.items()
            if entry_id == task_id
        ]
        if version is not None:
            _require_version(version)
            matches = [entry for entry in matches if entry.version == version]
        if len(matches) != 1:
            suffix = f"@{version}" if version else ""
            raise UnknownRegistryEntryError(f"未知或不唯一的 Task: {task_id}{suffix}")
        return matches[0]

    def validate_config(
        self, task_id: str, config: Mapping[str, Any], version: str | None = None
    ) -> dict[str, Any]:
        task = self.resolve(task_id, version)
        resolved = _require_mapping(config, "resolved config")
        errors = sorted(
            Draft202012Validator(task.config_schema).iter_errors(resolved),
            key=lambda error: list(error.path),
        )
        if errors:
            path = "/".join(str(part) for part in errors[0].absolute_path)
            raise ConfigurationValidationError(
                f"Task 配置不符合 schema [{path}]: {errors[0].message}"
            )
        return resolved

    def list(self) -> list[TaskDefinition]:
        return sorted(
            self._entries.values(), key=lambda entry: (entry.id, entry.version)
        )


def default_task_registry() -> TaskRegistry:
    """Return the stable public R1 smoke and R3 velocity task definitions."""

    return TaskRegistry(
        [
            TaskDefinition(
                id="robolab.motion.smoke.cartpole",
                version="1.0.0",
                capability="simulation.cpu_smoke",
                config_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "numEnvs": {"type": "integer", "minimum": 1},
                        "device": {"enum": ["cpu"]},
                    },
                },
                entrypoint="mjlab.tasks.cartpole.cartpole_balance_env_cfg",
                metadata={"evidence": "mjlab/tests/smoke_test.py"},
            ),
            TaskDefinition(
                id="robolab.motion.velocity.flat",
                version="1.0.0",
                capability="simulation.gpu_ppo",
                config_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["numEnvs", "device"],
                    "properties": {
                        "numEnvs": {"type": "integer", "minimum": 1},
                        "device": {"enum": ["cuda"]},
                        "gpuIds": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                        },
                        "iterations": {"type": "integer", "minimum": 1},
                        "seed": {"type": "integer", "minimum": 0},
                    },
                },
                entrypoint="mjlab.tasks.firedog2_2.firedog2_2_env_cfg.firedog2_2_velocity_flat_env_cfg",
                requires_robot=True,
                metadata={
                    "observationSchema": "robolab.motion.velocity.flat.observation@1.0.0",
                    "actionSchema": "robolab.motion.velocity.flat.action@1.0.0",
                    "evidence": "mjlab/src/mjlab/tasks/firedog2_2",
                },
            ),
        ]
    )


def default_robot_registry() -> RobotRegistry:
    """Return the R2 FireDog simulation-only Robot Profile binding."""

    return RobotRegistry(
        [
            RobotBinding(
                id="community.firedog2_2",
                version="1.0.0",
                entity="mjlab.tasks.firedog2_2.firedog2_2_env_cfg.firedog2_2_velocity_env_cfg",
                config="robots.firedog2_2.profile",
                capabilities=("simulation", "training"),
                metadata={
                    "profile": "robots/firedog2.2.SLDASM/profile.yaml",
                    "physicalDeployment": False,
                },
            )
        ]
    )


@dataclass(frozen=True)
class ResourceConfig:
    device: Literal["cpu", "cuda"] = "cpu"
    num_envs: int = 1
    gpu_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise MotionContractError("num_envs 必须大于 0")
        if self.device == "cpu" and self.gpu_ids:
            raise MotionContractError("CPU resource 不应声明 gpu_ids")

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "numEnvs": self.num_envs,
            "gpuIds": list(self.gpu_ids),
        }


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    media_type: str

    def __post_init__(self) -> None:
        if not self.path:
            raise MotionContractError("artifact path 不能为空")
        _require_sha256(self.sha256)
        if not self.media_type:
            raise MotionContractError("artifact media_type 不能为空")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256, "mediaType": self.media_type}


def artifact_ref_from_path(path: str | Path, *, media_type: str) -> ArtifactRef:
    source = Path(path)
    if not source.is_file():
        raise MotionContractError(f"artifact 不存在: {source}")
    return ArtifactRef(str(source), _hash_file(source), media_type)


@dataclass(frozen=True)
class JobCommand:
    operation: Operation
    toolchain: ToolchainIdentity
    task: TaskDefinition
    robot: RobotBinding | None
    resolved_config: Mapping[str, Any]
    spec: Mapping[str, Any]
    runtime: Mapping[str, Any]
    result: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.operation not in {"train", "play", "evaluate", "export"}:
            raise MotionContractError(f"不支持的 operation: {self.operation}")
        _require_mapping(self.resolved_config, "resolved_config")
        _require_mapping(self.spec, "command spec")
        _require_mapping(self.runtime, "command runtime")
        _require_mapping(self.result, "command result")

    def to_dict(self) -> dict[str, Any]:
        document = {
            "protocol": MOTION_PROTOCOL,
            "operation": self.operation,
            "toolchain": self.toolchain.to_dict(),
            "task": {"id": self.task.id, "version": self.task.version},
            "robot": self.robot.to_dict() if self.robot else None,
            "resolvedConfig": dict(self.resolved_config),
            "spec": dict(self.spec),
            "runtime": dict(self.runtime),
            "result": dict(self.result),
        }
        errors = validate_motion_command(document)
        if errors:
            raise MotionContractError(
                f"JobCommand schema 校验失败: {errors[0].message}"
            )
        return document


def _resolve_refs(
    task_registry: TaskRegistry,
    robot_registry: RobotRegistry,
    task_id: str,
    task_version: str | None,
    robot_id: str | None,
    robot_version: str | None,
    resolved_config: Mapping[str, Any],
) -> tuple[TaskDefinition, RobotBinding | None, dict[str, Any]]:
    task = task_registry.resolve(task_id, task_version)
    config = task_registry.validate_config(task.id, resolved_config, task.version)
    robot = None
    if robot_id is not None:
        robot = robot_registry.resolve(robot_id, robot_version)
    if task.requires_robot and robot is None:
        raise MotionContractError(
            f"Task {task.id}@{task.version} 必须绑定 Robot Profile"
        )
    return task, robot, config


def _base_runtime(operation: Operation) -> dict[str, Any]:
    return {
        "toolchain": TOOLCHAIN_ID,
        "backend": "customized_mjlab_1_6",
        "operation": operation,
    }


def build_train_command(
    *,
    task_id: str,
    resolved_config: Mapping[str, Any],
    seed: int,
    resources: ResourceConfig,
    output_dir: str | Path,
    toolchain: ToolchainIdentity,
    task_registry: TaskRegistry | None = None,
    robot_registry: RobotRegistry | None = None,
    task_version: str | None = None,
    robot_id: str | None = None,
    robot_version: str | None = None,
    resume: ArtifactRef | None = None,
) -> JobCommand:
    if seed < 0:
        raise MotionContractError("seed 不能为负数")
    task, robot, config = _resolve_refs(
        task_registry or default_task_registry(),
        robot_registry or default_robot_registry(),
        task_id,
        task_version,
        robot_id,
        robot_version,
        resolved_config,
    )
    output = str(output_dir)
    if not output:
        raise MotionContractError("output_dir 不能为空")
    return JobCommand(
        "train",
        toolchain,
        task,
        robot,
        config,
        {
            "seed": seed,
            "resume": resume.to_dict() if resume else None,
            "resources": resources.to_dict(),
            "outputDir": output,
        },
        _base_runtime("train"),
        {"format": "robolab-training-result-v1", "status": "pending"},
    )


def build_play_command(
    *,
    task_id: str,
    resolved_config: Mapping[str, Any],
    checkpoint: ArtifactRef,
    viewer: Literal["none", "native", "viser"],
    recording: Mapping[str, Any],
    deterministic: bool,
    toolchain: ToolchainIdentity,
    task_registry: TaskRegistry | None = None,
    robot_registry: RobotRegistry | None = None,
    task_version: str | None = None,
    robot_id: str | None = None,
    robot_version: str | None = None,
) -> JobCommand:
    task, robot, config = _resolve_refs(
        task_registry or default_task_registry(),
        robot_registry or default_robot_registry(),
        task_id,
        task_version,
        robot_id,
        robot_version,
        resolved_config,
    )
    recording_config = _require_mapping(recording, "recording")
    return JobCommand(
        "play",
        toolchain,
        task,
        robot,
        config,
        {
            "checkpoint": checkpoint.to_dict(),
            "viewer": viewer,
            "recording": recording_config,
            "deterministic": deterministic,
        },
        _base_runtime("play"),
        {"format": "robolab-play-result-v1", "status": "pending"},
    )


def build_evaluate_command(
    *,
    task_id: str,
    resolved_config: Mapping[str, Any],
    scene: str,
    episodes: int,
    metrics: Sequence[str],
    thresholds: Mapping[str, float],
    evidence_dir: str | Path,
    toolchain: ToolchainIdentity,
    checkpoint: ArtifactRef | None = None,
    task_registry: TaskRegistry | None = None,
    robot_registry: RobotRegistry | None = None,
    task_version: str | None = None,
    robot_id: str | None = None,
    robot_version: str | None = None,
) -> JobCommand:
    if not scene:
        raise MotionContractError("scene 不能为空")
    if episodes < 1:
        raise MotionContractError("episodes 必须大于 0")
    metric_names = tuple(dict.fromkeys(metrics))
    if not metric_names or any(not name for name in metric_names):
        raise MotionContractError("metrics 不能为空")
    threshold_map = {str(key): float(value) for key, value in thresholds.items()}
    unknown = set(threshold_map) - set(metric_names)
    if unknown:
        raise MotionContractError(f"threshold 没有对应 metric: {sorted(unknown)}")
    task, robot, config = _resolve_refs(
        task_registry or default_task_registry(),
        robot_registry or default_robot_registry(),
        task_id,
        task_version,
        robot_id,
        robot_version,
        resolved_config,
    )
    result = {
        "format": "robolab-validation-result-v1",
        "status": "pending",
        "scene": scene,
        "episodes": episodes,
        "metrics": {
            name: {"value": None, "threshold": threshold_map.get(name), "passed": None}
            for name in metric_names
        },
        "evidence": {"directory": str(evidence_dir), "toolchain": toolchain.to_dict()},
    }
    return JobCommand(
        "evaluate",
        toolchain,
        task,
        robot,
        config,
        {
            "scene": scene,
            "episodes": episodes,
            "metrics": list(metric_names),
            "thresholds": threshold_map,
            "checkpoint": checkpoint.to_dict() if checkpoint else None,
            "evidenceDir": str(evidence_dir),
        },
        _base_runtime("evaluate"),
        result,
    )


def build_export_command(
    *,
    task_id: str,
    resolved_config: Mapping[str, Any],
    source: ArtifactRef,
    observation_schema: str,
    action_schema: str,
    control_frequency_hz: int,
    action_scale: float,
    joint_order: Sequence[str],
    metadata: Mapping[str, Any],
    toolchain: ToolchainIdentity,
    output: str | Path,
    task_registry: TaskRegistry | None = None,
    robot_registry: RobotRegistry | None = None,
    task_version: str | None = None,
    robot_id: str | None = None,
    robot_version: str | None = None,
) -> JobCommand:
    if not observation_schema or not action_schema:
        raise MotionContractError("observation_schema/action_schema 不能为空")
    if control_frequency_hz < 1:
        raise MotionContractError("control_frequency_hz 必须大于 0")
    if action_scale <= 0:
        raise MotionContractError("action_scale 必须大于 0")
    if not joint_order or len(set(joint_order)) != len(joint_order):
        raise MotionContractError("joint_order 必须是非空且无重复序列")
    task, robot, config = _resolve_refs(
        task_registry or default_task_registry(),
        robot_registry or default_robot_registry(),
        task_id,
        task_version,
        robot_id,
        robot_version,
        resolved_config,
    )
    artifact_metadata = {
        "observationSchema": observation_schema,
        "actionSchema": action_schema,
        "controlFrequencyHz": control_frequency_hz,
        "actionScale": action_scale,
        "jointOrder": list(joint_order),
        "deployment": {
            "controlFrequencyHz": control_frequency_hz,
            "actionScale": action_scale,
            "jointOrder": list(joint_order),
        },
        "source": source.to_dict(),
        "toolchain": toolchain.to_dict(),
        "metadata": _require_mapping(metadata, "export metadata"),
    }
    return JobCommand(
        "export",
        toolchain,
        task,
        robot,
        config,
        {
            "source": source.to_dict(),
            "output": str(output),
            "artifact": artifact_metadata,
        },
        _base_runtime("export"),
        {
            "format": "robolab-policy-artifact-v1",
            "status": "pending",
            "artifact": artifact_metadata,
        },
    )


def make_evaluation_result(
    command: JobCommand, values: Mapping[str, float]
) -> dict[str, Any]:
    """Create a machine-readable validation result from measured metric values."""

    if command.operation != "evaluate":
        raise MotionContractError("只有 evaluate JobCommand 可以生成 validation result")
    expected = command.result["metrics"]
    if set(values) != set(expected):
        raise MotionContractError("测量结果必须覆盖且仅覆盖声明的 metrics")
    metrics: dict[str, Any] = {}
    passed = True
    for name, value in values.items():
        threshold = expected[name]["threshold"]
        metric_passed = threshold is None or value <= threshold
        metrics[name] = {
            "value": float(value),
            "threshold": threshold,
            "passed": metric_passed,
        }
        passed = passed and metric_passed
    result = dict(command.result)
    result["status"] = "passed" if passed else "failed"
    result["metrics"] = metrics
    result["evidence"] = {
        **result["evidence"],
        "jobCommandToolchain": command.toolchain.to_dict(),
    }
    return result


def persist_motion_job(
    runs_root: str | Path, command: JobCommand, *, job_id: str | None = None
) -> JobPaths:
    """Persist a constructed command as a normal ``robolab-job-v1`` run."""

    document = command.to_dict()
    paths = create_job_run(
        runs_root,
        action=f"robolab.motion.{command.operation}",
        parameters=document,
        metadata={
            "toolchain": document["toolchain"],
            "resolvedConfig": document["resolvedConfig"],
            "commandProtocol": MOTION_PROTOCOL,
        },
        job_id=job_id,
    )
    (paths.run_dir / "job_command.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def write_artifact_metadata(path: str | Path, command: JobCommand) -> Path:
    """Write the export PolicyArtifact metadata sidecar without claiming export."""

    if command.operation != "export":
        raise MotionContractError(
            "只有 export JobCommand 可以写 PolicyArtifact metadata"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(command.result["artifact"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def run_mjlab_cpu_smoke() -> dict[str, Any]:
    """Run the existing MJLab cartpole construction as an R1 CPU smoke probe."""

    import warnings
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg

    with (
        warnings.catch_warnings(),
        redirect_stdout(StringIO()),
        redirect_stderr(StringIO()),
    ):
        warnings.simplefilter("ignore")
        env = ManagerBasedRlEnv(cartpole_balance_env_cfg(), device="cpu")
        try:
            return {"ok": True, "simTime": env.sim.data.time, "device": "cpu"}
        finally:
            env.close()


__all__ = [
    "MJLAB_UPSTREAM_REVISION",
    "MOTION_PROTOCOL",
    "TOOLCHAIN_ID",
    "ArtifactRef",
    "ConfigurationValidationError",
    "DuplicateRegistryEntryError",
    "JobCommand",
    "MotionContractError",
    "ResourceConfig",
    "RobotBinding",
    "RobotRegistry",
    "TaskDefinition",
    "TaskRegistry",
    "ToolchainIdentity",
    "UnknownRegistryEntryError",
    "artifact_ref_from_path",
    "build_evaluate_command",
    "build_export_command",
    "build_play_command",
    "build_train_command",
    "default_robot_registry",
    "default_task_registry",
    "make_evaluation_result",
    "persist_motion_job",
    "resolve_toolchain_identity",
    "run_mjlab_cpu_smoke",
    "write_artifact_metadata",
]
