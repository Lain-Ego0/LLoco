"""Versioned, integrity-checked policy deployment bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
from onnx import TensorProto

from lainloco.core import (
  ExperimentSpec,
  ObservationField,
  PolicyContract,
  RecurrentStateSpec,
  RobotSpec,
  TaskSpec,
)

POLICY_BUNDLE_FORMAT_VERSION = "1"
SUPPORTED_POLICY_CONTRACT_VERSIONS = frozenset({"1"})
POLICY_FILENAME = "policy.onnx"
CONTRACT_FILENAME = "contract.json"
NORMALIZATION_FILENAME = "normalization.npz"
ROBOT_FILENAME = "robot.yaml"
TASK_FILENAME = "task.yaml"
MANIFEST_FILENAME = "manifest.json"

_ARTIFACT_FILENAMES = {
  "policy": POLICY_FILENAME,
  "contract": CONTRACT_FILENAME,
  "normalization": NORMALIZATION_FILENAME,
  "robot": ROBOT_FILENAME,
  "task": TASK_FILENAME,
}


class PolicyBundleError(ValueError):
  """A bundle is incomplete, corrupt, or incompatible with its consumer."""


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
  """Integrity information for one bundle artifact."""

  path: str
  sha256: str
  size: int


@dataclass(frozen=True, slots=True)
class PolicyBundleManifest:
  """The small, stable index read before any deployment artifact."""

  bundle_format_version: str
  robot_id: str
  task_id: str
  training_profile_id: str
  contract_version: str
  artifacts: dict[str, ArtifactRecord]


@dataclass(frozen=True, slots=True)
class NormalizationDescriptor:
  """Metadata stored beside normalization tensors in ``normalization.npz``."""

  schema_version: int
  mode: str
  contract: str
  observation_dim: int


@dataclass(frozen=True, slots=True)
class LoadedPolicyBundle:
  """Validated paths and contracts ready for a runtime to consume."""

  root: Path
  manifest: PolicyBundleManifest
  robot: RobotSpec
  task: TaskSpec
  contract: PolicyContract
  normalization: NormalizationDescriptor

  @property
  def policy_path(self) -> Path:
    return self.root / self.manifest.artifacts["policy"].path

  @property
  def normalization_path(self) -> Path:
    return self.root / self.manifest.artifacts["normalization"].path


def _write_json(path: Path, value: object) -> None:
  path.write_text(
    json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    + "\n",
    encoding="utf-8",
  )


def _read_json(path: Path) -> dict[str, object]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise PolicyBundleError(f"Cannot read JSON artifact {path.name}: {exc}") from exc
  if not isinstance(value, dict):
    raise PolicyBundleError(f"{path.name} must contain a JSON object")
  return cast(dict[str, object], value)


def _mapping(value: object, field: str) -> dict[str, object]:
  if not isinstance(value, Mapping):
    raise PolicyBundleError(f"{field} must be an object")
  if not all(isinstance(key, str) for key in value):
    raise PolicyBundleError(f"{field} keys must be strings")
  return cast(dict[str, object], dict(value))


def _sequence(value: object, field: str) -> list[object]:
  if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
    raise PolicyBundleError(f"{field} must be an array")
  return list(value)


def _text(value: object, field: str) -> str:
  if not isinstance(value, str) or not value:
    raise PolicyBundleError(f"{field} must be a non-empty string")
  return value


def _integer(value: object, field: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool):
    raise PolicyBundleError(f"{field} must be an integer")
  return value


def _number(value: object, field: str) -> float:
  if not isinstance(value, (int, float)) or isinstance(value, bool):
    raise PolicyBundleError(f"{field} must be a number")
  result = float(value)
  if not math.isfinite(result):
    raise PolicyBundleError(f"{field} must be finite")
  return result


def _strings(value: object, field: str) -> tuple[str, ...]:
  return tuple(_text(item, field) for item in _sequence(value, field))


def _numbers(value: object, field: str) -> tuple[float, ...]:
  return tuple(_number(item, field) for item in _sequence(value, field))


def _artifact_record(path: Path) -> ArtifactRecord:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return ArtifactRecord(
    path=path.name, sha256=digest.hexdigest(), size=path.stat().st_size
  )


def _manifest_to_dict(manifest: PolicyBundleManifest) -> dict[str, object]:
  return {
    "bundle_format_version": manifest.bundle_format_version,
    "robot_id": manifest.robot_id,
    "task_id": manifest.task_id,
    "training_profile_id": manifest.training_profile_id,
    "contract_version": manifest.contract_version,
    "artifacts": {
      name: asdict(record) for name, record in sorted(manifest.artifacts.items())
    },
  }


def _manifest_from_dict(data: dict[str, object]) -> PolicyBundleManifest:
  raw_artifacts = _mapping(data.get("artifacts"), "manifest.artifacts")
  artifacts: dict[str, ArtifactRecord] = {}
  for name, value in raw_artifacts.items():
    item = _mapping(value, f"manifest.artifacts.{name}")
    artifacts[name] = ArtifactRecord(
      path=_text(item.get("path"), f"manifest.artifacts.{name}.path"),
      sha256=_text(item.get("sha256"), f"manifest.artifacts.{name}.sha256"),
      size=_integer(item.get("size"), f"manifest.artifacts.{name}.size"),
    )
  return PolicyBundleManifest(
    bundle_format_version=_text(
      data.get("bundle_format_version"), "manifest.bundle_format_version"
    ),
    robot_id=_text(data.get("robot_id"), "manifest.robot_id"),
    task_id=_text(data.get("task_id"), "manifest.task_id"),
    training_profile_id=_text(
      data.get("training_profile_id"), "manifest.training_profile_id"
    ),
    contract_version=_text(data.get("contract_version"), "manifest.contract_version"),
    artifacts=artifacts,
  )


def _contract_from_dict(data: dict[str, object]) -> PolicyContract:
  raw_fields = _sequence(data.get("observation_fields"), "contract.observation_fields")
  fields = tuple(
    ObservationField(
      name=_text(
        _mapping(item, "contract.observation_fields[]").get("name"),
        "contract.observation_fields[].name",
      ),
      width=_integer(
        _mapping(item, "contract.observation_fields[]").get("width"),
        "contract.observation_fields[].width",
      ),
    )
    for item in raw_fields
  )
  raw_conditional_fields = _sequence(
    data.get("conditional_fields", ()), "contract.conditional_fields"
  )
  conditional_fields = tuple(
    ObservationField(
      name=_text(
        _mapping(item, "contract.conditional_fields[]").get("name"),
        "contract.conditional_fields[].name",
      ),
      width=_integer(
        _mapping(item, "contract.conditional_fields[]").get("width"),
        "contract.conditional_fields[].width",
      ),
    )
    for item in raw_conditional_fields
  )
  raw_state = data.get("recurrent_state")
  state = None
  if raw_state is not None:
    state_data = _mapping(raw_state, "contract.recurrent_state")
    state = RecurrentStateSpec(
      layers=_integer(state_data.get("layers"), "contract.recurrent_state.layers"),
      hidden_width=_integer(
        state_data.get("hidden_width"), "contract.recurrent_state.hidden_width"
      ),
    )
  try:
    return PolicyContract(
      contract_version=_text(data.get("contract_version"), "contract.contract_version"),
      robot_id=_text(data.get("robot_id"), "contract.robot_id"),
      task_id=_text(data.get("task_id"), "contract.task_id"),
      joint_order=_strings(data.get("joint_order"), "contract.joint_order"),
      action_dim=_integer(data.get("action_dim"), "contract.action_dim"),
      action_scale=_numbers(data.get("action_scale"), "contract.action_scale"),
      observation_fields=fields,
      history_length=_integer(data.get("history_length"), "contract.history_length"),
      history_order=_text(data.get("history_order"), "contract.history_order"),
      history_reset=_text(data.get("history_reset"), "contract.history_reset"),
      normalization=_text(data.get("normalization"), "contract.normalization"),
      recurrent_state=state,
      control_dt=_number(data.get("control_dt"), "contract.control_dt"),
      conditional_fields=conditional_fields,
    )
  except ValueError as exc:
    raise PolicyBundleError(f"Invalid policy contract: {exc}") from exc


def _robot_from_dict(data: dict[str, object]) -> RobotSpec:
  default_pose = tuple(
    (
      _text(_sequence(item, "robot.default_pose[]")[0], "robot.default_pose[].joint"),
      _number(_sequence(item, "robot.default_pose[]")[1], "robot.default_pose[].value"),
    )
    for item in _sequence(data.get("default_pose"), "robot.default_pose")
  )
  try:
    return RobotSpec(
      robot_id=_text(data.get("robot_id"), "robot.robot_id"),
      asset_factory=_text(data.get("asset_factory"), "robot.asset_factory"),
      joint_order=_strings(data.get("joint_order"), "robot.joint_order"),
      base_body=_text(data.get("base_body"), "robot.base_body"),
      foot_sites=_strings(data.get("foot_sites"), "robot.foot_sites"),
      collision_geoms=_strings(data.get("collision_geoms"), "robot.collision_geoms"),
      default_pose=default_pose,
      action_scale=_numbers(data.get("action_scale"), "robot.action_scale"),
      physics_dt=_number(data.get("physics_dt"), "robot.physics_dt"),
      control_dt=_number(data.get("control_dt"), "robot.control_dt"),
      hardware_joint_mapping=tuple(
        _integer(item, "robot.hardware_joint_mapping")
        for item in _sequence(
          data.get("hardware_joint_mapping"), "robot.hardware_joint_mapping"
        )
      ),
    )
  except (IndexError, ValueError) as exc:
    raise PolicyBundleError(f"Invalid robot specification: {exc}") from exc


def _task_from_dict(data: dict[str, object]) -> TaskSpec:
  try:
    return TaskSpec(
      task_id=_text(data.get("task_id"), "task.task_id"),
      robot_id=_text(data.get("robot_id"), "task.robot_id"),
      family=_text(data.get("family"), "task.family"),
      terrain_profile=_text(data.get("terrain_profile"), "task.terrain_profile"),
      command_profile=_text(data.get("command_profile"), "task.command_profile"),
      observation_profile=_text(
        data.get("observation_profile"), "task.observation_profile"
      ),
      reward_profile=_text(data.get("reward_profile"), "task.reward_profile"),
      termination_profile=_text(
        data.get("termination_profile"), "task.termination_profile"
      ),
      randomization_profile=_text(
        data.get("randomization_profile"), "task.randomization_profile"
      ),
      episode_length_s=_number(data.get("episode_length_s"), "task.episode_length_s"),
    )
  except ValueError as exc:
    raise PolicyBundleError(f"Invalid task specification: {exc}") from exc


def write_normalization_artifact(
  path: str | Path,
  contract: PolicyContract,
  *,
  mean: np.ndarray | None = None,
  scale: np.ndarray | None = None,
) -> None:
  """Write the canonical normalization artifact.

  With no arrays, normalization is already embedded in the ONNX graph. For an
  external transform, both arrays must be finite vectors matching the policy
  observation width, and every scale must be positive.
  """
  destination = Path(path)
  if (mean is None) != (scale is None):
    raise PolicyBundleError("External normalization requires both mean and scale")
  mode = "embedded"
  values: dict[str, Any] = {
    "schema_version": np.asarray(1, dtype=np.int64),
    "mode": np.asarray(mode),
    "contract": np.asarray(contract.normalization),
    "observation_dim": np.asarray(contract.observation_dim, dtype=np.int64),
  }
  if mean is not None and scale is not None:
    mode = "external"
    mean_array = np.asarray(mean, dtype=np.float32)
    scale_array = np.asarray(scale, dtype=np.float32)
    expected = (contract.observation_dim,)
    if mean_array.shape != expected or scale_array.shape != expected:
      raise PolicyBundleError(
        f"Normalization vectors must have shape {expected}, got "
        f"{mean_array.shape} and {scale_array.shape}"
      )
    if not np.all(np.isfinite(mean_array)) or not np.all(np.isfinite(scale_array)):
      raise PolicyBundleError("Normalization vectors must be finite")
    if np.any(scale_array <= 0):
      raise PolicyBundleError("Normalization scale must be positive")
    values["mode"] = np.asarray(mode)
    values["mean"] = mean_array
    values["scale"] = scale_array
  np.savez_compressed(destination, **values)


def _load_normalization(
  path: Path, contract: PolicyContract
) -> NormalizationDescriptor:
  try:
    with np.load(path, allow_pickle=False) as artifact:
      required = {"schema_version", "mode", "contract", "observation_dim"}
      missing = required - set(artifact.files)
      if missing:
        raise PolicyBundleError(
          f"normalization.npz is missing fields: {', '.join(sorted(missing))}"
        )
      schema_version = int(artifact["schema_version"].item())
      mode = str(artifact["mode"].item())
      contract_name = str(artifact["contract"].item())
      observation_dim = int(artifact["observation_dim"].item())
      if schema_version != 1:
        raise PolicyBundleError(
          f"Unsupported normalization schema version {schema_version!r}; expected 1"
        )
      if mode not in {"embedded", "external"}:
        raise PolicyBundleError(f"Unsupported normalization mode {mode!r}")
      if contract_name != contract.normalization:
        raise PolicyBundleError(
          "Normalization contract mismatch: "
          f"bundle={contract_name!r}, expected={contract.normalization!r}"
        )
      if observation_dim != contract.observation_dim:
        raise PolicyBundleError(
          "Normalization observation dimension mismatch: "
          f"bundle={observation_dim}, expected={contract.observation_dim}"
        )
      if mode == "external":
        if not {"mean", "scale"} <= set(artifact.files):
          raise PolicyBundleError("External normalization requires mean and scale")
        mean = np.asarray(artifact["mean"], dtype=np.float32)
        scale = np.asarray(artifact["scale"], dtype=np.float32)
        expected = (contract.observation_dim,)
        if mean.shape != expected or scale.shape != expected:
          raise PolicyBundleError(
            f"External normalization vectors must have shape {expected}"
          )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
          raise PolicyBundleError("External normalization vectors must be finite")
        if np.any(scale <= 0):
          raise PolicyBundleError("External normalization scale must be positive")
  except (OSError, ValueError, TypeError) as exc:
    if isinstance(exc, PolicyBundleError):
      raise
    raise PolicyBundleError(f"Cannot read normalization.npz: {exc}") from exc
  return NormalizationDescriptor(schema_version, mode, contract_name, observation_dim)


def _static_width(value_info: Any, index: int = -1) -> int | None:
  dimensions = value_info.type.tensor_type.shape.dim
  dimension = dimensions[index]
  return int(dimension.dim_value) if dimension.HasField("dim_value") else None


def _tensor_rank(value_info: Any) -> int:
  return len(value_info.type.tensor_type.shape.dim)


def _validate_onnx(path: Path, contract: PolicyContract) -> None:
  try:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
  except Exception as exc:
    raise PolicyBundleError(f"Invalid ONNX policy: {exc}") from exc
  if any(
    initializer.data_location == TensorProto.EXTERNAL
    for initializer in model.graph.initializer
  ):
    raise PolicyBundleError("ONNX policies with external tensor data are not supported")
  inputs = list(model.graph.input)
  outputs = list(model.graph.output)
  if not inputs or not outputs:
    raise PolicyBundleError("ONNX policy must have inputs and outputs")
  if _tensor_rank(inputs[0]) != 2 or _tensor_rank(outputs[0]) != 2:
    raise PolicyBundleError("ONNX policy observations and actions must be rank-2")
  if (
    inputs[0].type.tensor_type.elem_type != TensorProto.FLOAT
    or outputs[0].type.tensor_type.elem_type != TensorProto.FLOAT
  ):
    raise PolicyBundleError("ONNX policy observations and actions must be float32")
  recurrent = contract.recurrent_state
  if recurrent is None:
    if inputs[0].name not in {"actor", "obs"}:
      raise PolicyBundleError(
        f"ONNX primary input must be 'actor' or 'obs', got {inputs[0].name!r}"
      )
    if contract.conditional_fields:
      if [item.name for item in inputs] != ["actor", "conditional"]:
        raise PolicyBundleError(
          "A conditional policy contract requires ONNX inputs ['actor', 'conditional']"
        )
      conditional_width = _static_width(inputs[1])
      if (
        conditional_width is not None and conditional_width != contract.conditional_dim
      ):
        raise PolicyBundleError(
          "ONNX conditional dimension mismatch: "
          f"ONNX={conditional_width}, contract={contract.conditional_dim}"
        )
    elif len(inputs) != 1:
      raise PolicyBundleError(
        "A non-conditional policy contract requires exactly one ONNX input"
      )
  else:
    if [item.name for item in inputs] != ["obs", "h", "c"]:
      raise PolicyBundleError("Recurrent ONNX inputs must be ['obs', 'h', 'c']")
    if [item.name for item in outputs[:3]] != ["actions", "he", "ce"]:
      raise PolicyBundleError(
        "Recurrent ONNX outputs must start with ['actions', 'he', 'ce']"
      )
    for name, value_info in zip(
      ("h", "c", "he", "ce"),
      (inputs[1], inputs[2], outputs[1], outputs[2]),
      strict=True,
    ):
      if _tensor_rank(value_info) != 3:
        raise PolicyBundleError(f"Recurrent {name} must be rank-3")
      if value_info.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise PolicyBundleError(f"Recurrent {name} must be float32")
      layers = _static_width(value_info, 0)
      hidden_width = _static_width(value_info)
      if layers is not None and layers != recurrent.layers:
        raise PolicyBundleError(
          f"Recurrent {name} layer mismatch: ONNX={layers}, contract={recurrent.layers}"
        )
      if hidden_width is not None and hidden_width != recurrent.hidden_width:
        raise PolicyBundleError(
          f"Recurrent {name} hidden width mismatch: ONNX={hidden_width}, "
          f"contract={recurrent.hidden_width}"
        )
  input_width = _static_width(inputs[0])
  if input_width is not None and input_width != contract.observation_dim:
    raise PolicyBundleError(
      "ONNX observation dimension mismatch: "
      f"ONNX={input_width}, contract={contract.observation_dim}"
    )
  if outputs[0].name != "actions":
    raise PolicyBundleError(
      f"ONNX first output must be 'actions', got {outputs[0].name!r}"
    )
  output_width = _static_width(outputs[0])
  if output_width is not None and output_width != contract.action_dim:
    raise PolicyBundleError(
      f"ONNX action dimension mismatch: ONNX={output_width}, contract={contract.action_dim}"
    )
  metadata = {item.key: item.value for item in model.metadata_props}
  version = metadata.get("go2_policy_contract_version")
  if version is not None and version != contract.contract_version:
    raise PolicyBundleError(
      f"ONNX contract version mismatch: ONNX={version!r}, contract={contract.contract_version!r}"
    )
  actor_dim = metadata.get("go2_actor_dim")
  if actor_dim is not None and float(actor_dim) != contract.observation_dim:
    raise PolicyBundleError(
      f"ONNX actor dimension mismatch: ONNX={actor_dim}, contract={contract.observation_dim}"
    )
  action_dim = metadata.get("go2_action_dim")
  if action_dim is not None and float(action_dim) != contract.action_dim:
    raise PolicyBundleError(
      f"ONNX action dimension mismatch: ONNX={action_dim}, contract={contract.action_dim}"
    )
  conditional_dim = metadata.get("go2_conditional_dim")
  if conditional_dim is not None:
    if len(inputs) != 2 or inputs[1].name != "conditional":
      raise PolicyBundleError(
        "ONNX declares go2_conditional_dim without a conditional input"
      )
    graph_conditional_dim = _static_width(inputs[1])
    if (
      graph_conditional_dim is not None
      and float(conditional_dim) != graph_conditional_dim
    ):
      raise PolicyBundleError(
        "ONNX conditional dimension metadata does not match the graph: "
        f"metadata={conditional_dim}, graph={graph_conditional_dim}"
      )
  mode = metadata.get("go2_policy_mode")
  if mode in {"cts_student", "dreamwaq", "ts_student"}:
    if [item.name for item in inputs] != ["actor", "conditional"]:
      raise PolicyBundleError(
        f"ONNX mode {mode!r} requires inputs ['actor', 'conditional']"
      )
    expected_history_dim = contract.observation_dim * contract.history_length
    graph_history_dim = _static_width(inputs[1]) if len(inputs) == 2 else None
    if graph_history_dim is not None and graph_history_dim != expected_history_dim:
      raise PolicyBundleError(
        f"ONNX history dimension mismatch: ONNX={graph_history_dim}, "
        f"contract={expected_history_dim}"
      )
  history_order = metadata.get("go2_history_order")
  if history_order is not None and history_order != contract.history_order:
    raise PolicyBundleError(
      f"ONNX history order mismatch: ONNX={history_order!r}, "
      f"contract={contract.history_order!r}"
    )
  history_reset = metadata.get("go2_history_reset")
  if history_reset is not None and history_reset != contract.history_reset:
    raise PolicyBundleError(
      f"ONNX history reset mismatch: ONNX={history_reset!r}, "
      f"contract={contract.history_reset!r}"
    )
  joint_names = metadata.get("joint_names")
  if joint_names is not None and tuple(joint_names.split(",")) != contract.joint_order:
    raise PolicyBundleError("ONNX joint names/order do not match the policy contract")
  raw_scale = metadata.get("action_scale")
  if raw_scale is not None:
    try:
      onnx_scale = tuple(float(value) for value in raw_scale.split(","))
    except ValueError as exc:
      raise PolicyBundleError("ONNX action_scale metadata is invalid") from exc
    if len(onnx_scale) != len(contract.action_scale) or any(
      not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
      for actual, expected in zip(onnx_scale, contract.action_scale, strict=True)
    ):
      raise PolicyBundleError("ONNX action scale does not match the policy contract")


def _require_equal(field: str, actual: object, expected: object) -> None:
  if actual != expected:
    raise PolicyBundleError(
      f"{field} mismatch: bundle={actual!r}, expected={expected!r}"
    )


def _validate_semantics(
  manifest: PolicyBundleManifest,
  robot: RobotSpec,
  task: TaskSpec,
  contract: PolicyContract,
  expected_experiment: ExperimentSpec | None,
) -> None:
  _require_equal("manifest.robot_id", manifest.robot_id, robot.robot_id)
  _require_equal("manifest.task_id", manifest.task_id, task.task_id)
  _require_equal(
    "manifest.contract_version", manifest.contract_version, contract.contract_version
  )
  if contract.contract_version not in SUPPORTED_POLICY_CONTRACT_VERSIONS:
    raise PolicyBundleError(
      f"Unsupported policy contract version {contract.contract_version!r}; expected one of "
      f"{sorted(SUPPORTED_POLICY_CONTRACT_VERSIONS)!r}"
    )
  _require_equal("task.robot_id", task.robot_id, robot.robot_id)
  _require_equal("contract.robot_id", contract.robot_id, robot.robot_id)
  _require_equal("contract.task_id", contract.task_id, task.task_id)
  _require_equal("contract.joint_order", contract.joint_order, robot.joint_order)
  _require_equal("contract.action_dim", contract.action_dim, robot.action_dim)
  _require_equal("contract.action_scale", contract.action_scale, robot.action_scale)
  _require_equal("contract.control_dt", contract.control_dt, robot.control_dt)
  if expected_experiment is None:
    return
  _require_equal("robot", robot, expected_experiment.robot)
  _require_equal("task", task, expected_experiment.task)
  _require_equal("contract", contract, expected_experiment.contract)
  _require_equal(
    "training_profile_id",
    manifest.training_profile_id,
    expected_experiment.training.profile_id,
  )


def create_policy_bundle(
  destination: str | Path,
  policy_path: str | Path,
  experiment: ExperimentSpec,
  *,
  normalization_path: str | Path | None = None,
) -> LoadedPolicyBundle:
  """Create a complete bundle without overwriting an existing destination."""
  destination = Path(destination).resolve()
  source_policy = Path(policy_path).resolve()
  if destination.exists():
    raise FileExistsError(f"Policy bundle destination already exists: {destination}")
  if not source_policy.is_file():
    raise FileNotFoundError(f"ONNX policy does not exist: {source_policy}")
  _validate_onnx(source_policy, experiment.contract)
  source_normalization = (
    Path(normalization_path).resolve() if normalization_path is not None else None
  )
  if source_normalization is not None:
    if not source_normalization.is_file():
      raise FileNotFoundError(
        f"Normalization artifact does not exist: {source_normalization}"
      )
    _load_normalization(source_normalization, experiment.contract)
  destination.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(
    tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
  )
  try:
    shutil.copy2(source_policy, staging / POLICY_FILENAME)
    _write_json(staging / CONTRACT_FILENAME, asdict(experiment.contract))
    # JSON is a strict YAML 1.2 subset, so these files remain dependency-light
    # while retaining the human-readable deployment layout in the architecture.
    _write_json(staging / ROBOT_FILENAME, asdict(experiment.robot))
    _write_json(staging / TASK_FILENAME, asdict(experiment.task))
    if source_normalization is None:
      write_normalization_artifact(
        staging / NORMALIZATION_FILENAME, experiment.contract
      )
    else:
      shutil.copy2(source_normalization, staging / NORMALIZATION_FILENAME)
    records = {
      name: _artifact_record(staging / filename)
      for name, filename in _ARTIFACT_FILENAMES.items()
    }
    manifest = PolicyBundleManifest(
      bundle_format_version=POLICY_BUNDLE_FORMAT_VERSION,
      robot_id=experiment.robot.robot_id,
      task_id=experiment.task.task_id,
      training_profile_id=experiment.training.profile_id,
      contract_version=experiment.contract.contract_version,
      artifacts=records,
    )
    _write_json(staging / MANIFEST_FILENAME, _manifest_to_dict(manifest))
    os.rename(staging, destination)
  except Exception:
    if staging.exists():
      shutil.rmtree(staging)
    raise
  return load_policy_bundle(destination, expected_experiment=experiment)


def load_policy_bundle(
  path: str | Path,
  *,
  expected_experiment: ExperimentSpec | None = None,
) -> LoadedPolicyBundle:
  """Load a bundle only after integrity, schema, and semantic validation."""
  root = Path(path).resolve()
  if not root.is_dir():
    raise PolicyBundleError(f"Policy bundle is not a directory: {root}")
  manifest = _manifest_from_dict(_read_json(root / MANIFEST_FILENAME))
  if manifest.bundle_format_version != POLICY_BUNDLE_FORMAT_VERSION:
    raise PolicyBundleError(
      f"Unsupported policy bundle format {manifest.bundle_format_version!r}; "
      f"expected {POLICY_BUNDLE_FORMAT_VERSION!r}"
    )
  if set(manifest.artifacts) != set(_ARTIFACT_FILENAMES):
    raise PolicyBundleError(
      "Manifest artifacts must be exactly: " + ", ".join(sorted(_ARTIFACT_FILENAMES))
    )
  for name, expected_filename in _ARTIFACT_FILENAMES.items():
    record = manifest.artifacts[name]
    if record.path != expected_filename or Path(record.path).name != record.path:
      raise PolicyBundleError(
        f"Artifact {name!r} must use canonical path {expected_filename!r}"
      )
    artifact_path = root / record.path
    if artifact_path.is_symlink():
      raise PolicyBundleError(
        f"Bundle artifacts cannot be symbolic links: {record.path}"
      )
    if not artifact_path.is_file():
      raise PolicyBundleError(f"Bundle artifact is missing: {record.path}")
    actual = _artifact_record(artifact_path)
    if actual.size != record.size or actual.sha256 != record.sha256:
      raise PolicyBundleError(f"Bundle artifact integrity check failed: {record.path}")
  contract = _contract_from_dict(_read_json(root / CONTRACT_FILENAME))
  robot = _robot_from_dict(_read_json(root / ROBOT_FILENAME))
  task = _task_from_dict(_read_json(root / TASK_FILENAME))
  _validate_semantics(manifest, robot, task, contract, expected_experiment)
  normalization = _load_normalization(root / NORMALIZATION_FILENAME, contract)
  _validate_onnx(root / POLICY_FILENAME, contract)
  return LoadedPolicyBundle(root, manifest, robot, task, contract, normalization)
