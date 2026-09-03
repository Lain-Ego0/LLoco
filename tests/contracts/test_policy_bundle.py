"""Contract tests for the versioned deployment Policy Bundle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from lainloco.robots.unitree.go2.experiments import resolve_experiment
from lainloco.runtime import (
  PolicyBundleError,
  create_policy_bundle,
  load_policy_bundle,
  write_normalization_artifact,
)


def _write_policy(path: Path, observation_dim: int = 48, action_dim: int = 12) -> None:
  weights = numpy_helper.from_array(
    np.zeros((observation_dim, action_dim), dtype=np.float32), name="weights"
  )
  graph = helper.make_graph(
    [helper.make_node("MatMul", ["obs", "weights"], ["actions"])],
    "policy",
    [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, observation_dim])],
    [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [None, action_dim])],
    [weights],
  )
  model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 18)],
    producer_name="lainloco-test",
  )
  contract = resolve_experiment("go2/velocity-flat", "ppo").experiment.contract
  metadata = {
    "go2_policy_contract_version": contract.contract_version,
    "go2_actor_dim": str(observation_dim),
    "go2_action_dim": str(action_dim),
    "joint_names": ",".join(contract.joint_order),
    "action_scale": ",".join(str(value) for value in contract.action_scale),
  }
  for key, value in metadata.items():
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value
  onnx.save(model, path)


def _rewrite_manifest_digest(bundle: Path, artifact_name: str) -> None:
  manifest_path = bundle / "manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  artifact = bundle / manifest["artifacts"][artifact_name]["path"]
  manifest["artifacts"][artifact_name]["sha256"] = hashlib.sha256(
    artifact.read_bytes()
  ).hexdigest()
  manifest["artifacts"][artifact_name]["size"] = artifact.stat().st_size
  manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )


def _mutate_contract(
  bundle: Path,
  mutation: Callable[[dict[str, Any]], None],
  *,
  update_manifest_version: bool = False,
) -> None:
  path = bundle / "contract.json"
  contract = json.loads(path.read_text(encoding="utf-8"))
  mutation(contract)
  path.write_text(
    json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  _rewrite_manifest_digest(bundle, "contract")
  if update_manifest_version:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract_version"] = contract["contract_version"]
    manifest_path.write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@pytest.fixture
def policy_bundle(tmp_path: Path) -> Path:
  policy = tmp_path / "source.onnx"
  _write_policy(policy)
  experiment = resolve_experiment("go2/velocity-flat", "ppo").experiment
  bundle = tmp_path / "bundle"
  create_policy_bundle(bundle, policy, experiment)
  return bundle


def test_policy_bundle_round_trip_is_complete(policy_bundle: Path) -> None:
  experiment = resolve_experiment("go2/velocity-flat", "ppo").experiment
  loaded = load_policy_bundle(policy_bundle, expected_experiment=experiment)

  assert {path.name for path in policy_bundle.iterdir()} == {
    "policy.onnx",
    "contract.json",
    "normalization.npz",
    "robot.yaml",
    "task.yaml",
    "manifest.json",
  }
  assert loaded.robot == experiment.robot
  assert loaded.task == experiment.task
  assert loaded.contract == experiment.contract
  assert loaded.normalization.mode == "embedded"
  assert loaded.policy_path.name == "policy.onnx"


def test_policy_bundle_does_not_overwrite_existing_destination(
  policy_bundle: Path,
) -> None:
  experiment = resolve_experiment("go2/velocity-flat", "ppo").experiment
  with pytest.raises(FileExistsError, match="already exists"):
    create_policy_bundle(policy_bundle, policy_bundle / "policy.onnx", experiment)


def test_policy_bundle_rejects_integrity_failure(policy_bundle: Path) -> None:
  with (policy_bundle / "policy.onnx").open("ab") as stream:
    stream.write(b"tampered")
  with pytest.raises(PolicyBundleError, match="integrity check failed"):
    load_policy_bundle(policy_bundle)


def test_policy_bundle_rejects_wrong_joint_order(policy_bundle: Path) -> None:
  _mutate_contract(
    policy_bundle,
    lambda contract: contract["joint_order"].reverse(),
  )
  with pytest.raises(PolicyBundleError, match="joint_order mismatch"):
    load_policy_bundle(policy_bundle)


def test_policy_bundle_rejects_wrong_observation_dimension(policy_bundle: Path) -> None:
  def change_observation(contract: dict[str, Any]) -> None:
    contract["observation_fields"][0]["width"] += 1

  _mutate_contract(policy_bundle, change_observation)
  with pytest.raises(PolicyBundleError, match="observation dimension mismatch"):
    load_policy_bundle(policy_bundle)


def test_policy_bundle_rejects_wrong_control_dt(policy_bundle: Path) -> None:
  _mutate_contract(
    policy_bundle,
    lambda contract: contract.__setitem__("control_dt", 0.04),
  )
  with pytest.raises(PolicyBundleError, match="contract.control_dt mismatch"):
    load_policy_bundle(policy_bundle)


def test_policy_bundle_rejects_unsupported_contract_version(
  policy_bundle: Path,
) -> None:
  _mutate_contract(
    policy_bundle,
    lambda contract: contract.__setitem__("contract_version", "999"),
    update_manifest_version=True,
  )
  with pytest.raises(PolicyBundleError, match="Unsupported policy contract version"):
    load_policy_bundle(policy_bundle)


def test_policy_bundle_rejects_wrong_onnx_observation_width(tmp_path: Path) -> None:
  policy = tmp_path / "wrong.onnx"
  _write_policy(policy, observation_dim=47)
  experiment = resolve_experiment("go2/velocity-flat", "ppo").experiment
  with pytest.raises(PolicyBundleError, match="ONNX observation dimension mismatch"):
    create_policy_bundle(tmp_path / "bundle", policy, experiment)


def test_external_normalization_round_trip(tmp_path: Path) -> None:
  experiment = resolve_experiment("go2/velocity-flat", "ppo").experiment
  normalization = tmp_path / "external.npz"
  width = experiment.contract.observation_dim
  write_normalization_artifact(
    normalization,
    experiment.contract,
    mean=np.zeros(width, dtype=np.float32),
    scale=np.ones(width, dtype=np.float32),
  )
  policy = tmp_path / "source.onnx"
  _write_policy(policy)

  loaded = create_policy_bundle(
    tmp_path / "bundle",
    policy,
    experiment,
    normalization_path=normalization,
  )

  assert loaded.normalization.mode == "external"
