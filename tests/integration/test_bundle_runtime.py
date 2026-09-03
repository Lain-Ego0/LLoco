"""Continuous and stateful Policy Bundle runtime integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from lainloco.robots.unitree.go2.experiments import resolve_experiment
from lainloco.runtime import (
  BundlePolicyRuntime,
  SimToSimRuntime,
  SimulationStep,
  create_policy_bundle,
)


def _metadata(model: onnx.ModelProto, values: dict[str, str]) -> None:
  for key, value in values.items():
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value


def _write_history_policy(path: Path, conditional_dim: int = 225) -> None:
  starts = numpy_helper.from_array(np.asarray([213], dtype=np.int64), name="starts")
  ends = numpy_helper.from_array(np.asarray([225], dtype=np.int64), name="ends")
  axes = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="axes")
  graph = helper.make_graph(
    [helper.make_node("Slice", ["conditional", "starts", "ends", "axes"], ["actions"])],
    "history-policy",
    [
      helper.make_tensor_value_info("actor", TensorProto.FLOAT, [None, 45]),
      helper.make_tensor_value_info(
        "conditional", TensorProto.FLOAT, [None, conditional_dim]
      ),
    ],
    [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [None, 12])],
    [starts, ends, axes],
  )
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
  _metadata(
    model,
    {
      "go2_policy_contract_version": "1",
      "go2_policy_mode": "dreamwaq",
      "go2_actor_dim": "45",
      "go2_conditional_dim": "225",
      "go2_action_dim": "12",
    },
  )
  onnx.save(model, path)


def _write_recurrent_policy(path: Path) -> None:
  starts = numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="starts")
  ends = numpy_helper.from_array(np.asarray([12], dtype=np.int64), name="ends")
  axes = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="axes")
  one = numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name="one")
  graph = helper.make_graph(
    [
      helper.make_node("Slice", ["obs", "starts", "ends", "axes"], ["actions"]),
      helper.make_node("Add", ["h", "one"], ["he"]),
      helper.make_node("Add", ["c", "one"], ["ce"]),
    ],
    "recurrent-policy",
    [
      helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, 45]),
      helper.make_tensor_value_info("h", TensorProto.FLOAT, [3, None, 256]),
      helper.make_tensor_value_info("c", TensorProto.FLOAT, [3, None, 256]),
    ],
    [
      helper.make_tensor_value_info("actions", TensorProto.FLOAT, [None, 12]),
      helper.make_tensor_value_info("he", TensorProto.FLOAT, [3, None, 256]),
      helper.make_tensor_value_info("ce", TensorProto.FLOAT, [3, None, 256]),
    ],
    [starts, ends, axes, one],
  )
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
  _metadata(
    model,
    {
      "go2_policy_contract_version": "1",
      "go2_policy_mode": "ts_student_recurrent",
      "go2_actor_dim": "45",
      "go2_action_dim": "12",
    },
  )
  onnx.save(model, path)


def test_bundle_runtime_maintains_and_selectively_resets_history(
  tmp_path: Path,
) -> None:
  policy_path = tmp_path / "history.onnx"
  _write_history_policy(policy_path)
  experiment = resolve_experiment("go2/velocity-rough", "dreamwaq").experiment
  bundle = create_policy_bundle(tmp_path / "bundle", policy_path, experiment)
  runtime = BundlePolicyRuntime(bundle, batch_size=2)
  first = np.stack(
    (np.arange(45, dtype=np.float32), np.arange(45, dtype=np.float32) + 100)
  )

  np.testing.assert_array_equal(runtime.act({"actor": first}), np.zeros((2, 12)))
  np.testing.assert_array_equal(runtime.act({"actor": first + 1}), first[:, -12:])
  runtime.reset([1])
  result = runtime.act({"actor": first + 2})

  np.testing.assert_array_equal(result[0], first[0, -12:] + 1)
  np.testing.assert_array_equal(result[1], np.zeros(12))


def test_bundle_runtime_maintains_and_selectively_resets_recurrent_state(
  tmp_path: Path,
) -> None:
  policy_path = tmp_path / "recurrent.onnx"
  _write_recurrent_policy(policy_path)
  experiment = resolve_experiment("go2/velocity-rough", "ts-student").experiment
  bundle = create_policy_bundle(tmp_path / "bundle", policy_path, experiment)
  runtime = BundlePolicyRuntime(bundle, batch_size=2)
  observations = {"actor": np.ones((2, 45), dtype=np.float32)}

  runtime.act(observations)
  runtime.act(observations)
  state = runtime.recurrent_state
  assert state is not None
  hidden, cell = state
  np.testing.assert_array_equal(hidden, np.full((3, 2, 256), 2.0))
  np.testing.assert_array_equal(cell, np.full((3, 2, 256), 2.0))

  runtime.reset([0])
  state = runtime.recurrent_state
  assert state is not None
  hidden, cell = state
  np.testing.assert_array_equal(hidden[:, 0], 0.0)
  np.testing.assert_array_equal(cell[:, 0], 0.0)
  np.testing.assert_array_equal(hidden[:, 1], 2.0)
  np.testing.assert_array_equal(cell[:, 1], 2.0)


class _FakeSimulation:
  num_envs = 2
  physics_dt = 0.005
  control_dt = 0.02

  def __init__(self) -> None:
    self.step_count = 0

  def reset(self) -> dict[str, np.ndarray]:
    return {"actor": np.ones((2, 45), dtype=np.float32)}

  def step(self, actions: np.ndarray) -> SimulationStep:
    assert actions.shape == (2, 12)
    self.step_count += 1
    terminated = np.asarray([False, self.step_count == 2])
    return SimulationStep(
      {"actor": np.full((2, 45), self.step_count + 1, dtype=np.float32)},
      terminated,
      np.zeros(2, dtype=bool),
    )


def test_sim_to_sim_runs_one_policy_call_per_control_tick(tmp_path: Path) -> None:
  policy_path = tmp_path / "history.onnx"
  _write_history_policy(policy_path)
  experiment = resolve_experiment("go2/velocity-rough", "dreamwaq").experiment
  bundle = create_policy_bundle(tmp_path / "bundle", policy_path, experiment)
  policy = BundlePolicyRuntime(bundle, batch_size=2)

  stats = SimToSimRuntime(policy, _FakeSimulation()).run(steps=6)

  assert stats.control_steps == stats.policy_calls == 6
  assert stats.episode_resets == 1
  assert stats.simulated_seconds == 0.12


def test_sim_to_sim_rejects_wrong_control_frequency(tmp_path: Path) -> None:
  policy_path = tmp_path / "history.onnx"
  _write_history_policy(policy_path)
  experiment = resolve_experiment("go2/velocity-rough", "dreamwaq").experiment
  bundle = create_policy_bundle(tmp_path / "bundle", policy_path, experiment)
  policy = BundlePolicyRuntime(bundle, batch_size=2)
  simulation = _FakeSimulation()
  simulation.control_dt = 0.01

  try:
    SimToSimRuntime(policy, simulation)
  except ValueError as exc:
    assert "control_dt" in str(exc)
  else:
    raise AssertionError("wrong control frequency was accepted")


def test_bundle_rejects_wrong_conditional_history_width(tmp_path: Path) -> None:
  policy_path = tmp_path / "wrong-history.onnx"
  _write_history_policy(policy_path, conditional_dim=224)
  experiment = resolve_experiment("go2/velocity-rough", "dreamwaq").experiment

  try:
    create_policy_bundle(tmp_path / "bundle", policy_path, experiment)
  except ValueError as exc:
    assert "conditional dimension mismatch" in str(exc)
  else:
    raise AssertionError("wrong conditional history width was accepted")
