"""G1 coverage for the robot-neutral Policy Bundle runtime."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from lainloco.experiments import resolve_experiment
from lainloco.runtime import BundlePolicyRuntime, create_policy_bundle
from lainloco.workflows import run_mjlab_bundle


def _write_g1_policy(path: Path) -> None:
  starts = numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="starts")
  ends = numpy_helper.from_array(np.asarray([29], dtype=np.int64), name="ends")
  axes = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="axes")
  graph = helper.make_graph(
    [helper.make_node("Slice", ["obs", "starts", "ends", "axes"], ["actions"])],
    "g1-policy",
    [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, 99])],
    [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [None, 29])],
    [starts, ends, axes],
  )
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
  contract = resolve_experiment("g1/velocity-flat", "ppo").experiment.contract
  metadata = {
    "joint_names": ",".join(contract.joint_order),
    "action_scale": ",".join(format(value, ".17g") for value in contract.action_scale),
  }
  for key, value in metadata.items():
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value
  onnx.save(model, path)


def test_g1_bundle_validates_and_executes_variable_dimensions(tmp_path: Path) -> None:
  policy_path = tmp_path / "g1.onnx"
  _write_g1_policy(policy_path)
  experiment = resolve_experiment("g1/velocity-flat", "ppo").experiment
  bundle = create_policy_bundle(tmp_path / "bundle", policy_path, experiment)
  runtime = BundlePolicyRuntime(bundle, batch_size=2)

  observations = {"actor": np.ones((2, 99), dtype=np.float32)}
  actions = runtime.act(observations)

  assert bundle.robot.robot_id == "unitree/g1"
  assert bundle.contract.action_dim == 29
  np.testing.assert_array_equal(actions, np.ones((2, 29), dtype=np.float32))

  stats = run_mjlab_bundle(str(bundle.root), steps=2, num_envs=1, device="cpu")
  assert stats.control_steps == 2
  assert stats.policy_calls == 2
  assert stats.simulated_seconds == 0.04
