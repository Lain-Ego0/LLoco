"""Stateful ONNX execution directly from a validated Policy Bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from lainloco.core import PolicyContract

from .policy_bundle import LoadedPolicyBundle, load_policy_bundle


class BundlePolicyRuntime:
  """Execute single-input, conditional-history, or recurrent bundle policies."""

  _HISTORY_MODES = {"cts_student", "dreamwaq", "ts_student"}

  def __init__(
    self,
    bundle: str | Path | LoadedPolicyBundle,
    *,
    batch_size: int = 1,
    providers: Sequence[str] | None = None,
  ) -> None:
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    self.bundle = (
      bundle if isinstance(bundle, LoadedPolicyBundle) else load_policy_bundle(bundle)
    )
    try:
      import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - Python 3.10 optional runtime
      raise RuntimeError(
        "onnxruntime is required to execute a Policy Bundle; use Python 3.11+ "
        "and install the lainloco cpu or cu128 extra"
      ) from exc
    self.contract: PolicyContract = self.bundle.contract
    self.physics_dt = self.bundle.robot.physics_dt
    self.batch_size = batch_size
    self.session = ort.InferenceSession(
      str(self.bundle.policy_path),
      providers=list(providers) if providers is not None else ["CPUExecutionProvider"],
    )
    self.input_names = tuple(item.name for item in self.session.get_inputs())
    self.output_names = tuple(item.name for item in self.session.get_outputs())
    metadata = self.session.get_modelmeta().custom_metadata_map
    self.mode = metadata.get("go2_policy_mode")
    self._history: np.ndarray | None = None
    if self._uses_history():
      self._history = np.zeros(
        (batch_size, self.contract.history_length, self.contract.observation_dim),
        dtype=np.float32,
      )
    self._hidden: np.ndarray | None = None
    self._cell: np.ndarray | None = None
    if self.contract.recurrent_state is not None:
      state = self.contract.recurrent_state
      shape = (state.layers, batch_size, state.hidden_width)
      self._hidden = np.zeros(shape, dtype=np.float32)
      self._cell = np.zeros(shape, dtype=np.float32)
    self._normalization_mean: np.ndarray | None = None
    self._normalization_scale: np.ndarray | None = None
    if self.bundle.normalization.mode == "external":
      with np.load(self.bundle.normalization_path, allow_pickle=False) as artifact:
        self._normalization_mean = np.asarray(artifact["mean"], dtype=np.float32)
        self._normalization_scale = np.asarray(artifact["scale"], dtype=np.float32)

  def _uses_history(self) -> bool:
    if self.contract.recurrent_state is not None or len(self.input_names) != 2:
      return False
    return (
      len(self.contract.conditional_fields) == 1
      and self.contract.conditional_fields[0].name == "history"
      and self.mode in self._HISTORY_MODES | {None}
    )

  def reset(self, env_ids: np.ndarray | Sequence[int] | None = None) -> None:
    """Clear history/recurrent state for all or selected simulation instances."""
    if env_ids is None:
      ids = np.arange(self.batch_size, dtype=np.int64)
    else:
      ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    if np.any(ids < 0) or np.any(ids >= self.batch_size):
      raise IndexError("env_ids contain an index outside the runtime batch")
    if self._history is not None:
      self._history[ids] = 0.0
    if self._hidden is not None and self._cell is not None:
      self._hidden[:, ids] = 0.0
      self._cell[:, ids] = 0.0

  def _observation(self, observations: Mapping[str, np.ndarray]) -> np.ndarray:
    fields: list[np.ndarray] = []
    for field in self.contract.observation_fields:
      if field.name not in observations:
        raise ValueError(f"Missing policy observation field {field.name!r}")
      value = np.asarray(observations[field.name], dtype=np.float32)
      if value.ndim != 2 or value.shape != (self.batch_size, field.width):
        raise ValueError(
          f"Observation {field.name!r} must have shape "
          f"{(self.batch_size, field.width)}, got {value.shape}"
        )
      fields.append(value)
    observation = np.concatenate(fields, axis=-1)
    if self._normalization_mean is not None and self._normalization_scale is not None:
      observation = (observation - self._normalization_mean) / self._normalization_scale
    return observation.astype(np.float32, copy=False)

  def _conditional(
    self, observations: Mapping[str, np.ndarray], observation: np.ndarray
  ) -> np.ndarray:
    if self._history is not None:
      conditional = self._history.reshape(self.batch_size, -1).copy()
      self._history[:, :-1] = self._history[:, 1:]
      self._history[:, -1] = observation
      return conditional
    if self.contract.conditional_fields:
      values: list[np.ndarray] = []
      for field in self.contract.conditional_fields:
        if field.name not in observations:
          raise ValueError(f"Missing conditional observation field {field.name!r}")
        value = np.asarray(observations[field.name], dtype=np.float32)
        if value.ndim != 2 or value.shape != (self.batch_size, field.width):
          raise ValueError(
            f"Conditional observation {field.name!r} must have shape "
            f"{(self.batch_size, field.width)}, got {value.shape}"
          )
        values.append(value)
      conditional = np.concatenate(values, axis=-1)
    elif "conditional" in observations:
      conditional = np.asarray(observations["conditional"], dtype=np.float32)
    else:
      raise ValueError(
        "This policy requires an explicit 'conditional' observation because its "
        "ONNX metadata does not identify a supported history/teacher mode"
      )
    expected_width = self.session.get_inputs()[1].shape[-1]
    if conditional.ndim != 2 or conditional.shape[0] != self.batch_size:
      raise ValueError("conditional observation must be a batched matrix")
    if isinstance(expected_width, int) and conditional.shape[-1] != expected_width:
      raise ValueError(
        f"conditional observation width must be {expected_width}, "
        f"got {conditional.shape[-1]}"
      )
    return conditional

  def act(self, observations: Mapping[str, np.ndarray]) -> np.ndarray:
    """Run one deterministic control tick and advance runtime state."""
    observation = self._observation(observations)
    if self.contract.recurrent_state is not None:
      assert self._hidden is not None and self._cell is not None
      actions, hidden, cell = self.session.run(
        ["actions", "he", "ce"],
        {"obs": observation, "h": self._hidden, "c": self._cell},
      )
      self._hidden = np.asarray(hidden, dtype=np.float32)
      self._cell = np.asarray(cell, dtype=np.float32)
    elif len(self.input_names) == 1:
      actions = self.session.run(["actions"], {self.input_names[0]: observation})[0]
    elif self.input_names == ("actor", "conditional"):
      conditional = self._conditional(observations, observation)
      actions = self.session.run(
        ["actions"], {"actor": observation, "conditional": conditional}
      )[0]
    else:  # The bundle validator should make this unreachable.
      raise RuntimeError(f"Unsupported ONNX input signature: {self.input_names!r}")
    result = np.asarray(actions, dtype=np.float32)
    expected_shape = (self.batch_size, self.contract.action_dim)
    if result.shape != expected_shape:
      raise RuntimeError(
        f"Policy returned action shape {result.shape}, expected {expected_shape}"
      )
    if not np.all(np.isfinite(result)):
      raise RuntimeError("Policy returned non-finite actions")
    return result

  @property
  def history(self) -> np.ndarray | None:
    """Read-only snapshot for diagnostics and reset verification."""
    return None if self._history is None else self._history.copy()

  @property
  def recurrent_state(self) -> tuple[np.ndarray, np.ndarray] | None:
    """Read-only recurrent state snapshot for diagnostics."""
    if self._hidden is None or self._cell is None:
      return None
    return self._hidden.copy(), self._cell.copy()
