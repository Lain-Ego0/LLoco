"""Go2 deployment-side input adapters for conditional policies.

The current mjlab ONNX wrapper intentionally exposes two inputs (``actor`` and
``conditional``).  This module provides the small amount of state that a
sim-to-sim or Unitree loop must maintain between calls: the five-frame history
used by CTS/DreamWaQ/TS student policies and the explicit terrain/privileged
concatenation used by a TS teacher.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..contract import GO2_POLICY_CONTRACT_VERSION, Go2PolicyInputSpec


def go2_policy_contract_metadata(
  actor, *, recurrent: bool = False
) -> dict[str, str | float]:
  """Return versioned deployment metadata for a migrated conditional actor."""
  latent_kind = getattr(actor, "latent_kind", None)
  use_student = bool(getattr(actor, "use_student", False))
  if latent_kind == "cts":
    # Source CTS never deploys its privileged teacher path.
    mode = "cts_student"
    conditional_dim = Go2PolicyInputSpec().history_dim
  elif latent_kind == "dreamwaq":
    mode = "dreamwaq"
    conditional_dim = Go2PolicyInputSpec().history_dim
  elif latent_kind == "ts":
    mode = "ts_student" if use_student else "ts_teacher"
    conditional_dim = (
      Go2PolicyInputSpec().history_dim
      if use_student
      else Go2PolicyInputSpec().terrain_dim + Go2PolicyInputSpec().ts_privileged_dim
    )
  else:
    return {}
  if recurrent:
    if mode != "ts_student":
      return {}
    mode = "ts_student_recurrent"
  spec = Go2PolicyInputSpec()
  metadata: dict[str, str | float] = {
    "go2_policy_contract_version": GO2_POLICY_CONTRACT_VERSION,
    "go2_policy_mode": mode,
    "go2_actor_dim": float(spec.actor_dim),
    "go2_action_dim": float(spec.action_dim),
    "go2_history_order": "oldest_to_newest",
    "go2_history_reset": "zero",
    "go2_recurrent_state": "external" if recurrent else "none",
  }
  if not recurrent:
    metadata["go2_conditional_dim"] = float(conditional_dim)
  return metadata


class Go2HistoryBuffer:
  """Chronological fixed-length history with mjlab reset semantics."""

  def __init__(
    self,
    batch_size: int = 1,
    frame_dim: int = 45,
    history_length: int = 5,
    device: str | torch.device = "cpu",
  ) -> None:
    if batch_size <= 0 or frame_dim <= 0 or history_length <= 0:
      raise ValueError("batch_size, frame_dim and history_length must be positive")
    self.batch_size = batch_size
    self.frame_dim = frame_dim
    self.history_length = history_length
    self.device = torch.device(device)
    self._buffer = torch.zeros(
      (batch_size, history_length, frame_dim), device=self.device
    )
    self._initialized = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

  def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
    """Reset all or selected environments before their next observation."""
    if env_ids is None:
      ids = torch.arange(self.batch_size, device=self.device)
    else:
      ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
    if ids.numel() == 0:
      return
    self._buffer[ids] = 0.0
    self._initialized[ids] = False

  def push(self, frame: torch.Tensor) -> torch.Tensor:
    """Append a frame and return flattened oldest-to-newest history."""
    if frame.ndim != 2 or frame.shape != (self.batch_size, self.frame_dim):
      raise ValueError(
        f"Expected frame shape {(self.batch_size, self.frame_dim)}, got {tuple(frame.shape)}"
      )
    frame = frame.to(device=self.device, dtype=self._buffer.dtype)
    fresh = ~self._initialized
    if fresh.any():
      # The source Isaac-Gym deque is zero initialized.  Its first observation
      # update appends the current frame, leaving ``history_length - 1`` zero
      # frames before it; preserve that deployment contract instead of
      # duplicating the first frame across the whole history.
      self._buffer[fresh] = 0.0
      self._buffer[fresh, -1] = frame[fresh]
      self._initialized[fresh] = True
    existing = ~fresh
    if existing.any():
      self._buffer[existing, :-1] = self._buffer[existing, 1:].clone()
      self._buffer[existing, -1] = frame[existing]
    return self.flatten()

  def previous_then_push(self, frame: torch.Tensor) -> torch.Tensor:
    """Return the pre-update history, then append ``frame``.

    The source CTS/DreamWaQ/TS deployment loop feeds the history buffer to the
    policy before stepping the simulator; the frame used for that action is
    appended only for the next policy call.
    """
    previous = self.flatten().clone()
    self.push(frame)
    return previous

  def flatten(self) -> torch.Tensor:
    return self._buffer.reshape(self.batch_size, self.history_length * self.frame_dim)

  @property
  def value(self) -> torch.Tensor:
    return self._buffer


def _static_shape_dim(shape: list[object] | None, index: int) -> int | None:
  """Return a static ONNX dimension, or ``None`` for a symbolic dimension."""
  if shape is None or not shape:
    return None
  value = shape[index]
  return int(value) if isinstance(value, (int, np.integer)) else None


class Go2OnnxPolicy:
  """Lazy ONNX Runtime wrapper with shape checks for Go2 policy files."""

  def __init__(
    self,
    path: str | Path,
    providers: list[str] | None = None,
    expected_mode: str | None = None,
  ) -> None:
    try:
      import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional deployment dependency
      raise RuntimeError(
        "onnxruntime is required to run an exported Go2 policy"
      ) from exc
    session_providers = providers or ["CPUExecutionProvider"]
    self.session = ort.InferenceSession(str(path), providers=session_providers)
    metadata = self.session.get_modelmeta().custom_metadata_map
    self.contract_version = metadata.get("go2_policy_contract_version")
    self.mode = metadata.get("go2_policy_mode")
    if (
      self.contract_version is not None
      and self.contract_version != GO2_POLICY_CONTRACT_VERSION
    ):
      raise ValueError(
        f"Unsupported Go2 policy contract version {self.contract_version!r}; "
        f"expected {GO2_POLICY_CONTRACT_VERSION!r}"
      )
    if expected_mode is not None and self.mode != expected_mode:
      raise ValueError(f"Expected Go2 policy mode {expected_mode!r}, got {self.mode!r}")
    inputs = self.session.get_inputs()
    if [item.name for item in inputs] != ["actor", "conditional"]:
      raise ValueError(
        "Expected ONNX inputs ['actor', 'conditional'], "
        f"got {[item.name for item in inputs]}"
      )
    self.actor_dim = _static_shape_dim(inputs[0].shape, -1)
    self.conditional_dim = _static_shape_dim(inputs[1].shape, -1)
    if self.actor_dim is not None and self.actor_dim != Go2PolicyInputSpec.actor_dim:
      raise ValueError(
        f"Expected actor input width {Go2PolicyInputSpec.actor_dim}, got {self.actor_dim}"
      )
    expected_conditional_dims = {
      "cts_student": Go2PolicyInputSpec().history_dim,
      "dreamwaq": Go2PolicyInputSpec().history_dim,
      "ts_teacher": (
        Go2PolicyInputSpec().terrain_dim + Go2PolicyInputSpec().ts_privileged_dim
      ),
      "ts_student": Go2PolicyInputSpec().history_dim,
    }
    if self.mode in expected_conditional_dims and self.conditional_dim is not None:
      expected_dim = expected_conditional_dims[self.mode]
      if self.conditional_dim != expected_dim:
        raise ValueError(
          f"Go2 mode {self.mode!r} requires conditional width {expected_dim}, "
          f"got {self.conditional_dim}"
        )
    outputs = self.session.get_outputs()
    if [item.name for item in outputs] != ["actions"]:
      raise ValueError(
        f"Expected ONNX outputs ['actions'], got {[item.name for item in outputs]}"
      )
    self.action_dim = _static_shape_dim(outputs[0].shape, -1)
    if self.action_dim is not None and self.action_dim != Go2PolicyInputSpec.action_dim:
      raise ValueError(
        f"Expected action output width {Go2PolicyInputSpec.action_dim}, got {self.action_dim}"
      )

  def act(self, actor: np.ndarray, conditional: np.ndarray) -> np.ndarray:
    actor = np.asarray(actor, dtype=np.float32)
    conditional = np.asarray(conditional, dtype=np.float32)
    if (
      actor.ndim != 2 or conditional.ndim != 2 or actor.shape[0] != conditional.shape[0]
    ):
      raise ValueError(
        "actor and conditional must be batched arrays with equal batch size"
      )
    if self.actor_dim is not None and actor.shape[-1] != self.actor_dim:
      raise ValueError(f"actor must have shape [batch, {self.actor_dim}]")
    if (
      self.conditional_dim is not None and conditional.shape[-1] != self.conditional_dim
    ):
      raise ValueError(f"conditional must have shape [batch, {self.conditional_dim}]")
    output = np.asarray(
      self.session.run(["actions"], {"actor": actor, "conditional": conditional})[0],
      dtype=np.float32,
    )
    if output.ndim != 2 or output.shape[-1] != Go2PolicyInputSpec.action_dim:
      raise ValueError(
        f"ONNX actions must have shape [batch, {Go2PolicyInputSpec.action_dim}], "
        f"got {output.shape}"
      )
    return output


class Go2RecurrentOnnxPolicy:
  """ONNX wrapper for the source TS-Student recurrent deployment contract.

  The legacy exporter uses three inputs (``obs``, ``h``, ``c``) and returns
  actions together with the updated LSTM states.  Keeping state outside the
  ONNX session makes reset-on-episode-boundary explicit for sim-to-sim loops.
  """

  def __init__(
    self,
    path: str | Path,
    providers: list[str] | None = None,
    expected_mode: str | None = None,
  ) -> None:
    try:
      import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional deployment dependency
      raise RuntimeError(
        "onnxruntime is required to run an exported Go2 policy"
      ) from exc
    self.session = ort.InferenceSession(
      str(path), providers=providers or ["CPUExecutionProvider"]
    )
    metadata = self.session.get_modelmeta().custom_metadata_map
    self.contract_version = metadata.get("go2_policy_contract_version")
    self.mode = metadata.get("go2_policy_mode")
    if (
      self.contract_version is not None
      and self.contract_version != GO2_POLICY_CONTRACT_VERSION
    ):
      raise ValueError(
        f"Unsupported Go2 policy contract version {self.contract_version!r}; "
        f"expected {GO2_POLICY_CONTRACT_VERSION!r}"
      )
    if expected_mode is not None and self.mode != expected_mode:
      raise ValueError(f"Expected Go2 policy mode {expected_mode!r}, got {self.mode!r}")
    if self.mode is not None and self.mode != "ts_student_recurrent":
      raise ValueError(
        f"Recurrent Go2 policy requires mode 'ts_student_recurrent', got {self.mode!r}"
      )
    inputs = [item.name for item in self.session.get_inputs()]
    if inputs != ["obs", "h", "c"]:
      raise ValueError(
        f"Expected recurrent ONNX inputs ['obs', 'h', 'c'], got {inputs}"
      )
    input_specs = self.session.get_inputs()
    self.obs_dim = _static_shape_dim(input_specs[0].shape, -1)
    self.num_layers = _static_shape_dim(input_specs[1].shape, 0)
    self.hidden_dim = _static_shape_dim(input_specs[1].shape, -1)
    if self.obs_dim is not None and self.obs_dim != Go2PolicyInputSpec.actor_dim:
      raise ValueError(
        f"Expected recurrent obs width {Go2PolicyInputSpec.actor_dim}, got {self.obs_dim}"
      )
    if (
      self.num_layers is not None
      and self.num_layers != Go2PolicyInputSpec.ts_student_lstm_layers
    ):
      raise ValueError(
        f"Expected {Go2PolicyInputSpec.ts_student_lstm_layers} LSTM layers, got {self.num_layers}"
      )
    if (
      self.hidden_dim is not None
      and self.hidden_dim != Go2PolicyInputSpec.ts_student_hidden_dim
    ):
      raise ValueError(
        f"Expected recurrent hidden width {Go2PolicyInputSpec.ts_student_hidden_dim}, got {self.hidden_dim}"
      )
    outputs = [item.name for item in self.session.get_outputs()]
    if outputs[:3] != ["actions", "he", "ce"]:
      raise ValueError(
        f"Expected recurrent ONNX outputs ['actions', 'he', 'ce'], got {outputs}"
      )

  def act(
    self,
    obs: np.ndarray,
    hidden: np.ndarray,
    cell: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32)
    hidden = np.asarray(hidden, dtype=np.float32)
    cell = np.asarray(cell, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[-1] != Go2PolicyInputSpec.actor_dim:
      raise ValueError(f"obs must have shape [batch, {Go2PolicyInputSpec.actor_dim}]")
    if hidden.ndim != 3 or cell.shape != hidden.shape:
      raise ValueError(
        "hidden and cell must be matching [layers, batch, hidden] arrays"
      )
    if self.num_layers is not None and hidden.shape[0] != self.num_layers:
      raise ValueError(f"hidden/cell must have {self.num_layers} layers")
    if self.hidden_dim is not None and hidden.shape[2] != self.hidden_dim:
      raise ValueError(f"hidden/cell must have width {self.hidden_dim}")
    if hidden.shape[1] != obs.shape[0]:
      raise ValueError("hidden/cell batch dimension must match obs")
    actions, hidden_next, cell_next = self.session.run(
      ["actions", "he", "ce"], {"obs": obs, "h": hidden, "c": cell}
    )[:3]
    return (
      np.asarray(actions, dtype=np.float32),
      np.asarray(hidden_next, dtype=np.float32),
      np.asarray(cell_next, dtype=np.float32),
    )


class Go2RecurrentDeploymentAdapter:
  """Stateful adapter for the TS-Student recurrent ONNX policy.

  The ONNX session itself is intentionally stateless.  This adapter owns the
  three-layer LSTM hidden/cell tensors, clears selected environments on reset,
  and updates the state only after a successful policy call.  It therefore
  matches the source deployment loop while remaining usable from a batched
  sim-to-sim controller.
  """

  def __init__(
    self,
    policy: Go2RecurrentOnnxPolicy,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    num_layers: int = Go2PolicyInputSpec.ts_student_lstm_layers,
    hidden_dim: int = Go2PolicyInputSpec.ts_student_hidden_dim,
  ) -> None:
    if batch_size <= 0 or num_layers <= 0 or hidden_dim <= 0:
      raise ValueError("batch_size, num_layers and hidden_dim must be positive")
    self.policy = policy
    self.batch_size = batch_size
    self.device = torch.device(device)
    self._hidden = torch.zeros(
      num_layers, batch_size, hidden_dim, dtype=torch.float32, device=self.device
    )
    self._cell = torch.zeros_like(self._hidden)

  @property
  def hidden(self) -> torch.Tensor:
    return self._hidden

  @property
  def cell(self) -> torch.Tensor:
    return self._cell

  def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
    """Clear recurrent state for all or selected environments."""
    if env_ids is None:
      ids = torch.arange(self.batch_size, device=self.device)
    else:
      ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
    if ids.numel() == 0:
      return
    if torch.any(ids < 0) or torch.any(ids >= self.batch_size):
      raise IndexError("env_ids contain an index outside the adapter batch")
    self._hidden[:, ids] = 0.0
    self._cell[:, ids] = 0.0

  def act(self, obs: torch.Tensor) -> torch.Tensor:
    """Run one step and retain the returned LSTM state for the next call."""
    if obs.ndim != 2 or obs.shape != (self.batch_size, Go2PolicyInputSpec.actor_dim):
      raise ValueError(
        f"obs must have shape {(self.batch_size, Go2PolicyInputSpec.actor_dim)}, "
        f"got {tuple(obs.shape)}"
      )
    actions, hidden_next, cell_next = self.policy.act(
      obs.detach().cpu().numpy(),
      self._hidden.detach().cpu().numpy(),
      self._cell.detach().cpu().numpy(),
    )
    self._hidden = torch.from_numpy(hidden_next).to(self.device)
    self._cell = torch.from_numpy(cell_next).to(self.device)
    return torch.from_numpy(actions).to(self.device)


class Go2DeploymentAdapter:
  """Prepare conditional inputs for a selected exported policy variant."""

  _HISTORY_MODES = {"cts_student", "dreamwaq", "ts_student"}
  _TEACHER_MODES = {"cts_teacher", "ts_teacher"}

  def __init__(
    self,
    mode: str,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    policy: Go2OnnxPolicy | None = None,
  ) -> None:
    if mode not in self._HISTORY_MODES | self._TEACHER_MODES:
      raise ValueError(f"Unknown Go2 deployment mode: {mode}")
    self.mode = mode
    self.spec = Go2PolicyInputSpec()
    self.policy = policy
    if policy is not None and policy.mode is not None and policy.mode != mode:
      raise ValueError(
        f"Deployment adapter mode {mode!r} does not match ONNX policy mode "
        f"{policy.mode!r}"
      )
    self.history = (
      Go2HistoryBuffer(
        batch_size, self.spec.actor_dim, self.spec.history_length, device
      )
      if mode in self._HISTORY_MODES
      else None
    )

  def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
    if self.history is not None:
      self.history.reset(env_ids)

  def conditional(
    self,
    actor: torch.Tensor,
    terrain: torch.Tensor | None = None,
    privileged: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if actor.ndim != 2 or actor.shape[-1] != self.spec.actor_dim:
      raise ValueError(f"actor must have shape [batch, {self.spec.actor_dim}]")
    if self.history is not None:
      return self.history.previous_then_push(actor)
    if self.mode == "cts_teacher":
      if privileged is None:
        raise ValueError("cts_teacher requires privileged observations")
      if privileged.ndim != 2 or privileged.shape[-1] != self.spec.cts_privileged_dim:
        raise ValueError(
          f"cts_teacher privileged must have shape [batch, {self.spec.cts_privileged_dim}]"
        )
      return privileged
    if terrain is None or privileged is None:
      raise ValueError("ts_teacher requires terrain and privileged observations")
    if (
      terrain.ndim != 2
      or privileged.ndim != 2
      or terrain.shape[0] != privileged.shape[0]
    ):
      raise ValueError("terrain and privileged must be batched with equal batch size")
    if terrain.shape[-1] != self.spec.terrain_dim:
      raise ValueError(
        f"ts_teacher terrain must have shape [batch, {self.spec.terrain_dim}]"
      )
    if privileged.shape[-1] != self.spec.ts_privileged_dim:
      raise ValueError(
        f"ts_teacher privileged must have shape [batch, {self.spec.ts_privileged_dim}]"
      )
    return torch.cat((terrain, privileged), dim=-1)

  def act(
    self,
    actor: torch.Tensor,
    terrain: torch.Tensor | None = None,
    privileged: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if self.policy is None:
      raise RuntimeError("No ONNX policy was supplied to Go2DeploymentAdapter")
    conditional = self.conditional(actor, terrain, privileged)
    return torch.from_numpy(
      self.policy.act(actor.detach().cpu().numpy(), conditional.detach().cpu().numpy())
    ).to(device=actor.device)
