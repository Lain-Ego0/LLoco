"""Synchronous sim-to-sim control loop for bundle-backed policies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from lainloco.core import PolicyContract


@dataclass(frozen=True, slots=True)
class SimulationStep:
  """One vectorized simulator transition."""

  observations: Mapping[str, np.ndarray]
  terminated: np.ndarray
  truncated: np.ndarray


@dataclass(frozen=True, slots=True)
class SimToSimStats:
  """Finite-loop acceptance counters."""

  control_steps: int
  policy_calls: int
  episode_resets: int
  simulated_seconds: float


class PolicyRuntime(Protocol):
  """Minimal stateful policy interface consumed by the loop."""

  contract: PolicyContract
  batch_size: int
  physics_dt: float

  def act(self, observations: Mapping[str, np.ndarray]) -> np.ndarray: ...

  def reset(self, env_ids: np.ndarray | Sequence[int] | None = None) -> None: ...


class SimulationBackend(Protocol):
  """Minimal synchronous vector-simulation boundary."""

  num_envs: int
  physics_dt: float
  control_dt: float

  def reset(self) -> Mapping[str, np.ndarray]: ...

  def step(self, actions: np.ndarray) -> SimulationStep: ...


def _numpy_observations(
  observations: Mapping[str, torch.Tensor],
) -> dict[str, np.ndarray]:
  return {
    name: tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    for name, tensor in observations.items()
  }


class MjlabSimulationBackend:
  """Adapter that keeps mjlab details outside the policy runtime."""

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self.env = env
    self.num_envs = env.num_envs
    self.physics_dt = env.physics_dt
    self.control_dt = env.step_dt

  def reset(self) -> Mapping[str, np.ndarray]:
    observations, _ = self.env.reset()
    return _numpy_observations(cast(Mapping[str, torch.Tensor], observations))

  def step(self, actions: np.ndarray) -> SimulationStep:
    action_tensor = torch.as_tensor(
      actions, device=self.env.device, dtype=torch.float32
    )
    observations, _reward, terminated, truncated, _info = self.env.step(action_tensor)
    return SimulationStep(
      _numpy_observations(cast(Mapping[str, torch.Tensor], observations)),
      terminated.detach().cpu().numpy().astype(bool, copy=False),
      truncated.detach().cpu().numpy().astype(bool, copy=False),
    )

  def close(self) -> None:
    self.env.close()


class SimToSimRuntime:
  """Advance exactly one policy decision per simulator control tick."""

  def __init__(self, policy: PolicyRuntime, backend: SimulationBackend) -> None:
    if policy.batch_size != backend.num_envs:
      raise ValueError(
        f"Policy batch size {policy.batch_size} does not match "
        f"simulation environments {backend.num_envs}"
      )
    if not math.isclose(
      backend.control_dt, policy.contract.control_dt, rel_tol=0.0, abs_tol=1e-12
    ):
      raise ValueError(
        f"Simulation control_dt {backend.control_dt} does not match "
        f"policy contract {policy.contract.control_dt}"
      )
    if not math.isclose(
      backend.physics_dt, policy.physics_dt, rel_tol=0.0, abs_tol=1e-12
    ):
      raise ValueError(
        f"Simulation physics_dt {backend.physics_dt} does not match "
        f"bundle robot {policy.physics_dt}"
      )
    if backend.physics_dt <= 0:
      raise ValueError("Simulation physics_dt must be positive")
    ratio = backend.control_dt / backend.physics_dt
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
      raise ValueError("Simulation control_dt must be an integer physics_dt multiple")
    self.policy = policy
    self.backend = backend

  def run(self, steps: int) -> SimToSimStats:
    """Run a finite continuous loop, resetting policy state at episode boundaries."""
    if steps < 0:
      raise ValueError("steps cannot be negative")
    observations = self.backend.reset()
    self.policy.reset()
    resets = 0
    for _ in range(steps):
      actions = self.policy.act(observations)
      transition = self.backend.step(actions)
      done = np.logical_or(transition.terminated, transition.truncated).reshape(-1)
      if done.shape != (self.backend.num_envs,):
        raise RuntimeError(
          f"Simulator done flags must have shape {(self.backend.num_envs,)}"
        )
      reset_ids = np.flatnonzero(done)
      if reset_ids.size:
        self.policy.reset(reset_ids)
        resets += int(reset_ids.size)
      observations = transition.observations
    return SimToSimStats(
      control_steps=steps,
      policy_calls=steps,
      episode_resets=resets,
      simulated_seconds=steps * self.backend.control_dt,
    )
