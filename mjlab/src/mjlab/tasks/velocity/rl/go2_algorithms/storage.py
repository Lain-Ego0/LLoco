"""Rollout storage shared by the migrated Go2 custom algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Go2Transition:
  obs: torch.Tensor
  critic_obs: torch.Tensor
  actions: torch.Tensor
  rewards: torch.Tensor
  dones: torch.Tensor
  values: torch.Tensor
  log_prob: torch.Tensor
  history: torch.Tensor | None = None
  privileged: torch.Tensor | None = None
  terrain: torch.Tensor | None = None


class Go2RunningNormalizer:
  """Torch equivalent of the source AMP running mean/std normalizer.

  The Isaac Gym fork updates a ``RunningMeanStd`` object with both policy and
  expert AMP states and clips normalized values to ``[-10, 10]``.  Keeping the
  small implementation here makes the current RSL-RL algorithm independent of
  NumPy and lets the statistics travel with a checkpoint.
  """

  def __init__(
    self,
    state_dim: int,
    device: str | torch.device = "cpu",
    epsilon: float = 1e-4,
    clip_obs: float = 10.0,
  ) -> None:
    if state_dim <= 0:
      raise ValueError("state_dim must be positive")
    self.state_dim = state_dim
    self.device = torch.device(device)
    self.epsilon = float(epsilon)
    self.clip_obs = float(clip_obs)
    self.mean = torch.zeros(state_dim, device=self.device)
    self.var = torch.ones(state_dim, device=self.device)
    self.count = torch.tensor(self.epsilon, device=self.device)

  @torch.no_grad()
  def update(self, values: torch.Tensor) -> None:
    """Update moments from a ``[batch, state_dim]`` tensor."""
    if values.ndim != 2 or values.shape[-1] != self.state_dim:
      raise ValueError("normalizer values must have shape [batch, state_dim]")
    if values.shape[0] == 0:
      return
    values = values.detach().to(device=self.device, dtype=torch.float32)
    batch_mean = values.mean(dim=0)
    batch_var = values.var(dim=0, unbiased=False)
    batch_count = torch.tensor(float(values.shape[0]), device=self.device)
    delta = batch_mean - self.mean
    total = self.count + batch_count
    mean = self.mean + delta * batch_count / total
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + delta.square() * self.count * batch_count / total
    self.mean.copy_(mean)
    self.var.copy_(torch.clamp(m_2 / total, min=0.0))
    self.count.copy_(total)

  def normalize(self, values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != self.state_dim:
      raise ValueError("normalizer values must end in state_dim")
    mean = self.mean.to(device=values.device, dtype=values.dtype)
    var = self.var.to(device=values.device, dtype=values.dtype)
    normalized = (values - mean) / torch.sqrt(var + self.epsilon)
    return normalized.clamp(-self.clip_obs, self.clip_obs)

  def state_dict(self) -> dict[str, torch.Tensor | float]:
    return {
      "mean": self.mean,
      "var": self.var,
      "count": self.count,
      "epsilon": self.epsilon,
      "clip_obs": self.clip_obs,
    }

  def load_state_dict(self, state: dict[str, torch.Tensor | float]) -> None:
    self.mean.copy_(state["mean"])
    self.var.copy_(state["var"])
    self.count.copy_(state["count"])
    self.epsilon = float(state.get("epsilon", self.epsilon))
    self.clip_obs = float(state.get("clip_obs", self.clip_obs))


class Go2RolloutStorage:
  """Fixed-size tensor storage with GAE and shuffled minibatches."""

  def __init__(self, num_steps: int, num_envs: int, device: str = "cpu") -> None:
    self.num_steps = num_steps
    self.num_envs = num_envs
    self.device = device
    self._items: list[Go2Transition] = []
    self.returns: torch.Tensor | None = None
    self.advantages: torch.Tensor | None = None

  def add(self, transition: Go2Transition) -> None:
    if len(self._items) >= self.num_steps:
      raise RuntimeError("Go2 rollout storage is full")
    self._items.append(transition)

  def compute_returns(self, last_value: torch.Tensor, gamma: float = 0.99, lam: float = 0.95) -> None:
    if len(self._items) != self.num_steps:
      raise RuntimeError("compute_returns requires a complete rollout")
    values = torch.stack([item.values for item in self._items])
    rewards = torch.stack([item.rewards for item in self._items])
    dones = torch.stack([item.dones for item in self._items]).to(values.dtype)
    returns = torch.zeros_like(values)
    advantage = torch.zeros_like(last_value)
    for step in reversed(range(self.num_steps)):
      next_value = last_value if step == self.num_steps - 1 else values[step + 1]
      not_done = 1.0 - dones[step]
      delta = rewards[step] + gamma * next_value * not_done - values[step]
      advantage = delta + gamma * lam * not_done * advantage
      returns[step] = advantage + values[step]
    self.returns = returns
    self.advantages = (returns - values - (returns - values).mean()) / ((returns - values).std() + 1e-8)

  def batches(self, batch_size: int):
    if self.returns is None or self.advantages is None:
      raise RuntimeError("Call compute_returns before requesting batches")
    indices = torch.randperm(self.num_steps * self.num_envs, device=self.device)
    flat = [torch.stack([getattr(item, field) for item in self._items]).flatten(0, 1)
            for field in ("obs", "critic_obs", "actions", "log_prob")]
    for start in range(0, len(indices), batch_size):
      idx = indices[start : start + batch_size]
      yield tuple(t[idx] for t in flat) + (self.returns.flatten(0, 1)[idx], self.advantages.flatten(0, 1)[idx])

  def clear(self) -> None:
    self._items.clear()
    self.returns = None
    self.advantages = None


class Go2AmpReplayBuffer:
  """Circular replay buffer for policy AMP transitions.

  The legacy AMP algorithms retain policy ``(state, next_state)`` pairs across
  PPO iterations.  Keeping this small buffer independent from rollout storage
  preserves that behavior without coupling the current RSL-RL storage API to
  the source fork.
  """

  def __init__(
    self,
    state_dim: int,
    capacity: int = 100_000,
    device: str | torch.device = "cpu",
  ) -> None:
    if state_dim <= 0 or capacity <= 0:
      raise ValueError("state_dim and capacity must be positive")
    self.state_dim = state_dim
    self.capacity = capacity
    self.device = torch.device(device)
    self.states = torch.zeros((capacity, state_dim), device=self.device)
    self.next_states = torch.zeros_like(self.states)
    self._cursor = 0
    self._size = 0

  @property
  def size(self) -> int:
    return self._size

  def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
    if states.ndim != 2 or next_states.shape != states.shape or states.shape[-1] != self.state_dim:
      raise ValueError("AMP replay states must have shape [batch, state_dim]")
    states = states.detach().to(self.device)
    next_states = next_states.detach().to(self.device)
    count = states.shape[0]
    if count == 0:
      return
    if count >= self.capacity:
      states = states[-self.capacity :]
      next_states = next_states[-self.capacity :]
      count = self.capacity
    indices = (torch.arange(count, device=self.device) + self._cursor) % self.capacity
    self.states[indices] = states
    self.next_states[indices] = next_states
    self._cursor = (self._cursor + count) % self.capacity
    self._size = min(self.capacity, self._size + count)

  def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if self._size == 0:
      raise RuntimeError("Cannot sample an empty AMP replay buffer")
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    indices = torch.randint(0, self._size, (batch_size,), device=self.device)
    return self.states[indices], self.next_states[indices]

  def state_dict(self) -> dict[str, torch.Tensor | int]:
    return {
      "states": self.states,
      "next_states": self.next_states,
      "cursor": self._cursor,
      "size": self._size,
    }

  def load_state_dict(self, state: dict[str, torch.Tensor | int]) -> None:
    self.states.copy_(state["states"])
    self.next_states.copy_(state["next_states"])
    self._cursor = int(state["cursor"])
    self._size = int(state["size"])
