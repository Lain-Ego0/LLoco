"""Finite-rollout helper for custom learning algorithms."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv

from .storage import Go2RolloutStorage, Go2Transition


class Go2CustomRunner:
  """Collect a finite rollout from a mjlab environment for a custom updater.

  The runner intentionally keeps environment interaction separate from the
  algorithm.  CTS/AMP/DreamWaQ/TS can therefore attach their own model inputs
  and auxiliary update while sharing reset, timeout and storage handling.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    policy: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, ...]],
    num_steps: int = 24,
  ) -> None:
    self.env = env
    self.policy = policy
    self.storage = Go2RolloutStorage(num_steps, env.num_envs, str(env.device))

  def collect(self) -> Go2RolloutStorage:
    obs, _ = self.env.reset()
    tensor_obs = cast(dict[str, torch.Tensor], obs)
    for _ in range(self.storage.num_steps):
      policy_output = self.policy(tensor_obs)
      if len(policy_output) == 2:
        actions, values = policy_output
        log_prob = torch.zeros_like(values.squeeze(-1))
      elif len(policy_output) == 3:
        actions, values, log_prob = policy_output
      else:
        raise ValueError("Go2 policy must return (actions, values[, log_prob])")
      next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
      self.storage.add(
        Go2Transition(
          obs=tensor_obs["actor"],
          critic_obs=tensor_obs["critic"],
          actions=actions,
          rewards=rewards.unsqueeze(-1) if rewards.ndim == 1 else rewards,
          dones=(terminated | truncated).unsqueeze(-1)
          if terminated.ndim == 1
          else terminated | truncated,
          values=values,
          log_prob=log_prob.unsqueeze(-1) if log_prob.ndim == 1 else log_prob,
        )
      )
      obs = next_obs
      tensor_obs = cast(dict[str, torch.Tensor], obs)
    last_output = self.policy(tensor_obs)
    last_values = last_output[1]
    self.storage.compute_returns(last_values)
    return self.storage
