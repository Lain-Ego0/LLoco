"""Concurrent teacher/student PPO for the Go2 CTS migration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


class CtsStudentPolicy(nn.Module):
  def __init__(
    self,
    actor_dim: int = 45,
    history_dim: int = 225,
    teacher_dim: int = 233,
    critic_dim: int = 278,
    latent_dim: int = 32,
  ):
    super().__init__()
    self.teacher_encoder = nn.Sequential(
      nn.Linear(teacher_dim, 512),
      nn.ELU(),
      nn.Linear(512, 256),
      nn.ELU(),
      nn.Linear(256, latent_dim),
    )
    self.student_encoder = nn.Sequential(
      nn.Linear(history_dim, 512),
      nn.ELU(),
      nn.Linear(512, 256),
      nn.ELU(),
      nn.Linear(256, latent_dim),
    )
    self.actor = nn.Sequential(
      nn.Linear(actor_dim + latent_dim, 512),
      nn.ELU(),
      nn.Linear(512, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 12),
    )
    self.critic = nn.Sequential(
      nn.Linear(critic_dim + latent_dim, 512),
      nn.ELU(),
      nn.Linear(512, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 1),
    )
    self.std = nn.Parameter(torch.ones(12))

  @staticmethod
  def _normalize_latent(latent: torch.Tensor) -> torch.Tensor:
    # The Gym CTS reference applies L2 normalization to both encoders.
    return F.normalize(latent, p=2.0, dim=-1)

  @property
  def output_std(self) -> torch.Tensor:
    return self.std

  def _distribution(self, obs: torch.Tensor, latent: torch.Tensor) -> Normal:
    return Normal(
      self.actor(torch.cat((latent, obs), dim=1)),
      self.output_std.expand(obs.shape[0], -1),
    )

  def train_outputs(self, obs, teacher, critic_obs, history, teacher_mode):
    teacher_latent = self._normalize_latent(self.teacher_encoder(teacher))
    student_latent = self._normalize_latent(self.student_encoder(history))
    # The source CTS actor does not backpropagate PPO actor gradients through
    # the student encoder; it is trained solely by latent regression below.
    latent = torch.where(
      teacher_mode[:, None], teacher_latent, student_latent.detach()
    )
    dist = self._distribution(obs, latent)
    critic_latent = torch.where(
      teacher_mode[:, None], teacher_latent.detach(), student_latent
    )
    value = self.critic(torch.cat((critic_latent.detach(), critic_obs), dim=1))
    return dist, value, teacher_latent, student_latent

  def inference(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    latent = self._normalize_latent(self.student_encoder(history))
    return self.actor(torch.cat((latent, obs), dim=1))

  def forward(self, observations: Any) -> torch.Tensor:
    return self.inference(observations["actor"], observations["history"])


@dataclass
class _Transition:
  obs: torch.Tensor
  teacher: torch.Tensor
  critic_obs: torch.Tensor
  history: torch.Tensor
  actions: torch.Tensor
  log_prob: torch.Tensor
  values: torch.Tensor
  mean: torch.Tensor
  std: torch.Tensor


class _Storage:
  def __init__(self, steps: int, envs: int, device: torch.device):
    self.steps, self.envs, self.device = steps, envs, device
    self.data: list[_Transition] = []
    self.rewards: list[torch.Tensor] = []
    self.dones: list[torch.Tensor] = []
    self.returns: torch.Tensor | None = None
    self.advantages: torch.Tensor | None = None

  def add(self, transition, reward, done):
    self.data.append(transition)
    self.rewards.append(reward.detach().clone())
    self.dones.append(done.detach().clone())

  def compute_returns(self, last_value, gamma, lam):
    values = torch.stack([x.values.squeeze(-1) for x in self.data])
    rewards = torch.stack(self.rewards)
    dones = torch.stack(self.dones).float()
    adv = torch.zeros_like(last_value.squeeze(-1))
    advantages = torch.zeros_like(rewards)
    for i in reversed(range(self.steps)):
      next_value = last_value.squeeze(-1) if i == self.steps - 1 else values[i + 1]
      nonterminal = 1.0 - dones[i]
      delta = rewards[i] + gamma * next_value * nonterminal - values[i]
      adv = delta + gamma * lam * nonterminal * adv
      advantages[i] = adv
    self.returns = values + advantages
    self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  def clear(self):
    self.data.clear()
    self.rewards.clear()
    self.dones.clear()


class CtsPPO:
  def __init__(self, model, envs, steps, cfg, device):
    self.model, self.device = model.to(device), torch.device(device)
    self.gamma, self.lam = cfg.get("gamma", 0.99), cfg.get("lam", 0.95)
    self.clip_param = cfg.get("clip_param", 0.2)
    self.value_loss_coef = cfg.get("value_loss_coef", 1.0)
    self.entropy_coef = cfg.get("entropy_coef", 0.01)
    self.epochs = cfg.get("num_learning_epochs", 5)
    self.batches = cfg.get("num_mini_batches", 4)
    self.learning_rate = cfg.get("learning_rate", 1e-3)
    self.schedule = cfg.get("schedule", "fixed")
    self.desired_kl = cfg.get("desired_kl", 0.01)
    self.max_grad_norm = cfg.get("max_grad_norm", 1.0)
    indices = torch.arange(envs, device=self.device)
    self.teacher_idx, self.student_idx = (
      indices[indices % 4 != 0],
      indices[indices % 4 == 0],
    )
    trainable = (
      list(model.teacher_encoder.parameters())
      + list(model.actor.parameters())
      + list(model.critic.parameters())
      + [model.std]
    )
    self.optimizer = torch.optim.Adam(trainable, lr=self.learning_rate)
    self.student_optimizer = torch.optim.Adam(
      model.student_encoder.parameters(), lr=self.learning_rate
    )
    self.storage = _Storage(steps, envs, self.device)
    self._pending: _Transition | None = None

  @staticmethod
  def construct_algorithm(obs, env, cfg: dict, device: str):
    del obs
    # rsl_rl's logger expects this optional PPO field even when RND is absent.
    cfg["algorithm"].setdefault("rnd_cfg", None)
    return CtsPPO(
      CtsStudentPolicy(),
      env.num_envs,
      cfg["num_steps_per_env"],
      cfg["algorithm"],
      device,
    )

  def train_mode(self):
    self.model.train()

  def eval_mode(self):
    self.model.eval()

  def get_policy(self):
    return self.model

  def act(self, obs):
    actor, teacher, critic, history = (
      obs["actor"],
      obs["teacher"],
      obs["critic"],
      obs["history"],
    )
    mode = torch.zeros(actor.shape[0], dtype=torch.bool, device=self.device)
    mode[self.teacher_idx] = True
    dist, value, _, _ = self.model.train_outputs(actor, teacher, critic, history, mode)
    actions = dist.sample()
    self._pending = _Transition(
      actor.detach(),
      teacher.detach(),
      critic.detach(),
      history.detach(),
      actions.detach(),
      dist.log_prob(actions).sum(-1).detach(),
      value.detach(),
      dist.loc.detach(),
      dist.scale.detach(),
    )
    return actions

  def process_env_step(self, obs, rewards, dones, extras):
    del obs
    assert self._pending is not None
    reward = rewards.clone()
    if "time_outs" in extras:
      reward += (
        self.gamma
        * self._pending.values.squeeze(-1)
        * extras["time_outs"].to(self.device)
      )
    self.storage.add(self._pending, reward, dones)
    self._pending = None

  def compute_returns(self, obs):
    actor, teacher, critic, history = (
      obs["actor"],
      obs["teacher"],
      obs["critic"],
      obs["history"],
    )
    mode = torch.zeros(actor.shape[0], dtype=torch.bool, device=self.device)
    mode[self.teacher_idx] = True
    _, value, _, _ = self.model.train_outputs(actor, teacher, critic, history, mode)
    self.storage.compute_returns(value.detach(), self.gamma, self.lam)

  def update(self):
    assert self.storage.advantages is not None and self.storage.returns is not None
    obs = torch.cat([x.obs for x in self.storage.data])
    teacher = torch.cat([x.teacher for x in self.storage.data])
    critic = torch.cat([x.critic_obs for x in self.storage.data])
    history = torch.cat([x.history for x in self.storage.data])
    actions = torch.cat([x.actions for x in self.storage.data])
    old_log = torch.cat([x.log_prob for x in self.storage.data])
    old_values = torch.cat([x.values for x in self.storage.data]).squeeze(-1)
    old_mean = torch.cat([x.mean for x in self.storage.data])
    old_std = torch.cat([x.std for x in self.storage.data])
    advantages = self.storage.advantages.reshape(-1)
    returns = self.storage.returns.reshape(-1)
    env_ids = torch.arange(self.storage.envs, device=self.device).repeat(
      self.storage.steps
    )
    mode = env_ids % 4 != 0
    total = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "latent": 0.0}
    count = obs.shape[0]
    for _ in range(self.epochs):
      for ids in torch.randperm(count, device=self.device).split(
        max(1, count // self.batches)
      ):
        dist, value, _, _ = self.model.train_outputs(
          obs[ids], teacher[ids], critic[ids], history[ids], mode[ids]
        )
        log_prob = dist.log_prob(actions[ids]).sum(-1)
        ratio = torch.exp(log_prob - old_log[ids])
        clipped = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
        surrogate_terms = torch.maximum(
          -advantages[ids] * ratio, -advantages[ids] * clipped
        )
        # Gym CTS balances teacher and student policy gradients equally,
        # despite using a 3:1 environment split.  A flat mean would give the
        # student only one quarter of the actor update and produced the
        # observed standing-but-not-walking student policy.
        batch_mode = mode[ids]
        surrogate = (
          surrogate_terms[batch_mode].mean()
          + surrogate_terms[~batch_mode].mean()
        )
        current = value.squeeze(-1)
        old = old_values[ids]
        value_loss = torch.maximum(
          (current - returns[ids]).square(),
          (
            old
            + (current - old).clamp(-self.clip_param, self.clip_param)
            - returns[ids]
          ).square(),
        ).mean()
        entropy = dist.entropy().sum(-1).mean()
        if self.schedule == "adaptive" and self.desired_kl is not None:
          with torch.no_grad():
            kl = torch.log(dist.scale / old_std[ids]) + (
              old_std[ids].square() + (old_mean[ids] - dist.loc).square()
            ) / (2.0 * dist.scale.square()) - 0.5
            kl_mean = kl.sum(-1).mean()
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            for group in self.optimizer.param_groups:
              group["lr"] = self.learning_rate
        loss = (
          surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.max_grad_norm)
        self.optimizer.step()
        total["value"] += value_loss.item()
        total["surrogate"] += surrogate.item()
        total["entropy"] += entropy.item()
    student = ~mode
    student_ids = torch.where(student)[0]
    latent_sum = 0.0
    # Match the source implementation: one student update per PPO minibatch
    # and epoch, with a detached teacher target and gradient clipping.
    for _ in range(self.epochs):
      for ids in torch.randperm(student_ids.numel(), device=self.device).split(
        max(1, student_ids.numel() // self.batches)
      ):
        batch = student_ids[ids]
        student_latent = self.model._normalize_latent(self.model.student_encoder(history[batch]))
        with torch.no_grad():
          teacher_latent = self.model._normalize_latent(self.model.teacher_encoder(teacher[batch]))
        latent = (teacher_latent - student_latent).square().mean()
        self.student_optimizer.zero_grad()
        latent.backward()
        torch.nn.utils.clip_grad_norm_(self.model.student_encoder.parameters(), self.max_grad_norm)
        self.student_optimizer.step()
        latent_sum += latent.item()
    total["latent"] = latent_sum / max(1, self.epochs * self.batches)
    self.storage.clear()
    return total

  def save(self):
    return {
      "actor_state_dict": self.model.state_dict(),
      "critic_state_dict": {},
      "optimizer_state_dict": self.optimizer.state_dict(),
      "student_optimizer_state_dict": self.student_optimizer.state_dict(),
    }

  def load(self, loaded_dict, load_cfg, strict):
    if load_cfg is None or load_cfg.get("actor", True):
      actor_state = loaded_dict["actor_state_dict"]
      # Mjlab's generic loader migrates legacy ``std`` to the RSL-RL
      # distribution name.  CTS owns the Normal distribution directly, so
      # restore the parameter name expected by CtsStudentPolicy.
      if "std" not in actor_state and "distribution.std_param" in actor_state:
        actor_state["std"] = actor_state.pop("distribution.std_param")
      self.model.load_state_dict(actor_state, strict=strict)
    if load_cfg and load_cfg.get("optimizer") and "optimizer_state_dict" in loaded_dict:
      self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
    if (
      load_cfg
      and load_cfg.get("optimizer")
      and "student_optimizer_state_dict" in loaded_dict
    ):
      self.student_optimizer.load_state_dict(
        loaded_dict["student_optimizer_state_dict"]
      )
    return bool(load_cfg and load_cfg.get("iteration"))
