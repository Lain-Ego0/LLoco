"""PPO teacher policy for source Go2 TS (terrain/domain encoders included)."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal


class TsTeacherPolicy(nn.Module):
  def __init__(self, actor_dim=45, terrain_dim=187, domain_dim=74, critic_dim=309):
    super().__init__()
    self.terrain_encoder = nn.Sequential(
      nn.Linear(terrain_dim, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 16),
    )
    self.domain_encoder = nn.Sequential(
      nn.Linear(domain_dim, 128),
      nn.ELU(),
      nn.Linear(128, 64),
      nn.ELU(),
      nn.Linear(64, 16),
    )
    self.actor = nn.Sequential(
      nn.Linear(actor_dim + 32, 512),
      nn.ELU(),
      nn.Linear(512, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 12),
    )
    self.critic = nn.Sequential(
      nn.Linear(critic_dim, 512),
      nn.ELU(),
      nn.Linear(512, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 1),
    )
    self.std = nn.Parameter(torch.ones(12))

  def _latent(self, obs):
    return torch.cat(
      (self.terrain_encoder(obs["terrain"]), self.domain_encoder(obs["domain"])), 1
    )

  def _dist(self, obs):
    # Gym's normalized-std floor is 0.05.  MuJoCo can become unstable when
    # exploratory actions grow without bound, so keep a conservative upper
    # guard as well; this is only a safety clamp, not a reward change.
    scale = self.std.clamp(min=0.05, max=2.0).expand(obs["actor"].shape[0], -1)
    return Normal(
      self.actor(torch.cat((obs["actor"], self._latent(obs)), 1)),
      scale,
    )

  def act(self, obs):
    return self._dist(obs).sample()

  def value(self, obs):
    return self.critic(obs["critic"])

  def inference(self, obs):
    return self.actor(torch.cat((obs["actor"], self._latent(obs)), 1))

  @property
  def output_std(self):
    return self.std

  def forward(self, obs):
    return self.inference(obs)


@dataclass
class _T:
  obs: Any
  actions: torch.Tensor
  log_prob: torch.Tensor
  values: torch.Tensor
  mean: torch.Tensor
  std: torch.Tensor


class TsPPO:
  def __init__(self, model, envs, steps, cfg, device, min_std=None):
    self.model, self.device = model.to(device), torch.device(device)
    self.gamma, self.lam = cfg.get("gamma", 0.99), cfg.get("lam", 0.95)
    self.clip, self.epochs, self.batches = (
      cfg.get("clip_param", 0.2),
      cfg.get("num_learning_epochs", 5),
      cfg.get("num_mini_batches", 4),
    )
    self.learning_rate = cfg.get("learning_rate", 1e-3)
    self.schedule = cfg.get("schedule", "fixed")
    self.desired_kl = cfg.get("desired_kl", 0.01)
    self.min_std = min_std
    self.entropy_coef = cfg.get("entropy_coef", 0.01)
    self.value_loss_coef = cfg.get("value_loss_coef", 1.0)
    self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
    self.steps, self.envs = steps, envs
    self.data, self.rewards, self.dones = [], [], []

  @staticmethod
  def construct_algorithm(obs, env, cfg, device):
    del obs
    cfg["algorithm"].setdefault("rnd_cfg", None)
    robot = env.unwrapped.scene["robot"]
    from ..shared.contacts import JOINT_NAMES
    joint_ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    if tuple(names) != JOINT_NAMES:
      raise RuntimeError(f"Go2 joint order mismatch: {names}")
    limits = robot.data.soft_joint_pos_limits[0, joint_ids].to(device)
    min_std = 0.05 * (limits[:, 1] - limits[:, 0])
    return TsPPO(
      TsTeacherPolicy(),
      env.num_envs,
      cfg["num_steps_per_env"],
      cfg["algorithm"],
      device,
      min_std=min_std,
    )

  def train_mode(self):
    self.model.train()

  def eval_mode(self):
    self.model.eval()

  def get_policy(self):
    return self.model

  def act(self, obs):
    dist = self.model._dist(obs)
    actions = dist.sample()
    self.data.append(
      _T(
        obs,
        actions.detach(),
        dist.log_prob(actions).sum(-1).detach(),
        self.model.value(obs).detach(),
        dist.loc.detach(),
        dist.scale.detach(),
      )
    )
    return actions

  def process_env_step(self, obs, rewards, dones, extras):
    # Gym bootstraps time-limit transitions (only true contacts are terminal).
    del obs
    timeout = extras.get("time_outs") if isinstance(extras, dict) else None
    if timeout is not None:
      rewards = (
        rewards
        # Use V(s_t), as Gym's PPO_TS does.  Evaluating the post-step
        # observation is incorrect for auto-reset environments: on a timeout
        # it is already the reset state (and on a contact it is terminal).
        + self.gamma * self.data[-1].values.detach().squeeze(-1) * timeout.float()
      )
    self.rewards.append(rewards.detach())
    self.dones.append(dones.detach())

  def compute_returns(self, obs):
    values = torch.stack([x.values.squeeze(-1) for x in self.data])
    rewards = torch.stack(self.rewards)
    dones = torch.stack(self.dones).float()
    adv = torch.zeros_like(values[0])
    advantages = torch.zeros_like(rewards)
    last = self.model.value(obs).detach().squeeze(-1)
    for i in reversed(range(self.steps)):
      nxt = last if i == self.steps - 1 else values[i + 1]
      delta = rewards[i] + self.gamma * nxt * (1 - dones[i]) - values[i]
      adv = delta + self.gamma * self.lam * (1 - dones[i]) * adv
      advantages[i] = adv
    # Keep value targets unnormalized; normalize only the policy advantages.
    self.returns = values + advantages
    self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  def update(self):
    obs = {k: torch.cat([x.obs[k] for x in self.data]) for k in self.data[0].obs.keys()}
    actions = torch.cat([x.actions for x in self.data])
    old_log = torch.cat([x.log_prob for x in self.data])
    old_values = torch.cat([x.values for x in self.data]).squeeze(-1)
    old_mean = torch.cat([x.mean for x in self.data])
    old_std = torch.cat([x.std for x in self.data])
    adv = self.advantages.flatten()
    returns = self.returns.flatten()
    total = {"value": 0.0, "surrogate": 0.0}
    count = actions.shape[0]
    for _ in range(self.epochs):
      for ids in torch.randperm(count, device=self.device).split(
        max(1, count // self.batches)
      ):
        dist = self.model._dist({k: v[ids] for k, v in obs.items()})
        # Match rsl_rl PPO_TS adaptive-KL learning-rate schedule.
        if self.schedule == "adaptive" and self.desired_kl is not None:
          with torch.no_grad():
            kl = torch.sum(
              torch.log(dist.scale / old_std[ids] + 1.e-5)
              + (old_std[ids].square() + (old_mean[ids] - dist.loc).square())
              / (2.0 * dist.scale.square()) - 0.5,
              dim=-1,
            ).mean()
            if kl > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl < self.desired_kl / 2.0 and kl > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            for group in self.optimizer.param_groups:
              group["lr"] = self.learning_rate
        log = dist.log_prob(actions[ids]).sum(-1)
        ratio = torch.exp(log - old_log[ids])
        s = torch.maximum(
          -adv[ids] * ratio, -adv[ids] * ratio.clamp(1 - self.clip, 1 + self.clip)
        ).mean()
        value = self.model.value({k: v[ids] for k, v in obs.items()}).squeeze(-1)
        vl = torch.maximum(
          (value - returns[ids]).square(),
          (
            old_values[ids]
            + (value - old_values[ids]).clamp(-self.clip, self.clip)
            - returns[ids]
          ).square(),
        ).mean()
        loss = (
          s
          + self.value_loss_coef * vl
          - self.entropy_coef * dist.entropy().sum(-1).mean()
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        # Gym's normalized-std floor is 0.05; keep the learned parameter from
        # crossing it between updates (the distribution applies the same
        # lower bound during sampling).
        if self.min_std is None:
          self.model.std.data.clamp_(min=0.05)
        else:
          self.model.std.data.copy_(torch.maximum(self.model.std.data, self.min_std))
        total["value"] += vl.item()
        total["surrogate"] += s.item()
    self.data.clear()
    self.rewards.clear()
    self.dones.clear()
    return total

  def save(self):
    return {
      "actor_state_dict": self.model.state_dict(),
      "critic_state_dict": {},
      "optimizer_state_dict": self.optimizer.state_dict(),
    }

  def load(self, loaded_dict, load_cfg, strict):
    state = dict(loaded_dict["actor_state_dict"])
    # Checkpoints produced by the earlier rsl_rl bridge stored the Gaussian
    # scale as ``distribution.std_param``.  Keep old TS checkpoints usable.
    if "std" not in state and "distribution.std_param" in state:
      state["std"] = state.pop("distribution.std_param")
    self.model.load_state_dict(state, strict=strict)
