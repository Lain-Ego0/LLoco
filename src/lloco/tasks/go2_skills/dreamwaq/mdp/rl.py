"""DreamWaQ VAE actor and PPO extension on the installed rsl_rl 5.4.2 API."""

from itertools import chain

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict


class _VAE(torch.nn.Module):
  """The source's 225→(3+16) VAE, without importing old rsl_rl."""

  def __init__(self) -> None:
    super().__init__()
    self.encoder = torch.nn.Sequential(
      torch.nn.Linear(225, 128), torch.nn.ELU(), torch.nn.Linear(128, 64), torch.nn.ELU()
    )
    self.mean_latent = torch.nn.Linear(64, 16)
    self.logvar_latent = torch.nn.Sequential(torch.nn.Linear(64, 16), torch.nn.Hardtanh(-5., 5.))
    self.mean_vel = torch.nn.Linear(64, 3)
    self.logvar_vel = torch.nn.Sequential(torch.nn.Linear(64, 3), torch.nn.Hardtanh(-5., 5.))
    self.decoder = torch.nn.Sequential(
      torch.nn.Linear(19, 128), torch.nn.ELU(), torch.nn.Linear(128, 128), torch.nn.ELU(), torch.nn.Linear(128, 45)
    )

  def forward(self, history: torch.Tensor, sample: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = self.encoder(history)
    mean_latent, logvar_latent = self.mean_latent(encoded), self.logvar_latent(encoded)
    mean_vel, logvar_vel = self.mean_vel(encoded), self.logvar_vel(encoded)
    if sample:
      latent = mean_latent + torch.randn_like(mean_latent) * torch.exp(.5 * logvar_latent)
      velocity = mean_vel + torch.randn_like(mean_vel) * torch.exp(.5 * logvar_vel)
    else:
      latent, velocity = mean_latent, mean_vel
    return torch.cat((velocity, latent), dim=-1), velocity, self.decoder(torch.cat((velocity, latent), dim=-1)), mean_latent, logvar_latent


class DreamWaQActor(MLPModel):
  """45-field policy augmented with source 3+16-dimensional VAE code."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.vae = _VAE()

  def _get_latent_dim(self) -> int:
    return 45 + 19

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    del masks, hidden_state
    actor = torch.cat([obs[name] for name in self.obs_groups], dim=-1)
    actor = self.obs_normalizer(actor)
    # Source samples in PPO rollout/update and uses posterior means in play.
    code, _, _, _, _ = self.vae(obs["history"], sample=self.training)
    return torch.cat((code, actor), dim=-1)


class DreamWaQPPO(PPO):
  """PPO plus the source's separate VAE reconstruction/velocity/KL update."""

  def __init__(self, *args, vae_learning_rate: float = 1e-3, vae_kl_weight: float = 1.0, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.vae_kl_weight = vae_kl_weight
    # Gym's PPO optimizer deliberately excludes VAE parameters.
    vae_ids = {id(p) for p in self.actor.vae.parameters()}  # type: ignore[attr-defined]
    policy_params = (p for p in chain(self.actor.parameters(), self.critic.parameters()) if id(p) not in vae_ids)
    self.optimizer = torch.optim.Adam(policy_params, lr=self.learning_rate)
    self.vae_optimizer = torch.optim.Adam(self.actor.vae.parameters(), lr=vae_learning_rate)  # type: ignore[attr-defined]

  def update(self) -> dict[str, float]:
    # The VAE does not share parameters with PPO, so doing this before the base
    # PPO pass is mathematically equivalent to source's second optimizer step.
    vae_loss = vel_loss = recon_loss = kl_loss = 0.0
    updates = 0
    for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
      history = batch.observations["history"]
      target_velocity = batch.observations["velocity"]
      target_frame = batch.observations["critic"][:, -45:]
      _, estimated_velocity, decoded, mean_latent, logvar_latent = self.actor.vae(history, sample=True)  # type: ignore[attr-defined]
      velocity_cost = torch.nn.functional.mse_loss(estimated_velocity, target_velocity)
      reconstruction_cost = torch.nn.functional.mse_loss(decoded, target_frame)
      kl_cost = -.5 * torch.mean(torch.sum(1 + logvar_latent - mean_latent.square() - logvar_latent.exp(), dim=-1))
      loss = velocity_cost + reconstruction_cost + self.vae_kl_weight * kl_cost
      self.vae_optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.actor.vae.parameters(), self.max_grad_norm)  # type: ignore[attr-defined]
      self.vae_optimizer.step()
      vae_loss += loss.item(); vel_loss += velocity_cost.item(); recon_loss += reconstruction_cost.item(); kl_loss += kl_cost.item(); updates += 1
    result = super().update()
    result.update({
      "vae": vae_loss / updates,
      "velocity_estimation": vel_loss / updates,
      "reconstruction": recon_loss / updates,
      "vae_kl": kl_loss / updates,
    })
    return result

  def save(self) -> dict:
    result = super().save()
    result["vae_optimizer_state_dict"] = self.vae_optimizer.state_dict()
    return result

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    loaded_iteration = super().load(loaded_dict, load_cfg, strict)
    if load_cfg is None and "vae_optimizer_state_dict" in loaded_dict:
      self.vae_optimizer.load_state_dict(loaded_dict["vae_optimizer_state_dict"])
    return loaded_iteration
