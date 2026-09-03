"""PyTorch models for CTS, AMP, DreamWaQ and teacher-student policies."""

from __future__ import annotations

import copy
from typing import NamedTuple

import torch
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import GaussianDistribution
from rsl_rl.modules.mlp import MLP
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


def _activation(name: str) -> nn.Module:
  if name == "elu":
    return nn.ELU()
  if name == "relu":
    return nn.ReLU()
  if name == "tanh":
    return nn.Tanh()
  raise ValueError(f"Unsupported activation: {name}")


def _mlp(
  input_dim: int,
  output_dim: int,
  hidden_dims: tuple[int, ...],
  activation: str = "elu",
) -> nn.Sequential:
  layers: list[nn.Module] = []
  current = input_dim
  for hidden in hidden_dims:
    layers.extend((nn.Linear(current, hidden), _activation(activation)))
    current = hidden
  layers.append(nn.Linear(current, output_dim))
  return nn.Sequential(*layers)


class PolicyOutput(NamedTuple):
  action: torch.Tensor
  log_prob: torch.Tensor
  mean: torch.Tensor
  std: torch.Tensor


class Go2ClampedGaussianDistribution(GaussianDistribution):
  """Gaussian policy noise with the source per-joint minimum standard deviation."""

  def __init__(
    self,
    output_dim: int,
    min_std: tuple[float, ...],
    **kwargs,
  ) -> None:
    super().__init__(output_dim, **kwargs)
    if len(min_std) != output_dim:
      raise ValueError(f"Expected {output_dim} minimum std values, got {len(min_std)}")
    self.min_std: torch.Tensor
    self.register_buffer("min_std", torch.tensor(min_std, dtype=torch.float32))

  def update(self, mlp_output: torch.Tensor) -> None:
    super().update(mlp_output)
    assert self._distribution is not None
    self._distribution = Normal(
      self._distribution.mean,
      torch.maximum(self._distribution.stddev, self.min_std),
    )


class _GaussianPolicy(nn.Module):
  def __init__(self, input_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
    super().__init__()
    self.policy = _mlp(input_dim, action_dim, hidden_dims)
    self.log_std = nn.Parameter(torch.zeros(action_dim))

  def distribution(self, features: torch.Tensor) -> Normal:
    mean = self.policy(features)
    return Normal(mean, self.log_std.exp().expand_as(mean))

  def sample(self, features: torch.Tensor, deterministic: bool = False) -> PolicyOutput:
    distribution = self.distribution(features)
    action = distribution.mean if deterministic else distribution.rsample()
    return PolicyOutput(
      action,
      distribution.log_prob(action).sum(dim=-1),
      distribution.mean,
      distribution.stddev,
    )


class CtsActorCritic(nn.Module):
  """Concurrent teacher-student policy from the source CTS task."""

  def __init__(
    self,
    obs_dim: int,
    privileged_dim: int,
    critic_dim: int,
    history_dim: int,
    action_dim: int = 12,
    latent_dim: int = 32,
    hidden_dims: tuple[int, ...] = (512, 256, 128),
  ) -> None:
    super().__init__()
    self.teacher_encoder = _mlp(privileged_dim, latent_dim, (512, 256))
    self.student_encoder = _mlp(history_dim, latent_dim, (512, 256))
    self.policy = _GaussianPolicy(latent_dim + obs_dim, action_dim, hidden_dims)
    self.value = _mlp(latent_dim + critic_dim, 1, hidden_dims)

  def encode_teacher(self, privileged: torch.Tensor) -> torch.Tensor:
    return F.normalize(self.teacher_encoder(privileged), p=2.0, dim=-1)

  def encode_student(self, history: torch.Tensor) -> torch.Tensor:
    return F.normalize(self.student_encoder(history), p=2.0, dim=-1)

  def act(
    self,
    obs: torch.Tensor,
    privileged: torch.Tensor,
    history: torch.Tensor,
    teacher: bool = True,
    deterministic: bool = False,
  ) -> PolicyOutput:
    latent = (
      self.encode_teacher(privileged) if teacher else self.encode_student(history)
    )
    return self.policy.sample(torch.cat((latent, obs), dim=-1), deterministic)

  def evaluate(
    self,
    critic_obs: torch.Tensor,
    privileged: torch.Tensor,
    history: torch.Tensor,
    teacher: bool = True,
  ) -> torch.Tensor:
    latent = (
      self.encode_teacher(privileged) if teacher else self.encode_student(history)
    )
    return self.value(torch.cat((latent.detach(), critic_obs), dim=-1))

  def distillation_loss(
    self, privileged: torch.Tensor, history: torch.Tensor
  ) -> torch.Tensor:
    return F.mse_loss(
      self.encode_student(history), self.encode_teacher(privileged).detach()
    )


class DreamWaQVAE(nn.Module):
  """CENet/VAE used for latent and explicit velocity estimation."""

  def __init__(
    self, history_dim: int, latent_dim: int, explicit_dim: int, decode_dim: int
  ):
    super().__init__()
    # Source CENet applies ELU after both encoder linear layers, including the
    # 64-D bottleneck.  ``_mlp`` intentionally leaves output layers linear, so
    # spell this encoder out rather than silently dropping the final ELU.
    self.encoder = nn.Sequential(
      nn.Linear(history_dim, 128),
      nn.ELU(),
      nn.Linear(128, 64),
      nn.ELU(),
    )
    self.mean_latent = nn.Linear(64, latent_dim)
    self.logvar_latent = nn.Sequential(
      nn.Linear(64, latent_dim), nn.Hardtanh(-5.0, 5.0)
    )
    self.mean_explicit = nn.Linear(64, explicit_dim)
    self.logvar_explicit = nn.Sequential(
      nn.Linear(64, explicit_dim), nn.Hardtanh(-5.0, 5.0)
    )
    self.decoder = _mlp(latent_dim + explicit_dim, decode_dim, (128, 128))

  @staticmethod
  def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

  def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
    encoded = self.encoder(history)
    mean_latent = self.mean_latent(encoded)
    logvar_latent = self.logvar_latent(encoded)
    mean_explicit = self.mean_explicit(encoded)
    logvar_explicit = self.logvar_explicit(encoded)
    latent = self.reparameterize(mean_latent, logvar_latent)
    explicit = self.reparameterize(mean_explicit, logvar_explicit)
    # Keep the source VAE code layout: sampled velocity estimate first,
    # followed by the stochastic latent code.  The decoder is trained against
    # this ordering, so changing only the actor input would still make the
    # auxiliary reconstruction objective semantically inconsistent.
    decoded = self.decoder(torch.cat((explicit, latent), dim=-1))
    return {
      "latent": latent,
      "explicit": explicit,
      "decoded": decoded,
      "mean_latent": mean_latent,
      "logvar_latent": logvar_latent,
      "mean_explicit": mean_explicit,
      "logvar_explicit": logvar_explicit,
    }


class DreamWaQActorCritic(nn.Module):
  """DreamWaQ actor-critic with a trainable history VAE."""

  def __init__(
    self,
    obs_dim: int,
    critic_dim: int,
    history_dim: int,
    action_dim: int = 12,
    latent_dim: int = 16,
    explicit_dim: int = 3,
    hidden_dims: tuple[int, ...] = (256, 256, 256),
  ) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    self.vae = DreamWaQVAE(history_dim - obs_dim, latent_dim, explicit_dim, obs_dim)
    self.policy = _GaussianPolicy(
      obs_dim + latent_dim + explicit_dim, action_dim, hidden_dims
    )
    self.value = _mlp(critic_dim, 1, hidden_dims)

  def act(
    self, obs: torch.Tensor, history: torch.Tensor, deterministic: bool = False
  ) -> PolicyOutput:
    encoded = self.vae(history[..., : -self.obs_dim])
    # The source DreamWaQ actor receives ``code_vel || code_latent || obs``.
    # Keep the explicit velocity code first; swapping these two  fields would
    # preserve the tensor width while silently making source checkpoints
    # incompatible.
    features = torch.cat((encoded["explicit"], encoded["latent"], obs), dim=-1)
    return self.policy.sample(features, deterministic)

  def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
    return self.value(critic_obs)

  def auxiliary_loss(
    self,
    history: torch.Tensor,
    target_obs: torch.Tensor,
    target_velocity: torch.Tensor | None = None,
    live_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    encoded = self.vae(history[..., : -self.obs_dim])
    if live_mask is None:
      live_mask = torch.ones(
        (history.shape[0], 1), device=history.device, dtype=history.dtype
      )
    elif live_mask.ndim == 1:
      live_mask = live_mask.unsqueeze(-1)
    reconstruction = F.mse_loss(encoded["decoded"] * live_mask, target_obs * live_mask)
    velocity = (
      F.mse_loss(encoded["explicit"] * live_mask, target_velocity * live_mask)
      if target_velocity is not None
      else encoded["explicit"].new_zeros(())
    )
    kl_per_sample = torch.sum(
      1
      + encoded["logvar_latent"]
      - encoded["mean_latent"].square()
      - encoded["logvar_latent"].exp(),
      dim=-1,
    )
    kl_latent = -0.5 * torch.mean(kl_per_sample * live_mask.squeeze(-1))
    # The source update regularizes only the stochastic latent branch; the
    # explicit velocity branch is supervised directly and has no KL penalty.
    return velocity + reconstruction + kl_latent


class AmpDiscriminator(nn.Module):
  """Binary discriminator for motion-reference AMP observations."""

  def __init__(
    self,
    input_dim: int,
    hidden_dims: tuple[int, ...] = (1024, 512),
    amp_reward_coef: float = 0.5,
  ):
    super().__init__()
    self.state_dim = input_dim
    self.amp_reward_coef = float(amp_reward_coef)
    # The legacy AMP discriminator classifies a transition, not an isolated
    # frame: current state and next state are concatenated at its input.
    # It uses LeakyReLU and least-squares targets (+1 expert / -1 policy).
    layers: list[nn.Module] = []
    current_dim = input_dim * 2
    for hidden_dim in hidden_dims:
      layers.extend((nn.Linear(current_dim, hidden_dim), nn.LeakyReLU()))
      current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1))
    self.net = nn.Sequential(*layers)

  def forward(
    self, amp_obs: torch.Tensor, next_amp_obs: torch.Tensor | None = None
  ) -> torch.Tensor:
    if next_amp_obs is None:
      next_amp_obs = amp_obs
    return self.net(torch.cat((amp_obs, next_amp_obs), dim=-1))

  def loss(
    self,
    expert: torch.Tensor,
    policy: torch.Tensor,
    expert_next: torch.Tensor | None = None,
    policy_next: torch.Tensor | None = None,
    gradient_penalty_weight: float = 10.0,
  ) -> torch.Tensor:
    expert_logits = self(expert, expert_next)
    policy_logits = self(policy, policy_next)
    expert_loss = F.mse_loss(expert_logits, torch.ones_like(expert_logits))
    policy_loss = F.mse_loss(policy_logits, -torch.ones_like(policy_logits))
    if gradient_penalty_weight <= 0.0:
      return 0.5 * (expert_loss + policy_loss)
    if expert_next is None:
      expert_next = expert
    expert_pair = torch.cat((expert, expert_next), dim=-1).detach().requires_grad_(True)
    expert_prediction = self.net(expert_pair)
    gradient = torch.autograd.grad(
      outputs=expert_prediction,
      inputs=expert_pair,
      grad_outputs=torch.ones_like(expert_prediction),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    gradient_penalty = gradient.norm(2, dim=-1).square().mean()
    return (
      0.5 * (expert_loss + policy_loss) + gradient_penalty_weight * gradient_penalty
    )

  def reward(
    self,
    policy: torch.Tensor,
    next_policy: torch.Tensor | None = None,
    normalizer=None,
  ) -> torch.Tensor:
    """Return the source AMP shaping reward for a policy transition."""
    if normalizer is not None:
      policy = normalizer.normalize(policy)
      if next_policy is None:
        next_policy = policy
      else:
        next_policy = normalizer.normalize(next_policy)
    with torch.no_grad():
      logits = self(policy, next_policy)
      amp_reward = (
        0.02
        * self.amp_reward_coef
        * torch.clamp(1.0 - 0.25 * (logits - 1.0).square(), min=0.0)
      )
    return amp_reward.squeeze(-1)


class TeacherStudentActorCritic(nn.Module):
  """Teacher/student policy with terrain, privileged and LSTM encoders."""

  def __init__(
    self,
    obs_dim: int,
    critic_dim: int,
    terrain_dim: int = 187,
    # The migrated Go2 TS environments expose 70 domain-randomization fields
    # plus four foot-contact bits to the teacher encoder.
    privileged_dim: int = 74,
    action_dim: int = 12,
    encoder_dim: int = 16,
    hidden_dims: tuple[int, ...] = (512, 256, 128),
  ) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    self.terrain_encoder = _mlp(terrain_dim, encoder_dim, (256, 128))
    self.privileged_encoder = _mlp(privileged_dim, encoder_dim, (128, 64))
    self.student_lstm = nn.LSTM(obs_dim, 256, batch_first=True)
    self.student_head = _mlp(256, encoder_dim * 2, (256, 128))
    self.policy = _GaussianPolicy(obs_dim + encoder_dim * 2, action_dim, hidden_dims)
    self.value = _mlp(critic_dim, 1, hidden_dims)

  def teacher_features(
    self, terrain: torch.Tensor, privileged: torch.Tensor
  ) -> torch.Tensor:
    return torch.cat(
      (self.terrain_encoder(terrain), self.privileged_encoder(privileged)), dim=-1
    )

  def student_features(self, history: torch.Tensor) -> torch.Tensor:
    if history.ndim == 2:
      if history.shape[-1] == self.obs_dim:
        history = history.unsqueeze(1)
      elif history.shape[-1] % self.obs_dim == 0:
        history = history.view(history.shape[0], -1, self.obs_dim)
      else:
        raise ValueError(
          f"History dimension {history.shape[-1]} is not divisible by obs_dim {self.obs_dim}"
        )
    sequence, _ = self.student_lstm(history)
    return self.student_head(sequence[:, -1])

  def act(
    self, obs: torch.Tensor, features: torch.Tensor, deterministic: bool = False
  ) -> PolicyOutput:
    return self.policy.sample(torch.cat((obs, features), dim=-1), deterministic)

  def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
    return self.value(critic_obs)

  def distillation_loss(
    self, terrain: torch.Tensor, privileged: torch.Tensor, history: torch.Tensor
  ) -> torch.Tensor:
    return F.mse_loss(
      self.student_features(history),
      self.teacher_features(terrain, privileged).detach(),
    )


class _Go2ConditionalActor(MLPModel):
  """RSL-RL actor whose latent input is assembled from extra mjlab groups."""

  latent_kind: str = "none"
  use_student: bool = False
  teacher_encoder: nn.Sequential
  student_encoder: nn.Sequential
  vae: DreamWaQVAE
  student_lstm: nn.LSTM
  student_head: nn.Sequential
  terrain_encoder: nn.Sequential
  privileged_encoder: nn.Sequential

  def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs) -> None:
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    actor_dim = int(obs["actor"].shape[-1])
    self._go2_actor_dim = actor_dim
    self._go2_conditional_dim = 0
    self._go2_conditional_splits: tuple[int, ...] = ()
    hidden_dims = tuple(kwargs.get("hidden_dims", (512, 256, 128)))
    activation = kwargs.get("activation", "elu")
    self.latent_kind = type(self).latent_kind
    self.use_student = type(self).use_student
    latent_dim = 0
    if self.latent_kind == "cts":
      latent_dim = 32
      if self.use_student:
        source_dim = int(obs["history"].shape[-1])
        self._go2_cts_student_dim = source_dim - actor_dim
        self._go2_conditional_dim = self._go2_cts_student_dim
        self._go2_conditional_splits = (self._go2_conditional_dim,)
        self.student_encoder = _mlp(source_dim - actor_dim, latent_dim, (512, 256))
      else:
        privileged_dim = int(obs["privileged"].shape[-1])
        history_dim = int(obs["history"].shape[-1])
        self._go2_cts_student_dim = history_dim - actor_dim
        self._go2_conditional_dim = privileged_dim
        self._go2_conditional_splits = (privileged_dim, history_dim)
        self.teacher_encoder = _mlp(privileged_dim, latent_dim, (512, 256))
        self.student_encoder = _mlp(history_dim - actor_dim, latent_dim, (512, 256))
    elif self.latent_kind == "dreamwaq":
      history_dim = int(obs["history"].shape[-1])
      self._go2_conditional_dim = history_dim - actor_dim
      self._go2_conditional_splits = (self._go2_conditional_dim,)
      # The source VAE consumes history excluding the newest actor frame;
      # that 45-D frame is concatenated directly to the latent downstream.
      self.vae = DreamWaQVAE(history_dim - actor_dim, 16, 3, actor_dim)
      latent_dim = 19
    elif self.latent_kind == "ts":
      latent_dim = 32
      if self.use_student:
        history_dim = int(obs["history"].shape[-1])
        self._go2_conditional_dim = history_dim
        self._go2_conditional_splits = (history_dim,)
        # Match the source TS student encoder capacity so the same weights
        # can be used by the external-state deployment exporter.
        self.student_lstm = nn.LSTM(actor_dim, 256, num_layers=3, batch_first=True)
        self.student_head = _mlp(256, latent_dim, (256, 128))
      else:
        terrain_dim = int(obs["terrain"].shape[-1])
        privileged_dim = int(obs["privileged"].shape[-1])
        self._go2_conditional_dim = terrain_dim + privileged_dim
        self._go2_conditional_splits = (terrain_dim, privileged_dim)
        self.terrain_encoder = _mlp(terrain_dim, 16, (256, 128))
        self.privileged_encoder = _mlp(privileged_dim, 16, (128, 64))
    self.mlp = MLP(
      actor_dim + latent_dim,
      self.distribution.input_dim if self.distribution else output_dim,
      hidden_dims,
      activation,
    )
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

  def _student_history_features(self, history: torch.Tensor) -> torch.Tensor:
    if history.ndim == 2:
      if history.shape[-1] % self.obs_dim == 0:
        history = history.view(history.shape[0], -1, self.obs_dim)
      else:
        history = history.unsqueeze(1)
    sequence, _ = self.student_lstm(history)
    return self.student_head(sequence[:, -1])

  def _conditional_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    if self.latent_kind == "cts":
      # Source ``obs_hist_buf`` excludes the current observation.  The mjlab
      # compatibility group retains one extra frame so dropping its newest
      # 45-D block reproduces the source five-frame input exactly.
      student = F.normalize(
        self.student_encoder(obs["history"][..., : -self._go2_actor_dim]),
        p=2.0,
        dim=-1,
      )
      if self.use_student:
        return student
      teacher = F.normalize(self.teacher_encoder(obs["privileged"]), p=2.0, dim=-1)
      mask = obs.get("teacher_mask")
      if mask is None:
        return teacher
      # Match the source CTS update: PPO trains the teacher path, while the
      # student path is updated only by the latent distillation loss.
      return torch.where(mask > 0.5, teacher, student.detach())
    if self.latent_kind == "dreamwaq":
      encoded = self.vae(obs["history"][..., : -self._go2_actor_dim])
      # Source ``ActorCriticDreamWaQ`` concatenates the sampled velocity code
      # before the sampled latent code, then appends the current actor frame.
      return torch.cat((encoded["explicit"], encoded["latent"]), dim=-1)
    if self.latent_kind == "ts":
      if self.use_student:
        return self._student_history_features(obs["history"])
      return torch.cat(
        (
          self.terrain_encoder(obs["terrain"]),
          self.privileged_encoder(obs["privileged"]),
        ),
        dim=-1,
      )
    return obs["actor"].new_zeros((obs["actor"].shape[0], 0))

  def get_latent(self, obs, masks=None, hidden_state=None):
    del masks, hidden_state
    actor_obs = self.obs_normalizer(obs["actor"])
    conditional = self._conditional_features(obs)
    # All four source conditional policies build their actor input as
    # ``latent || current_observation`` (CTS, DreamWaQ and AMP-TS included).
    # Keep the ordinary PPO actor-only path unchanged.
    return (
      torch.cat((conditional, actor_obs), dim=-1)
      if conditional.shape[-1]
      else actor_obs
    )

  def as_onnx(self, verbose: bool):
    """Return a multi-input deterministic wrapper for deployment export.

    RSL-RL's default MLP wrapper assumes a single observation group.  Go2
    teacher/student actors additionally consume history or privileged inputs,
    so the deployment contract is ``(actor, conditional)`` where
    ``conditional`` is either the single selected group or the concatenation
    ``terrain || privileged`` for a teacher TS policy.
    """
    return _Go2ConditionalOnnxModel(self, verbose)


class CtsCriticModel(MLPModel):
  """CTS value model with the source latent||critic observation contract."""

  def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs) -> None:
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    privileged_dim = int(obs["privileged"].shape[-1])
    history_dim = int(obs["history"].shape[-1])
    self._go2_actor_dim = int(obs["actor"].shape[-1])
    hidden_dims = tuple(kwargs.get("hidden_dims", (512, 256, 128)))
    activation = kwargs.get("activation", "elu")
    self.teacher_encoder = _mlp(privileged_dim, 32, (512, 256))
    self.student_encoder = _mlp(history_dim - self._go2_actor_dim, 32, (512, 256))
    self.mlp = MLP(self.obs_dim + 32, output_dim, hidden_dims, activation)

  def get_latent(self, obs, masks=None, hidden_state=None):
    del masks, hidden_state
    critic_obs = self.obs_normalizer(
      torch.cat([obs[group] for group in self.obs_groups], dim=-1)
    )
    teacher = F.normalize(self.teacher_encoder(obs["privileged"]), p=2.0, dim=-1)
    student = F.normalize(
      self.student_encoder(obs["history"][..., : -self._go2_actor_dim]),
      p=2.0,
      dim=-1,
    )
    mask = obs.get("teacher_mask")
    if mask is None:
      latent = teacher
    else:
      latent = torch.where(mask > 0.5, teacher, student)
    # Source CTS explicitly detaches the selected encoder latent before the
    # value network so critic loss cannot update teacher/student encoders.
    return torch.cat((latent.detach(), critic_obs), dim=-1)


class _Go2ConditionalOnnxModel(nn.Module):
  """ONNX-safe deterministic wrapper around a conditional Go2 actor."""

  is_recurrent = False
  teacher_encoder: nn.Sequential
  student_encoder: nn.Sequential
  vae: DreamWaQVAE
  student_lstm: nn.LSTM
  student_head: nn.Sequential
  terrain_encoder: nn.Sequential
  privileged_encoder: nn.Sequential

  def __init__(self, model: _Go2ConditionalActor, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.mlp = copy.deepcopy(model.mlp)
    self.latent_kind = model.latent_kind
    # CTS is concurrent only during rollout collection.  The source
    # ``act_inference`` and deployment exporter always use the distilled
    # history encoder, never the privileged teacher.  Force that path for the
    # exported artifact even though the training actor itself is a mixed
    # teacher/student model.
    self.use_student = model.use_student or model.latent_kind == "cts"
    self.actor_dim = model._go2_actor_dim
    self.conditional_dim = (
      model._go2_cts_student_dim
      if model.latent_kind == "cts"
      else model._go2_conditional_dim
    )
    self.conditional_splits = model._go2_conditional_splits
    # Keep only the conditional modules needed by this policy.  Assigning the
    # modules (rather than deep-copying) preserves the trained weights when the
    # runner moves this wrapper to CPU for export.
    for name in (
      "encoder",
      "teacher_encoder",
      "student_encoder",
      "vae",
      "student_lstm",
      "student_head",
      "terrain_encoder",
      "privileged_encoder",
    ):
      if hasattr(model, name):
        setattr(self, name, copy.deepcopy(getattr(model, name)))
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

  def _features(self, conditional: torch.Tensor) -> torch.Tensor:
    if self.latent_kind == "cts":
      encoder = self.student_encoder if self.use_student else self.teacher_encoder
      return F.normalize(encoder(conditional), p=2.0, dim=-1)
    if self.latent_kind == "dreamwaq":
      # Deployment callers already provide the source five-frame history;
      # only the in-environment observation group carries an extra current
      # frame that must be removed before reaching this wrapper.
      encoded = self.vae.encoder(conditional)
      mean_latent = self.vae.mean_latent(encoded)
      mean_explicit = self.vae.mean_explicit(encoded)
      return torch.cat((mean_explicit, mean_latent), dim=-1)
    if self.latent_kind == "ts":
      if self.use_student:
        history = conditional.view(conditional.shape[0], -1, self.actor_dim)
        sequence, _ = self.student_lstm(history)
        return self.student_head(sequence[:, -1])
      terrain_dim, privileged_dim = self.conditional_splits
      terrain, privileged = torch.split(
        conditional, [terrain_dim, privileged_dim], dim=-1
      )
      return torch.cat(
        (self.terrain_encoder(terrain), self.privileged_encoder(privileged)), dim=-1
      )
    return conditional.new_zeros((conditional.shape[0], 0))

  def forward(self, actor: torch.Tensor, conditional: torch.Tensor) -> torch.Tensor:
    features = self._features(conditional)
    actor_obs = self.obs_normalizer(actor)
    # This wrapper is created only for conditional Go2 actors, so its feature
    # width is statically non-zero.  Avoid a tensor-shape Python branch here;
    # otherwise ONNX tracing bakes in the dummy batch and emits a misleading
    # generalization warning.
    latent = torch.cat((features, actor_obs), dim=-1)
    return self.deterministic_output(self.mlp(latent))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self.actor_dim),
      torch.zeros(1, self.conditional_dim),
    )

  @property
  def input_names(self) -> list[str]:
    return ["actor", "conditional"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]

  @property
  def dynamic_axes(self) -> dict[str, dict[int, str]]:
    return {
      "actor": {0: "batch"},
      "conditional": {0: "batch"},
      "actions": {0: "batch"},
    }


class CtsActorModel(_Go2ConditionalActor):
  latent_kind = "cts"


class CtsStudentActorModel(_Go2ConditionalActor):
  latent_kind = "cts"
  use_student = True


class DreamWaQActorModel(_Go2ConditionalActor):
  latent_kind = "dreamwaq"


class TeacherActorModel(_Go2ConditionalActor):
  latent_kind = "ts"


class StudentActorModel(_Go2ConditionalActor):
  latent_kind = "ts"
  use_student = True

  def as_recurrent_onnx(self, verbose: bool = False) -> nn.Module:
    """Return the source TS-Student ``obs,h,c -> actions,he,ce`` exporter.

    The regular mjlab exporter keeps the migrated five-frame conditional
    input.  The legacy student deployment instead carries an external
    three-layer LSTM state, so this wrapper exposes that contract while
    reusing the trained student encoder and actor head.
    """
    return _Go2TsStudentRecurrentOnnxModel(self, verbose)


class _Go2TsStudentRecurrentOnnxModel(nn.Module):
  """ONNX-safe recurrent wrapper backed by :class:`StudentActorModel`."""

  is_recurrent = True

  def __init__(self, model: StudentActorModel, verbose: bool = False) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.student_lstm = copy.deepcopy(model.student_lstm)
    self.student_head = copy.deepcopy(model.student_head)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.actor_dim = model._go2_actor_dim
    self.hidden_size = model.student_lstm.hidden_size
    self.num_layers = model.student_lstm.num_layers

  def forward(
    self,
    obs: torch.Tensor,
    h: torch.Tensor,
    c: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = self.obs_normalizer(obs)
    encoded, (h_next, c_next) = self.student_lstm(obs[:, None, :], (h, c))
    latent = self.student_head(encoded[:, -1])
    actions = self.deterministic_output(self.mlp(torch.cat((latent, obs), dim=-1)))
    return actions, h_next, c_next

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    return (
      torch.zeros(1, self.actor_dim),
      torch.zeros(self.num_layers, 1, self.hidden_size),
      torch.zeros(self.num_layers, 1, self.hidden_size),
    )

  @property
  def input_names(self) -> list[str]:
    return ["obs", "h", "c"]

  @property
  def output_names(self) -> list[str]:
    return ["actions", "he", "ce"]

  @property
  def dynamic_axes(self) -> dict[str, dict[int, str]]:
    return {
      "obs": {0: "batch"},
      "h": {1: "batch"},
      "c": {1: "batch"},
      "actions": {0: "batch"},
      "he": {1: "batch"},
      "ce": {1: "batch"},
    }
