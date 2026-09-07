"""Source-faithful AMP extension for DreamWaQ PPO."""

from itertools import chain
from typing import Any

import torch
from rsl_rl.algorithms import PPO

from ..dreamwaq.mdp.rl import DreamWaQPPO
from ..shared.contacts import JOINT_NAMES
from .motion import Go2AmpMotionLoader


class _Discriminator(torch.nn.Module):
  def __init__(
    self, device: torch.device, hidden_dims: tuple[int, ...] = (1024, 512)
  ) -> None:
    super().__init__()
    layers: list[torch.nn.Module] = []
    input_dim = 62
    for output_dim in hidden_dims:
      layers.extend((torch.nn.Linear(input_dim, output_dim), torch.nn.LeakyReLU()))
      input_dim = output_dim
    self.trunk = torch.nn.Sequential(*layers)
    self.head = torch.nn.Linear(input_dim, 1)
    self.to(device)

  def forward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    return self.head(self.trunk(torch.cat((state, next_state), dim=1)))

  def gradient_penalty(
    self, state: torch.Tensor, next_state: torch.Tensor
  ) -> torch.Tensor:
    expert = torch.cat((state, next_state), dim=1).requires_grad_(True)
    prediction = self.head(self.trunk(expert))
    gradient = torch.autograd.grad(
      prediction,
      expert,
      grad_outputs=torch.ones_like(prediction),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return 10.0 * gradient.norm(2, dim=1).square().mean()


class _AmpReplayBuffer:
  """Fixed-size ring buffer retaining policy transitions across iterations."""

  def __init__(self, capacity: int, device: torch.device) -> None:
    self.states = torch.empty((capacity, 31), device=device)
    self.next_states = torch.empty_like(self.states)
    self.capacity = capacity
    self.position = 0
    self.size = 0

  def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
    if states.shape != next_states.shape or states.shape[1] != 31:
      raise ValueError(f"Invalid AMP transition shapes: {states.shape}, {next_states.shape}")
    if states.shape[0] >= self.capacity:
      states, next_states = states[-self.capacity:], next_states[-self.capacity:]
    count = states.shape[0]
    first = min(count, self.capacity - self.position)
    self.states[self.position:self.position + first].copy_(states[:first])
    self.next_states[self.position:self.position + first].copy_(next_states[:first])
    remaining = count - first
    if remaining:
      self.states[:remaining].copy_(states[first:])
      self.next_states[:remaining].copy_(next_states[first:])
    self.position = (self.position + count) % self.capacity
    self.size = min(self.capacity, self.size + count)

  def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not self.size:
      raise RuntimeError("Cannot sample an empty AMP replay buffer")
    ids = torch.randint(self.size, (count,), device=self.states.device)
    return self.states[ids], self.next_states[ids]


class _RunningMeanStd(torch.nn.Module):
  """Torch equivalent of the Gym source Normalizer (including +/-10 clip)."""

  mean: torch.Tensor
  variance: torch.Tensor
  count: torch.Tensor

  def __init__(self, dimension: int) -> None:
    super().__init__()
    self.register_buffer("mean", torch.zeros(dimension, dtype=torch.float64))
    self.register_buffer("variance", torch.ones(dimension, dtype=torch.float64))
    self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

  @torch.no_grad()
  def update(self, values: torch.Tensor) -> None:
    values = values.detach().to(device=self.mean.device, dtype=torch.float64)
    batch_mean = values.mean(0)
    batch_variance = values.var(0, correction=0)
    batch_count = values.shape[0]
    delta = batch_mean - self.mean
    total = self.count + batch_count
    new_mean = self.mean + delta * batch_count / total
    moment = (
      self.variance * self.count
      + batch_variance * batch_count
      + delta.square() * self.count * batch_count / total
    )
    self.mean.copy_(new_mean)
    self.variance.copy_(moment / total)
    self.count.add_(batch_count)

  def normalize(self, values: torch.Tensor) -> torch.Tensor:
    mean = self.mean.to(dtype=values.dtype)
    std = torch.sqrt((self.variance + 1e-4).to(dtype=values.dtype))
    return torch.clamp((values - mean) / std, -10.0, 10.0)


class AmpDreamWaQPPO(DreamWaQPPO):
  """DreamWaQ PPO with Gym-compatible AMP data flow and optimization."""

  def __init__(
    self,
    *args,
    amp_replay_buffer_size: int = 1_000_000,
    amp_num_preload_transitions: int = 2_000_000,
    amp_reward_coef: float = .5,
    amp_discr_hidden_dims: tuple[int, ...] = (1024, 512),
    min_normalized_std: float = .05,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if self.rnd is not None or self.symmetry is not None or self.is_multi_gpu:
      raise ValueError("AMP-DreamWaQ source parity does not support RND, symmetry, or multi-GPU")
    device = torch.device(self.device)
    self.amp_reward_coef = amp_reward_coef
    self.discriminator = _Discriminator(device, tuple(amp_discr_hidden_dims))
    self.motion = Go2AmpMotionLoader(
      device, dt=.02, preload=amp_num_preload_transitions
    )
    self.amp_replay = _AmpReplayBuffer(amp_replay_buffer_size, device)
    self.amp_normalizer = _RunningMeanStd(31).to(device)
    self._amp_previous: torch.Tensor | None = None
    self._min_normalized_std = min_normalized_std
    self.min_action_std: torch.Tensor | None = None

    vae_ids = {id(parameter) for parameter in self.actor.vae.parameters()}  # type: ignore[attr-defined]
    self._policy_parameters = [
      parameter
      for parameter in chain(self.actor.parameters(), self.critic.parameters())
      if id(parameter) not in vae_ids
    ]
    self.optimizer = torch.optim.Adam(
      (
        {"params": self._policy_parameters, "name": "actor_critic"},
        {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-4, "name": "amp_trunk"},
        {"params": self.discriminator.head.parameters(), "weight_decay": 1e-2, "name": "amp_head"},
      ),
      lr=self.learning_rate,
    )

  @staticmethod
  def construct_algorithm(obs, env: Any, cfg: dict, device: str):
    algorithm = PPO.construct_algorithm(obs, env, cfg, device)
    assert isinstance(algorithm, AmpDreamWaQPPO)
    robot = env.unwrapped.scene["robot"]
    joint_ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    if tuple(names) != JOINT_NAMES:
      raise RuntimeError(f"Go2 joint order mismatch: {names}")
    limits = robot.data.soft_joint_pos_limits[0, joint_ids]
    algorithm.min_action_std = algorithm._min_normalized_std * (
      limits[:, 1] - limits[:, 0]
    ).to(device)
    return algorithm

  def act(self, obs):
    self._amp_previous = obs["amp"].detach().clone()
    return super().act(obs)

  def process_env_step(self, obs, rewards, dones, extras):
    if self._amp_previous is None:
      raise RuntimeError("AMP state was not captured before env.step")
    reset_state = obs["amp"].detach().clone()
    transition_next = reset_state.clone()
    terminal_ids = extras.get("amp_terminal_env_ids")
    terminal_states = extras.get("amp_terminal_states")
    if terminal_ids is not None:
      assert terminal_states is not None
      terminal_ids = terminal_ids.to(device=transition_next.device, dtype=torch.long)
      terminal_states = terminal_states.to(transition_next.device)
      transition_next[terminal_ids] = terminal_states
    done_ids = dones.nonzero(as_tuple=False).flatten()
    if done_ids.numel() and (
      terminal_ids is None
      or not torch.equal(torch.sort(done_ids).values, torch.sort(terminal_ids).values)
    ):
      raise RuntimeError("Missing pre-reset terminal AMP states for done environments")

    with torch.no_grad():
      prediction = self.discriminator(
        self.amp_normalizer.normalize(self._amp_previous),
        self.amp_normalizer.normalize(transition_next),
      )
      amp_reward = (
        .02
        * self.amp_reward_coef
        * torch.clamp(1.0 - .25 * torch.square(prediction - 1.0), min=0.0)
      ).squeeze(1)
    self.amp_replay.insert(self._amp_previous, transition_next)
    super().process_env_step(obs, rewards + amp_reward, dones, extras)
    self._amp_previous = reset_state

  def update(self) -> dict[str, float]:
    totals = {
      "value": 0.0, "surrogate": 0.0, "entropy": 0.0,
      "vae": 0.0, "velocity_estimation": 0.0, "reconstruction": 0.0,
      "vae_kl": 0.0, "amp": 0.0, "amp_grad_penalty": 0.0,
      "amp_policy_prediction": 0.0, "amp_expert_prediction": 0.0,
    }
    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for batch in generator:
      assert batch.observations is not None
      assert batch.actions is not None
      assert batch.old_distribution_params is not None
      assert batch.old_actions_log_prob is not None
      assert batch.advantages is not None
      assert batch.values is not None
      assert batch.returns is not None
      self.actor(batch.observations, stochastic_output=True)
      action_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations)
      distribution_params = self.actor.output_distribution_params
      entropy = self.actor.output_entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl_mean = self.actor.get_kl_divergence(
            batch.old_distribution_params, distribution_params
          ).mean()
          if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif 0.0 < kl_mean < self.desired_kl / 2.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

      ratio = torch.exp(action_log_prob - batch.old_actions_log_prob.squeeze())
      surrogate = -batch.advantages.squeeze() * ratio
      surrogate_clipped = -batch.advantages.squeeze() * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_loss = torch.maximum(
          (values - batch.returns).square(),
          (value_clipped - batch.returns).square(),
        ).mean()
      else:
        value_loss = (batch.returns - values).square().mean()

      policy_state_raw, policy_next_raw = self.amp_replay.sample(batch.actions.shape[0])
      expert_state_raw, expert_next_raw = self.motion.sample(batch.actions.shape[0])
      with torch.no_grad():
        policy_state = self.amp_normalizer.normalize(policy_state_raw)
        policy_next = self.amp_normalizer.normalize(policy_next_raw)
        expert_state = self.amp_normalizer.normalize(expert_state_raw)
        expert_next = self.amp_normalizer.normalize(expert_next_raw)
      policy_prediction = self.discriminator(policy_state, policy_next)
      expert_prediction = self.discriminator(expert_state, expert_next)
      amp_loss = .5 * (
        torch.nn.functional.mse_loss(expert_prediction, torch.ones_like(expert_prediction))
        + torch.nn.functional.mse_loss(policy_prediction, -torch.ones_like(policy_prediction))
      )
      gradient_penalty = self.discriminator.gradient_penalty(
        expert_state, expert_next
      )
      loss = (
        surrogate_loss + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean() + amp_loss + gradient_penalty
      )
      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self._policy_parameters, self.max_grad_norm)
      self.optimizer.step()
      self._clamp_action_std()
      self.amp_normalizer.update(policy_state_raw)
      self.amp_normalizer.update(expert_state_raw)

      history = batch.observations["history"]
      target_velocity = batch.observations["velocity"]
      target_frame = batch.observations["critic"][:, -45:]
      _, estimated_velocity, decoded, mean_latent, logvar_latent = self.actor.vae(  # type: ignore[attr-defined]
        history, sample=True
      )
      velocity_loss = torch.nn.functional.mse_loss(estimated_velocity, target_velocity)
      reconstruction_loss = torch.nn.functional.mse_loss(decoded, target_frame)
      kl_loss = -.5 * torch.mean(torch.sum(
        1 + logvar_latent - mean_latent.square() - logvar_latent.exp(), dim=-1
      ))
      vae_loss = velocity_loss + reconstruction_loss + self.vae_kl_weight * kl_loss
      self.vae_optimizer.zero_grad()
      vae_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.actor.vae.parameters(), self.max_grad_norm)  # type: ignore[attr-defined]
      self.vae_optimizer.step()

      values_to_add = {
        "value": value_loss, "surrogate": surrogate_loss, "entropy": entropy.mean(),
        "vae": vae_loss, "velocity_estimation": velocity_loss,
        "reconstruction": reconstruction_loss, "vae_kl": kl_loss,
        "amp": amp_loss, "amp_grad_penalty": gradient_penalty,
        "amp_policy_prediction": policy_prediction.mean(),
        "amp_expert_prediction": expert_prediction.mean(),
      }
      for name, value in values_to_add.items():
        totals[name] += value.item()

    updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    return {name: value / updates for name, value in totals.items()}

  @torch.no_grad()
  def _clamp_action_std(self) -> None:
    if self.min_action_std is None:
      raise RuntimeError("Minimum action std was not initialized from joint limits")
    distribution = self.actor.distribution
    if distribution.std_type == "scalar":  # type: ignore[attr-defined]
      distribution.std_param.data.copy_(  # type: ignore[attr-defined]
        torch.maximum(distribution.std_param.data, self.min_action_std)  # type: ignore[attr-defined]
      )
    else:
      distribution.log_std_param.data.copy_(  # type: ignore[attr-defined]
        torch.maximum(distribution.log_std_param.data, self.min_action_std.log())  # type: ignore[attr-defined]
      )

  def train_mode(self) -> None:
    super().train_mode()
    self.discriminator.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    self.discriminator.eval()

  def save(self) -> dict:
    result = super().save()
    result["discriminator_state_dict"] = self.discriminator.state_dict()
    result["amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
    return result

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    loaded_iteration = super().load(loaded_dict, load_cfg, strict)
    if load_cfg is None:
      self.discriminator.load_state_dict(
        loaded_dict["discriminator_state_dict"], strict=strict
      )
      self.amp_normalizer.load_state_dict(
        loaded_dict["amp_normalizer_state_dict"], strict=strict
      )
    return loaded_iteration
