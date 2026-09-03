"""Minimal update loops for the migrated Go2 custom algorithm components."""

from __future__ import annotations

import os
from itertools import chain
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO
from torch import nn

from .models import (
  AmpDiscriminator,
  CtsActorCritic,
  DreamWaQActorCritic,
  TeacherStudentActorCritic,
)
from .storage import Go2AmpReplayBuffer, Go2RolloutStorage, Go2RunningNormalizer


class _OptimizerMixin:
  def _step_loss(self, loss: torch.Tensor) -> float:
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.parameters, 1.0)
    self.optimizer.step()
    return float(loss.detach())


class CtsAlgorithm(_OptimizerMixin):
  """PPO-compatible CTS update with an explicit latent distillation term."""

  def __init__(self, model: CtsActorCritic, storage: Go2RolloutStorage, lr: float = 1e-3):
    self.model = model
    self.storage = storage
    self.parameters = list(model.parameters())
    self.optimizer = torch.optim.Adam(self.parameters, lr=lr)

  def update(
    self,
    obs: torch.Tensor | None = None,
    privileged: torch.Tensor | None = None,
    critic_obs: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    old_log_prob: torch.Tensor | None = None,
    returns: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    clip_param: float = 0.2,
  ) -> dict[str, float]:
    if any(item is None for item in (obs, privileged, critic_obs, history, actions, old_log_prob, returns, advantages)):
      if self.storage.advantages is None:
        raise RuntimeError("CTS storage has no returns")
      loss = self.storage.advantages.square().mean()
      return {"ppo": self._step_loss(loss), "distillation": 0.0}
    assert obs is not None and privileged is not None and critic_obs is not None
    assert history is not None and actions is not None and old_log_prob is not None
    assert returns is not None and advantages is not None
    output = self.model.act(obs, privileged, history, teacher=True)
    log_prob = self.model.policy.distribution(
      torch.cat((self.model.encode_teacher(privileged), obs), dim=-1)
    ).log_prob(actions).sum(dim=-1)
    ratio = torch.exp(log_prob - old_log_prob)
    surrogate = -torch.minimum(ratio * advantages, ratio.clamp(1.0 - clip_param, 1.0 + clip_param) * advantages).mean()
    value_loss = (self.model.evaluate(critic_obs, privileged, history, teacher=True).squeeze(-1) - returns).square().mean()
    distill = self.model.distillation_loss(privileged, history)
    loss = surrogate + value_loss + 0.1 * distill - 0.01 * output.log_prob.mean()
    return {"ppo": self._step_loss(loss), "distillation": float(distill.detach())}


class AmpAlgorithm(_OptimizerMixin):
  """AMP discriminator update; PPO policy update remains handled by mjlab."""

  def __init__(self, discriminator: AmpDiscriminator, lr: float = 1e-4):
    self.discriminator = discriminator
    self.parameters = list(discriminator.parameters())
    self.optimizer = torch.optim.Adam(self.parameters, lr=lr)

  def update(self, expert: torch.Tensor, policy: torch.Tensor) -> dict[str, float]:
    loss = self.discriminator.loss(expert, policy)
    return {"amp": self._step_loss(loss)}


class DreamWaQAlgorithm(_OptimizerMixin):
  """DreamWaQ VAE auxiliary update with reconstruction and KL losses."""

  def __init__(self, model: DreamWaQActorCritic, lr: float = 1e-3):
    self.model = model
    self.parameters = list(model.parameters())
    self.optimizer = torch.optim.Adam(self.parameters, lr=lr)

  def update(self, history: torch.Tensor, target_obs: torch.Tensor) -> dict[str, float]:
    loss = self.model.auxiliary_loss(history, target_obs)
    return {"dreamwaq": self._step_loss(loss)}


class TeacherStudentAlgorithm(_OptimizerMixin):
  """Teacher/student latent distillation update used by TS variants."""

  def __init__(self, model: TeacherStudentActorCritic, lr: float = 1e-3):
    self.model = model
    self.parameters = list(model.parameters())
    self.optimizer = torch.optim.Adam(self.parameters, lr=lr)

  def update(
    self,
    terrain: torch.Tensor,
    privileged: torch.Tensor,
    history: torch.Tensor,
  ) -> dict[str, float]:
    loss = self.model.distillation_loss(terrain, privileged, history)
    return {"distillation": self._step_loss(loss)}


class Go2AuxiliaryPPO(PPO):
  """Current RSL-RL PPO with a Go2-specific auxiliary update hook.

  PPO remains the native ``rsl_rl`` update while Go2-specific encoders and the
  AMP discriminator are trained from observation groups collected by mjlab.
  Conditional actors and their explicit ONNX wrappers keep deployment paths
  available during the migration.
  """

  auxiliary_kind: str = "none"
  uses_amp: bool = False
  amp_reward_coef: float = 0.5

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if (
      self.auxiliary_kind == "cts"
      and hasattr(self.actor, "teacher_encoder")
      and hasattr(self.critic, "teacher_encoder")
    ):
      # Source ActorCriticCTS owns one encoder pair shared by policy and
      # value paths.  RSL-RL constructs actor/critic as separate models, so
      # explicitly tie them after construction; critic latents are detached.
      self.critic.teacher_encoder = self.actor.teacher_encoder
      self.critic.student_encoder = self.actor.student_encoder
      # ``PPO`` created its optimizer before the modules were tied.  Rebuild
      # it so detached encoder copies are dropped and every active shared
      # parameter appears exactly once.
      unique_parameters = []
      seen_parameter_ids: set[int] = set()
      for parameter in chain(self.actor.parameters(), self.critic.parameters()):
        if id(parameter) not in seen_parameter_ids:
          unique_parameters.append(parameter)
          seen_parameter_ids.add(id(parameter))
      self.optimizer = type(self.optimizer)(unique_parameters, **self.optimizer.defaults)
    elif self.auxiliary_kind == "dreamwaq" and hasattr(self.actor, "vae"):
      # The source DreamWaQ PPO optimizer owns only actor/critic/noise
      # parameters.  CENet/VAE is updated exclusively by its separate 1e-3
      # reconstruction/velocity/KL optimizer, even though PPO backpropagates
      # through the sampled code while evaluating the actor.
      vae_parameter_ids = {id(parameter) for parameter in self.actor.vae.parameters()}
      ppo_parameters = [
        parameter
        for parameter in chain(self.actor.parameters(), self.critic.parameters())
        if id(parameter) not in vae_parameter_ids
      ]
      self.optimizer = type(self.optimizer)(
        ppo_parameters, **self.optimizer.defaults
      )
    self.auxiliary: nn.Module | None = None
    self.auxiliary_optimizer: torch.optim.Optimizer | None = None
    self._amp_rollout_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if self.uses_amp:
      # All migrated Go2 AMP variants use the same 31-D discriminator state.
      # Construct it before collection so iteration zero receives the source
      # random-discriminator shaping reward as well.
      self._ensure_amp_components(31)

  def _ensure_amp_components(self, state_dim: int) -> None:
    """Create AMP discriminator and statistics lazily from the env groups."""
    if not hasattr(self, "amp_auxiliary"):
      self.amp_auxiliary = AmpDiscriminator(
        state_dim,
        # Source Go2 AMP configs use ``amp_discr_hidden_dims=[1024, 512]``.
        # Keep that capacity for every AMP composition rather than silently
        # shrinking the discriminator in the mjlab adapter.
        hidden_dims=(1024, 512),
        amp_reward_coef=self.amp_reward_coef,
      ).to(self.device)
      self.amp_optimizer = torch.optim.Adam(
        (
          {
            "params": self.amp_auxiliary.net[:-1].parameters(),
            "weight_decay": 1.0e-4,
          },
          {
            "params": self.amp_auxiliary.net[-1].parameters(),
            "weight_decay": 1.0e-2,
          },
        ),
        # The discriminator shares the algorithm learning rate in the source:
        # 1e-3 for AMP-CTS/DreamWaQ and 1e-5 for AMP-TS.
        lr=self.learning_rate,
      )
    if not hasattr(self, "amp_normalizer"):
      self.amp_normalizer = Go2RunningNormalizer(state_dim, device=self.device)

  def _load_ts_teacher_encoders(self, auxiliary: TeacherStudentActorCritic) -> bool:
    """Load frozen TS teacher encoders for a standalone student task.

    The legacy ``*_ts_student`` runner first trains ``go2_ts`` and then uses
    that checkpoint as the teacher.  A student-only mjlab actor has no teacher
    modules of its own, so silently leaving the auxiliary teacher random would
    turn distillation into noise.  The checkpoint path is intentionally
    explicit via ``GO2_TS_TEACHER_CHECKPOINT`` and accepts both current
    ``actor_state_dict`` files and legacy ``model_state_dict`` containers.
    """
    checkpoint = os.environ.get("GO2_TS_TEACHER_CHECKPOINT")
    if not checkpoint:
      return False
    path = Path(checkpoint)
    if not path.is_file():
      raise FileNotFoundError(f"GO2_TS_TEACHER_CHECKPOINT does not exist: {path}")
    loaded = torch.load(path, map_location=self.device, weights_only=False)
    state = loaded.get("actor_state_dict", loaded.get("model_state_dict", loaded))
    if not isinstance(state, dict):
      raise ValueError(f"Teacher checkpoint has no actor state dictionary: {path}")
    # Current checkpoints use bare actor keys.  Legacy containers commonly
    # prefix them with ``actor.`` or store the actor under ``model.``.
    candidate = {}
    for key, value in state.items():
      normalized = key
      for prefix in ("actor.", "model."):
        if normalized.startswith(prefix):
          normalized = normalized[len(prefix) :]
      if normalized.startswith(("terrain_encoder.", "privileged_encoder.")):
        candidate[normalized] = value
    auxiliary.load_state_dict(candidate, strict=False)
    loaded_groups = {
      key.split(".", 1)[0] for key in candidate if "." in key
    }
    if not {"terrain_encoder", "privileged_encoder"}.issubset(loaded_groups):
      raise ValueError(
        f"Teacher checkpoint {path} lacks terrain/privileged encoder weights; "
        "export a go2_ts teacher checkpoint first"
      )
    for encoder in (auxiliary.terrain_encoder, auxiliary.privileged_encoder):
      encoder.eval()
      for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return True

  def process_env_step(self, obs, rewards, dones, extras):
    """Add AMP shaping before delegating to native RSL-RL PPO storage."""
    # Rollout collection runs under ``torch.inference_mode``.  Do not lazily
    # construct trainable modules there: inference tensors cannot later be
    # updated by the discriminator optimizer.  The first update initializes
    # the modules; subsequent iterations receive the shaping reward here.
    if self.uses_amp and hasattr(self, "amp_auxiliary") and hasattr(self, "amp_normalizer") and self.transition.observations is not None:
      current = self.transition.observations.get("amp")
      next_state = obs.get("amp")
      if current is not None and next_state is not None:
        next_state = next_state.clone()
        terminal = extras.get("terminal_amp_states")
        if terminal is not None:
          done_mask = dones.to(torch.bool)
          next_state[done_mask] = terminal[done_mask]
        # Preserve every physical transition, including the final rollout
        # step and terminal replacements.  Reconstructing pairs later by
        # shifting storage would lose the last step and cross reset boundaries.
        self._amp_rollout_pairs.append((current.clone(), next_state.clone()))
        rewards = rewards + self.amp_auxiliary.reward(
          self.amp_normalizer.normalize(current),
          self.amp_normalizer.normalize(next_state),
        )
    return super().process_env_step(obs, rewards, dones, extras)

  @staticmethod
  def _flat_group(storage, name: str) -> torch.Tensor | None:
    if name not in storage.observations.keys():
      return None
    tensor = storage.observations[name]
    return tensor.reshape(-1, tensor.shape[-1])

  @staticmethod
  def _amp_transition_pairs(storage) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return temporally valid AMP pairs without crossing env boundaries.

    Rollout storage is laid out as ``[time, env, feature]``.  Flattening it
    and calling ``roll`` silently pairs the last environment of one timestep
    with the first environment of the next timestep.  Such pairs are not
    physical transitions and can destabilize the discriminator.  Terminal
    rows are excluded because their successor observation belongs to a reset
    episode (the terminal-state replacement is handled by env-step shaping).
    """
    if "amp" not in storage.observations.keys() or storage.num_transitions_per_env < 2:
      return None
    sequence = storage.observations["amp"]
    done = storage.dones[:-1].squeeze(-1).to(torch.bool)
    valid = ~done
    current = sequence[:-1][valid]
    next_state = sequence[1:][valid]
    if current.shape[0] == 0:
      return None
    return current, next_state

  def _build_auxiliary(self) -> None:
    from .models import (
      AmpDiscriminator,
      CtsActorCritic,
      DreamWaQActorCritic,
      TeacherStudentActorCritic,
    )

    actor = self._flat_group(self.storage, "actor")
    history = self._flat_group(self.storage, "history")
    privileged = self._flat_group(self.storage, "privileged")
    terrain = self._flat_group(self.storage, "terrain")
    amp = self._flat_group(self.storage, "amp")
    if actor is None:
      return
    if self.auxiliary_kind == "cts" and history is not None and privileged is not None:
      critic = self._flat_group(self.storage, "critic")
      if critic is not None:
        self.auxiliary = CtsActorCritic(
          actor.shape[-1], privileged.shape[-1], critic.shape[-1],
          history.shape[-1] - actor.shape[-1],
          action_dim=12, hidden_dims=(128, 64)
        ).to(self.device)
        # The deployed CTS actor owns the teacher encoder.  Share it with the
        # auxiliary student/teacher module so distillation updates the policy
        # that PPO and ONNX export actually use.
        if hasattr(self.actor, "teacher_encoder"):
          self.auxiliary.teacher_encoder = self.actor.teacher_encoder
        if hasattr(self.actor, "student_encoder"):
          self.auxiliary.student_encoder = self.actor.student_encoder
    elif self.auxiliary_kind == "dreamwaq" and history is not None:
      critic = self._flat_group(self.storage, "critic")
      if critic is not None:
        self.auxiliary = DreamWaQActorCritic(
          actor.shape[-1], critic.shape[-1], history.shape[-1], action_dim=12,
          hidden_dims=(128, 64)
        ).to(self.device)
        if hasattr(self.actor, "vae"):
          self.auxiliary.vae = self.actor.vae
    elif self.auxiliary_kind == "ts" and history is not None and privileged is not None and terrain is not None:
      critic = self._flat_group(self.storage, "critic")
      if critic is not None:
        self.auxiliary = TeacherStudentActorCritic(
          actor.shape[-1], critic.shape[-1], terrain_dim=terrain.shape[-1],
          privileged_dim=privileged.shape[-1], action_dim=12, encoder_dim=16,
          hidden_dims=(128, 64)
        ).to(self.device)
        # Student tasks must distill into the recurrent encoder used by the
        # exported policy; teacher tasks likewise reuse their active teacher
        # encoders.  Untied modules remain in the auxiliary model for the
        # complementary side of the distillation pair.
        for name in ("student_lstm", "student_head", "terrain_encoder", "privileged_encoder"):
          if hasattr(self.actor, name):
            setattr(self.auxiliary, name, getattr(self.actor, name))
        if not hasattr(self.actor, "terrain_encoder"):
          self._ts_teacher_checkpoint_loaded = self._load_ts_teacher_encoders(self.auxiliary)
          if not self._ts_teacher_checkpoint_loaded:
            # Keep an explicit marker so callers can distinguish a finite
            # smoke fallback from source-equivalent student training.
            print(
              "[WARN] TS-Student has no teacher checkpoint; set "
              "GO2_TS_TEACHER_CHECKPOINT for source-compatible distillation."
            )
    elif self.auxiliary_kind == "amp":
      amp_input = actor if amp is None else amp
      self.auxiliary = AmpDiscriminator(amp_input.shape[-1], hidden_dims=(1024, 512)).to(self.device)
    if self.auxiliary is not None:
      # Source CTS student encoders and DreamWaQ VAE both use 1e-3.
      auxiliary_parameters = (
        self.actor.vae.parameters()
        if self.auxiliary_kind == "dreamwaq" and hasattr(self.actor, "vae")
        else self.auxiliary.parameters()
      )
      self.auxiliary_optimizer = torch.optim.Adam(
        auxiliary_parameters, lr=1e-3
      )

  def _auxiliary_update(self) -> dict[str, float]:
    if self.auxiliary is None:
      self._build_auxiliary()
    if self.auxiliary is None or self.auxiliary_optimizer is None:
      return {}
    actor = self._flat_group(self.storage, "actor")
    history = self._flat_group(self.storage, "history")
    privileged = self._flat_group(self.storage, "privileged")
    terrain = self._flat_group(self.storage, "terrain")
    explicit = self._flat_group(self.storage, "explicit")
    critic = self._flat_group(self.storage, "critic")
    if actor is None:
      return {}
    if self.auxiliary_kind == "cts" and isinstance(self.auxiliary, CtsActorCritic):
      assert history is not None and privileged is not None
      history = history[..., :-actor.shape[-1]]
      teacher_mask = self._flat_group(self.storage, "teacher_mask")
      if teacher_mask is not None:
        student_rows = teacher_mask.squeeze(-1) <= 0.5
        history = history[student_rows]
        privileged = privileged[student_rows]
      inputs = (privileged, history)
      key = "cts_distillation"
    elif self.auxiliary_kind == "dreamwaq" and isinstance(self.auxiliary, DreamWaQActorCritic):
      assert history is not None and critic is not None
      # Source reconstruction targets the uncorrupted current frame stored at
      # the tail of the newest privileged critic frame, not the noisy actor
      # observation.  Terminal samples are masked out of all VAE losses.
      reconstruction = critic[..., -actor.shape[-1]:]
      live = (~self.storage.dones.to(torch.bool)).reshape(-1, 1).to(actor.dtype)
      inputs = (history, reconstruction, explicit, live)
      key = "dreamwaq_auxiliary"
    elif self.auxiliary_kind == "ts" and isinstance(self.auxiliary, TeacherStudentActorCritic):
      assert history is not None and privileged is not None and terrain is not None
      inputs = (terrain, privileged, history)
      key = "ts_distillation"
    elif self.auxiliary_kind == "amp" and isinstance(self.auxiliary, AmpDiscriminator):
      pairs = self._amp_transition_pairs(self.storage)
      if pairs is None:
        return {"amp": 0.0}
      policy, policy_next = (item.detach() for item in pairs)
      # No motion directory is available in this auxiliary-only fallback;
      # use a time-shifted copy of valid transitions while preserving pair
      # dimensions.  The full AMP path below replaces it with expert data.
      expert = policy.roll(1, dims=0)
      expert_next = policy_next.roll(1, dims=0)
      loss = self.auxiliary.loss(expert, policy, expert_next, policy_next)
      key = "amp"
    else:
      return {}
    # The source algorithms update their auxiliary encoder/VAE once per PPO
    # mini-batch and epoch, rather than once over the complete rollout.
    sample_count = inputs[0].shape[0]
    if sample_count == 0:
      return {key: 0.0}
    total_loss = 0.0
    update_count = 0
    for _ in range(self.num_learning_epochs):
      for indices in torch.randperm(sample_count, device=self.device).tensor_split(
        self.num_mini_batches
      ):
        if indices.numel() == 0:
          continue
        if self.auxiliary_kind == "cts":
          loss = self.auxiliary.distillation_loss(
            inputs[0][indices], inputs[1][indices]
          )
        elif self.auxiliary_kind == "dreamwaq":
          target_velocity = None if inputs[2] is None else inputs[2][indices]
          loss = self.auxiliary.auxiliary_loss(
            inputs[0][indices], inputs[1][indices], target_velocity,
            inputs[3][indices],
          )
        else:
          loss = self.auxiliary.distillation_loss(
            inputs[0][indices], inputs[1][indices], inputs[2][indices]
          )
        self.auxiliary_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.auxiliary.parameters(), 1.0)
        self.auxiliary_optimizer.step()
        total_loss += float(loss.detach())
        update_count += 1
    return {key: total_loss / update_count}

  def _amp_update(self) -> dict[str, float]:
    from .motion import Go2MotionLoader

    # Loading is opt-in so the task remains runnable when the source dataset is
    # not present.  Set GO2_MOTION_DIR to the migrated/source motion directory
    # to train the discriminator against real expert transitions.
    if not hasattr(self, "motion_loader"):
      motion_dir = os.environ.get("GO2_MOTION_DIR")
      self.motion_loader = (
        Go2MotionLoader.from_directory(motion_dir, device=self.device)
        if motion_dir and Path(motion_dir).is_dir()
        else None
      )

    if self._amp_rollout_pairs:
      policy_current = torch.cat(
        [pair[0] for pair in self._amp_rollout_pairs], dim=0
      ).clone()
      policy_next = torch.cat(
        [pair[1] for pair in self._amp_rollout_pairs], dim=0
      ).clone()
      self._amp_rollout_pairs.clear()
      pairs = (policy_current, policy_next)
    else:
      pairs = self._amp_transition_pairs(self.storage)
    if pairs is None:
      actor = self._flat_group(self.storage, "actor")
      amp_obs = self._flat_group(self.storage, "amp")
      if amp_obs is None:
        amp_obs = actor
      if amp_obs is None:
        return {}
      # Keep the dimension available for lazy initialization even when a
      # very short smoke rollout has no valid temporal pair.
      self._ensure_amp_components(int(amp_obs.shape[-1]))
      return {}
    policy_current, policy_next = pairs
    self._ensure_amp_components(int(policy_current.shape[-1]))
    if not hasattr(self, "amp_replay"):
      self.amp_replay = Go2AmpReplayBuffer(
        policy_current.shape[-1],
        capacity=int(getattr(self, "amp_replay_buffer_size", 100_000)),
        device=self.device,
      )
    policy_current = policy_current.detach()
    policy_next = policy_next.detach()
    self.amp_replay.insert(policy_current, policy_next)
    sample_count = max(1, policy_current.shape[0] // self.num_mini_batches)
    total_loss = 0.0
    update_count = self.num_learning_epochs * self.num_mini_batches
    for param_group in self.amp_optimizer.param_groups:
      param_group["lr"] = self.learning_rate
    for _ in range(update_count):
      policy, policy_next = self.amp_replay.sample(sample_count)
      if self.motion_loader is not None:
        expert, expert_next = self.motion_loader.sample_transition(sample_count)
      else:
        # Preserve a useful finite smoke path when no motion files are supplied.
        expert = policy.roll(1, dims=0)
        expert_next = expert.roll(-1, dims=0)
      # The legacy implementation normalizes with the existing statistics and
      # updates them from unnormalized current states after the optimizer step.
      policy_n = self.amp_normalizer.normalize(policy)
      policy_next_n = self.amp_normalizer.normalize(policy_next)
      expert_n = self.amp_normalizer.normalize(expert)
      expert_next_n = self.amp_normalizer.normalize(expert_next)
      loss = self.amp_auxiliary.loss(
        expert_n, policy_n, expert_next_n, policy_next_n
      )
      self.amp_optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.amp_auxiliary.parameters(), 1.0)
      self.amp_optimizer.step()
      self.amp_normalizer.update(policy)
      self.amp_normalizer.update(expert)
      total_loss += float(loss.detach())
    return {"amp": total_loss / update_count}

  def update(self) -> dict[str, float]:
    # Keep the source's PPO-before-auxiliary ordering.  CTS first completes
    # every PPO mini-batch and only then distils the student encoder; DreamWaQ
    # performs each VAE step after a PPO step.  Current RSL-RL owns its PPO
    # mini-batch loop, so this adapter finishes that loop before running the
    # same number of auxiliary mini-batch updates.  This is important for CTS
    # in particular: updating the student first would change its (detached)
    # action-conditioning latent before PPO evaluates the rollout actions.
    # ``PPO.update`` only resets the storage cursor, so its tensors remain
    # available to the following auxiliary phase.
    metrics = super().update()
    auxiliary_metrics = self._auxiliary_update()
    metrics.update(auxiliary_metrics)
    # Recurrent TS updates can leave sizable temporary CUDA allocator blocks
    # alive while MuJoCo-Warp captures the next simulation graph.  Releasing
    # cached (not live) blocks here prevents the 1024-env student validation
    # from failing on the following ``env.step`` without changing tensors or
    # the source rollout semantics.
    if torch.device(self.device).type == "cuda":
      torch.cuda.empty_cache()
    return metrics

  def save(self) -> dict:
    saved = super().save()
    if hasattr(self, "student_distill_optimizer"):
      saved["go2_student_distill_optimizer_state_dict"] = (
        self.student_distill_optimizer.state_dict()
      )
    if self.auxiliary is not None:
      saved["go2_auxiliary_state_dict"] = self.auxiliary.state_dict()
      if self.auxiliary_optimizer is not None:
        saved["go2_auxiliary_optimizer_state_dict"] = self.auxiliary_optimizer.state_dict()
    if hasattr(self, "amp_auxiliary"):
      saved["go2_amp_discriminator_state_dict"] = self.amp_auxiliary.state_dict()
      saved["go2_amp_optimizer_state_dict"] = self.amp_optimizer.state_dict()
    if hasattr(self, "amp_normalizer"):
      saved["go2_amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
    if hasattr(self, "amp_replay"):
      saved["go2_amp_replay_state_dict"] = self.amp_replay.state_dict()
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    if self.auxiliary is None and "go2_auxiliary_state_dict" in loaded_dict:
      self._build_auxiliary()
    if self.auxiliary is not None and "go2_auxiliary_state_dict" in loaded_dict:
      self.auxiliary.load_state_dict(loaded_dict["go2_auxiliary_state_dict"], strict=strict)
      if self.auxiliary_optimizer is not None and "go2_auxiliary_optimizer_state_dict" in loaded_dict:
        self.auxiliary_optimizer.load_state_dict(loaded_dict["go2_auxiliary_optimizer_state_dict"])
    result = super().load(loaded_dict, load_cfg, strict)
    if (
      hasattr(self, "student_distill_optimizer")
      and "go2_student_distill_optimizer_state_dict" in loaded_dict
    ):
      self.student_distill_optimizer.load_state_dict(
        loaded_dict["go2_student_distill_optimizer_state_dict"]
      )
    if "go2_amp_discriminator_state_dict" in loaded_dict:
      if not hasattr(self, "amp_auxiliary"):
        self._amp_update()
      self.amp_auxiliary.load_state_dict(loaded_dict["go2_amp_discriminator_state_dict"], strict=strict)
      self.amp_optimizer.load_state_dict(loaded_dict["go2_amp_optimizer_state_dict"])
      if "go2_amp_normalizer_state_dict" in loaded_dict:
        self.amp_normalizer.load_state_dict(loaded_dict["go2_amp_normalizer_state_dict"])
      if "go2_amp_replay_state_dict" in loaded_dict:
        replay_state = loaded_dict["go2_amp_replay_state_dict"]
        if not hasattr(self, "amp_replay"):
          self.amp_replay = Go2AmpReplayBuffer(
            int(replay_state["states"].shape[-1]),
            capacity=int(replay_state["states"].shape[0]),
            device=self.device,
          )
        self.amp_replay.load_state_dict(replay_state)
    return result


class CtsPPO(Go2AuxiliaryPPO):
  auxiliary_kind = "cts"

  def update(self) -> dict[str, float]:
    # Source CTS computes teacher and student surrogate means independently
    # and adds them, giving the 1/4 student partition equal group weight to
    # the 3/4 teacher partition.  Pre-weighting advantages reproduces that
    # objective inside the native shuffled PPO mini-batch implementation.
    if "teacher_mask" in self.storage.observations.keys():
      teacher = self.storage.observations["teacher_mask"] > 0.5
      group_weight = torch.where(teacher, 1.0 / 0.75, 1.0 / 0.25)
      # Rollout returns are computed under inference mode by RSL-RL.  Clone
      # before weighting so the PPO update owns a regular mutable tensor.
      self.storage.advantages = self.storage.advantages.clone() * group_weight
    return super().update()


class DreamWaQPPO(Go2AuxiliaryPPO):
  auxiliary_kind = "dreamwaq"


class AmpPPO(Go2AuxiliaryPPO):
  auxiliary_kind = "amp"
  uses_amp = True

  # The source AMP configurations reserve a million policy transitions.  The
  # buffer is allocated lazily after the first valid rollout pair, so keeping
  # this capacity does not inflate ordinary PPO or pre-step memory usage.
  amp_replay_buffer_size = 1_000_000

  def _auxiliary_update(self) -> dict[str, float]:
    return self._amp_update()


class TeacherStudentPPO(Go2AuxiliaryPPO):
  auxiliary_kind = "ts"


class AmpCtsPPO(CtsPPO):
  uses_amp = True
  amp_replay_buffer_size = 1_000_000
  def _auxiliary_update(self) -> dict[str, float]:
    metrics = super()._auxiliary_update()
    metrics.update(self._amp_update())
    return metrics


class AmpDreamWaQPPO(DreamWaQPPO):
  uses_amp = True
  amp_replay_buffer_size = 1_000_000
  def _auxiliary_update(self) -> dict[str, float]:
    metrics = super()._auxiliary_update()
    metrics.update(self._amp_update())
    return metrics


class AmpTeacherStudentPPO(TeacherStudentPPO):
  uses_amp = True
  amp_replay_buffer_size = 1_000_000
  def _auxiliary_update(self) -> dict[str, float]:
    metrics = super()._auxiliary_update()
    metrics.update(self._amp_update())
    return metrics
