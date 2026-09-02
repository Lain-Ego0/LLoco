"""Minimal update loops for the migrated Go2 custom algorithm components."""

from __future__ import annotations

import os
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
    self.auxiliary: nn.Module | None = None
    self.auxiliary_optimizer: torch.optim.Optimizer | None = None

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
      self.amp_optimizer = torch.optim.Adam(self.amp_auxiliary.parameters(), lr=1e-4)
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
          actor.shape[-1], privileged.shape[-1], critic.shape[-1], history.shape[-1],
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
      self.auxiliary_optimizer = torch.optim.Adam(self.auxiliary.parameters(), lr=1e-4)

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
    if actor is None:
      return {}
    if self.auxiliary_kind == "cts" and isinstance(self.auxiliary, CtsActorCritic):
      assert history is not None and privileged is not None
      loss = self.auxiliary.distillation_loss(privileged, history)
      key = "cts_distillation"
    elif self.auxiliary_kind == "dreamwaq" and isinstance(self.auxiliary, DreamWaQActorCritic):
      assert history is not None
      loss = self.auxiliary.auxiliary_loss(history, actor, explicit)
      key = "dreamwaq_auxiliary"
    elif self.auxiliary_kind == "ts" and isinstance(self.auxiliary, TeacherStudentActorCritic):
      assert history is not None and privileged is not None and terrain is not None
      loss = self.auxiliary.distillation_loss(terrain, privileged, history)
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
    self.auxiliary_optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.auxiliary.parameters(), 1.0)
    self.auxiliary_optimizer.step()
    return {key: float(loss.detach())}

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
      self.amp_replay = Go2AmpReplayBuffer(policy_current.shape[-1], device=self.device)
    policy_current = policy_current.detach()
    policy_next = policy_next.detach()
    self.amp_replay.insert(policy_current, policy_next)
    sample_count = min(policy_current.shape[0], 256)
    policy, policy_next = self.amp_replay.sample(sample_count)
    if self.motion_loader is not None:
      expert, expert_next = self.motion_loader.sample_transition(sample_count)
    else:
      # Preserve a useful finite smoke path when no motion files are supplied.
      expert = policy.roll(1, dims=0)
      expert_next = expert.roll(-1, dims=0)
    # Match the legacy Normalizer: statistics are updated from unnormalized
    # policy/expert states, then both transition endpoints are clipped.
    self.amp_normalizer.update(torch.cat((policy, policy_next, expert, expert_next), dim=0))
    policy_n = self.amp_normalizer.normalize(policy)
    policy_next_n = self.amp_normalizer.normalize(policy_next)
    expert_n = self.amp_normalizer.normalize(expert)
    expert_next_n = self.amp_normalizer.normalize(expert_next)
    loss = self.amp_auxiliary.loss(expert_n, policy_n, expert_next_n, policy_next_n)
    self.amp_optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.amp_auxiliary.parameters(), 1.0)
    self.amp_optimizer.step()
    return {"amp": float(loss.detach())}

  def update(self) -> dict[str, float]:
    auxiliary_metrics = self._auxiliary_update()
    metrics = super().update()
    metrics.update(auxiliary_metrics)
    return metrics

  def save(self) -> dict:
    saved = super().save()
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


class DreamWaQPPO(Go2AuxiliaryPPO):
  auxiliary_kind = "dreamwaq"


class AmpPPO(Go2AuxiliaryPPO):
  auxiliary_kind = "amp"
  uses_amp = True

  def _auxiliary_update(self) -> dict[str, float]:
    return self._amp_update()


class TeacherStudentPPO(Go2AuxiliaryPPO):
  auxiliary_kind = "ts"


class AmpCtsPPO(CtsPPO):
  uses_amp = True
  def _auxiliary_update(self) -> dict[str, float]:
    metrics = super()._auxiliary_update()
    metrics.update(self._amp_update())
    return metrics


class AmpDreamWaQPPO(DreamWaQPPO):
  uses_amp = True
  def _auxiliary_update(self) -> dict[str, float]:
    metrics = super()._auxiliary_update()
    metrics.update(self._amp_update())
    return metrics


class AmpTeacherStudentPPO(TeacherStudentPPO):
  uses_amp = True
  def _auxiliary_update(self) -> dict[str, float]:
    metrics = super()._auxiliary_update()
    metrics.update(self._amp_update())
    return metrics
