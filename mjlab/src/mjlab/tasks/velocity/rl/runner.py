import copy
import os
import time
from pathlib import Path

import torch
import wandb
from torch import nn

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl.go2_algorithms.deployment import (
  go2_policy_contract_metadata,
)


class _RecurrentStudentPolicy:
  """Stateful callable used by the normal mjlab play loop for TS students."""

  def __init__(self, actor, env) -> None:
    self.actor = actor
    self.env = env
    self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None

  @torch.no_grad()
  def __call__(self, obs) -> torch.Tensor:
    actor_obs = self.actor.obs_normalizer(obs["actor"])
    batch_size = actor_obs.shape[0]
    if self.hidden is None or self.hidden[0].shape[1] != batch_size:
      shape = (
        self.actor.student_lstm.num_layers,
        batch_size,
        self.actor.student_lstm.hidden_size,
      )
      self.hidden = (actor_obs.new_zeros(shape), actor_obs.new_zeros(shape))
    # mjlab resets episode_length_buf before exposing the first observation of
    # a new episode, so it provides the done mask unavailable to this callable.
    reset = self.env.unwrapped.episode_length_buf == 0
    if torch.any(reset):
      keep = (~reset).view(1, -1, 1)
      self.hidden = tuple(state * keep for state in self.hidden)  # type: ignore[assignment]
    encoded, self.hidden = self.actor.student_lstm(
      actor_obs[:, None, :], self.hidden
    )
    features = self.actor.student_head(encoded[:, -1])
    return self.actor.mlp(torch.cat((features, actor_obs), dim=-1))


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def get_inference_policy(self, device: str | None = None):
    """Return the source inference path for conditional Go2 policies."""
    self.alg.eval_mode()
    actor = self.alg.get_policy().to(device)
    if getattr(actor, "latent_kind", None) == "cts":
      # Source CTS trains privileged-teacher and history-student partitions
      # concurrently, but playback always calls ``act_inference`` and thus
      # uses the distilled student for every environment.
      actor.use_student = True
    return actor

  def save(self, path: str, infos=None):
    super().save(path, infos)
    # Large 1024-env recurrent validation runs can exhaust the CUDA graph
    # workspace when ONNX tracing is performed at iteration 0.  Validation
    # scripts may disable export while retaining the regular .pt checkpoint;
    # standalone ONNX smoke/export remains available by leaving this unset.
    if os.environ.get("GO2_SKIP_ONNX_EXPORT", "").lower() in {"1", "true", "yes"}:
      return
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name
        if self.logger.logger_type in ("wandb", "WandbLogWriter") and wandb.run
        else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      metadata.update(go2_policy_contract_metadata(self.alg.get_policy()))
      attach_metadata_to_onnx(str(onnx_path), metadata)
      # TS-Student also has a source-compatible recurrent artifact.  Keep the
      # regular two-input ONNX file above for mjlab callers and emit the
      # optional ``obs,h,c`` companion beside it when the actor supports it.
      try:
        recurrent_filename = f"{onnx_path.stem}_recurrent.onnx"
        recurrent_path = policy_dir / recurrent_filename
        if self.export_recurrent_policy_to_onnx(str(policy_dir), recurrent_filename):
          recurrent_metadata = dict(metadata)
          recurrent_metadata.update(
            go2_policy_contract_metadata(self.alg.get_policy(), recurrent=True)
          )
          attach_metadata_to_onnx(str(recurrent_path), recurrent_metadata)
      except Exception as recurrent_error:
        print(f"[WARN] recurrent ONNX export failed (training continues): {recurrent_error}")
      if (
        self.logger.logger_type in ("wandb", "WandbLogWriter")
        and self.cfg["upload_model"]
      ):
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")


class VelocityDistillationRunner(VelocityOnPolicyRunner):
  """Source-compatible recurrent TS student distillation runner.

  The legacy Go2 student jobs are not PPO jobs: they freeze a completed TS
  teacher, clone its actor into the student, and optimize a three-layer LSTM
  encoder plus the cloned actor against teacher latents and actions.  Reusing
  the ordinary on-policy collection loop would silently add PPO gradients and
  would reset the LSTM to a five-frame window at every action, so the two
  student task ids use this dedicated learning loop.
  """

  def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
    super().__init__(env, train_cfg, log_dir, device)
    from .go2_algorithms.models import StudentActorModel, _mlp

    self.student_actor = self.alg.get_policy()
    if not isinstance(self.student_actor, StudentActorModel):
      raise TypeError(
        "VelocityDistillationRunner requires StudentActorModel, got "
        f"{type(self.student_actor).__name__}"
      )

    self.teacher_terrain_encoder = _mlp(187, 16, (256, 128)).to(self.device)
    self.teacher_privileged_encoder = _mlp(74, 16, (128, 64)).to(self.device)
    self.teacher_actor = copy.deepcopy(self.student_actor.mlp).to(self.device)
    self.teacher_obs_normalizer = copy.deepcopy(
      self.student_actor.obs_normalizer
    ).to(self.device)
    self._load_teacher_checkpoint()
    # The source DistillPolicyRunner starts from an exact copy of the teacher
    # actor while leaving the recurrent encoder randomly initialized.
    self.student_actor.mlp.load_state_dict(self.teacher_actor.state_dict())
    self.student_actor.obs_normalizer = copy.deepcopy(
      self.teacher_obs_normalizer
    ).to(self.device)
    for module in (
      self.teacher_terrain_encoder,
      self.teacher_privileged_encoder,
      self.teacher_actor,
      self.teacher_obs_normalizer,
    ):
      module.eval()
      for parameter in module.parameters():
        parameter.requires_grad_(False)

    self.distill_optimizer = torch.optim.Adam(
      [
        *self.student_actor.student_lstm.parameters(),
        *self.student_actor.student_head.parameters(),
        *self.student_actor.mlp.parameters(),
      ],
      lr=1.0e-3,
    )
    # Go2AuxiliaryPPO persists this optimizer alongside the normal actor state.
    self.alg.student_distill_optimizer = self.distill_optimizer
    self._student_hidden: tuple[torch.Tensor, torch.Tensor] | None = None

  @staticmethod
  def _checkpoint_actor_state(path: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    state = loaded.get("actor_state_dict", loaded.get("model_state_dict", loaded))
    if not isinstance(state, dict):
      raise ValueError(f"Teacher checkpoint has no actor state dictionary: {path}")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
      name = key
      for prefix in ("actor.", "model."):
        if name.startswith(prefix):
          name = name[len(prefix) :]
      normalized[name] = value
    return normalized

  def _load_teacher_checkpoint(self) -> None:
    checkpoint = os.environ.get("GO2_TS_TEACHER_CHECKPOINT")
    if not checkpoint:
      print(
        "[WARN] TS-Student has no teacher checkpoint; set "
        "GO2_TS_TEACHER_CHECKPOINT for source-compatible distillation."
      )
      return
    path = Path(checkpoint)
    if not path.is_file():
      raise FileNotFoundError(f"GO2_TS_TEACHER_CHECKPOINT does not exist: {path}")
    state = self._checkpoint_actor_state(path)

    def select(prefix: str) -> dict[str, torch.Tensor]:
      return {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
      }

    required = {
      "terrain_encoder.": self.teacher_terrain_encoder,
      "privileged_encoder.": self.teacher_privileged_encoder,
      "mlp.": self.teacher_actor,
    }
    for prefix, module in required.items():
      selected = select(prefix)
      if not selected:
        raise ValueError(f"Teacher checkpoint {path} lacks {prefix[:-1]} weights")
      module.load_state_dict(selected, strict=True)
    normalizer = select("obs_normalizer.")
    if normalizer:
      # Checkpoints generated before source-normalization parity used RSL-RL's
      # running normalizer.  Preserve that teacher's exact input transform so
      # those checkpoints remain valid distillation sources; new source-parity
      # teacher checkpoints use Identity and have no normalizer state.
      try:
        self.teacher_obs_normalizer.load_state_dict(normalizer, strict=True)
      except RuntimeError:
        from rsl_rl.modules import EmpiricalNormalization

        self.teacher_obs_normalizer = EmpiricalNormalization(45).to(self.device)
        self.teacher_obs_normalizer.load_state_dict(normalizer, strict=True)

  def _teacher_step(
    self, obs: dict[str, torch.Tensor]
  ) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
      features = torch.cat(
        (
          self.teacher_terrain_encoder(obs["terrain"]),
          self.teacher_privileged_encoder(obs["privileged"]),
        ),
        dim=-1,
      )
      actor_obs = self.teacher_obs_normalizer(obs["actor"])
      actions = self.teacher_actor(torch.cat((features, actor_obs), dim=-1))
    return actions, features

  def _student_step(
    self, obs: dict[str, torch.Tensor]
  ) -> tuple[torch.Tensor, torch.Tensor]:
    actor_obs = self.student_actor.obs_normalizer(obs["actor"])
    batch_size = actor_obs.shape[0]
    if self._student_hidden is None:
      shape = (
        self.student_actor.student_lstm.num_layers,
        batch_size,
        self.student_actor.student_lstm.hidden_size,
      )
      self._student_hidden = (
        actor_obs.new_zeros(shape),
        actor_obs.new_zeros(shape),
      )
    encoded, self._student_hidden = self.student_actor.student_lstm(
      actor_obs[:, None, :], self._student_hidden
    )
    features = self.student_actor.student_head(encoded[:, -1])
    actions = self.student_actor.mlp(torch.cat((features, actor_obs), dim=-1))
    return actions, features

  def _reset_student_hidden(self, dones: torch.Tensor) -> None:
    if self._student_hidden is None:
      return
    keep = (~dones.to(torch.bool)).view(1, -1, 1)
    self._student_hidden = tuple(state * keep for state in self._student_hidden)  # type: ignore[assignment]

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )
    obs = self.env.get_observations().to(self.device)
    self.student_actor.train()
    self.logger.init_logging_writer()
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      collect_start = time.time()
      teacher_features: list[torch.Tensor] = []
      student_features: list[torch.Tensor] = []
      teacher_actions: list[torch.Tensor] = []
      student_actions: list[torch.Tensor] = []
      for _ in range(self.cfg["num_steps_per_env"]):
        target_actions, target_features = self._teacher_step(obs)
        actions, features = self._student_step(obs)
        teacher_actions.append(target_actions)
        teacher_features.append(target_features)
        student_actions.append(actions)
        student_features.append(features)
        # Match the source warm-up: iteration zero is driven by the teacher;
        # subsequent rollouts are driven by the learned recurrent student.
        env_actions = target_actions if it == 0 else actions.detach()
        obs, rewards, dones, extras = self.env.step(env_actions.to(self.env.device))
        obs, rewards, dones = (
          obs.to(self.device), rewards.to(self.device), dones.to(self.device)
        )
        self._reset_student_hidden(dones)
        self.logger.process_env_step(rewards, dones, extras, None)
      collect_time = time.time() - collect_start

      learn_start = time.time()
      latent_loss = torch.linalg.vector_norm(
        torch.cat(teacher_features).detach() - torch.cat(student_features), dim=-1
      ).mean()
      action_loss = torch.linalg.vector_norm(
        torch.cat(teacher_actions).detach() - torch.cat(student_actions), dim=-1
      ).mean()
      loss = latent_loss + action_loss
      self.distill_optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(
        [
          *self.student_actor.student_lstm.parameters(),
          *self.student_actor.student_head.parameters(),
        ],
        1.0,
      )
      nn.utils.clip_grad_norm_(self.student_actor.mlp.parameters(), 1.0)
      self.distill_optimizer.step()
      if self._student_hidden is not None:
        self._student_hidden = tuple(state.detach() for state in self._student_hidden)  # type: ignore[assignment]
      learn_time = time.time() - learn_start
      self.current_learning_iteration = it
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict={
          "ts_latent_distillation": float(latent_loss.detach()),
          "ts_action_distillation": float(action_loss.detach()),
        },
        learning_rate=self.distill_optimizer.param_groups[0]["lr"],
        action_std=(
          self.student_actor.distribution.std_param.clamp(
            *self.student_actor.distribution.std_range
          )
        ),
        rnd_weight=None,
      )
      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
    if self.logger.writer is not None:
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
      )
      self.logger.stop_logging_writer()

  def get_inference_policy(self, device: str | None = None):
    """Return the recurrent student path used by source student playback."""
    self.alg.eval_mode()
    actor = self.alg.get_policy().to(device)
    return _RecurrentStudentPolicy(actor, self.env)
