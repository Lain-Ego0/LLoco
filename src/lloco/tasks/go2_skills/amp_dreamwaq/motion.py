"""Go2 AMP expert-transition loader, translated from the source AMPLoader."""

import json
from pathlib import Path

import torch


class Go2AmpMotionLoader:
  """Weighted, preloaded 31-field AMP state transitions at the policy dt."""

  observation_dim = 31

  def __init__(self, device: torch.device, dt: float, preload: int = 2_000_000) -> None:
    self.device, self.dt = device, dt
    root = Path(__file__).parents[3] / "assets" / "motions" / "go2_amp"
    frames, frame_durations, weights = [], [], []
    for path in sorted(root.glob("*.txt")):
      payload = json.loads(path.read_text())
      data = torch.tensor(payload["Frames"], dtype=torch.float32, device=device)
      frames.append(data)
      frame_durations.append(float(payload["FrameDuration"]))
      weights.append(float(payload["MotionWeight"]))
    if not frames:
      raise FileNotFoundError(f"No Go2 AMP motion files found in {root}")
    self.frames = frames
    self.frame_durations = frame_durations
    self.weights = torch.tensor(weights, device=device)
    self.weights /= self.weights.sum()
    # Exact source uses two million samples.  Retain that default for training;
    # tests can select a smaller preload explicitly.
    # Source uses NumPy's CPU weighted sampler.  CUDA multinomial rejects this
    # two-million-element launch on some consumer GPUs, so retain the source
    # sampling domain and transfer only the sampled indices.
    motion_ids = torch.multinomial(self.weights.cpu(), preload, replacement=True).to(device)
    self.states = torch.empty((preload, 31), device=device)
    self.next_states = torch.empty_like(self.states)
    # Sampling stays within a motion: concatenating motions would create false
    # expert transitions at file boundaries.
    for index, frame in enumerate(self.frames):
      mask = motion_ids == index
      count = int(mask.sum())
      if count:
        duration = self.frame_durations[index]
        trajectory_length = (frame.shape[0] - 1) * duration
        # Match AMPLoader exactly: sample continuous motion time, then evaluate
        # the motion at t and t + policy_dt.  Motion files are 25 Hz (0.04 s),
        # while the policy is 50 Hz (0.02 s), so adjacent-frame pairing is not
        # a valid source transition.
        times = torch.clamp(
          torch.rand(count, device=device) * trajectory_length - (self.dt + duration),
          min=0.0,
        )
        self.states[mask] = self._state_at_time(frame, times, trajectory_length)
        self.next_states[mask] = self._state_at_time(
          frame, times + self.dt, trajectory_length
        )

  def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.randint(0, self.states.shape[0], (count,), device=self.device)
    return self.states[ids], self.next_states[ids]

  @staticmethod
  def _state_at_time(
    frames: torch.Tensor, times: torch.Tensor, trajectory_length: float
  ) -> torch.Tensor:
    phase = times / trajectory_length * frames.shape[0]
    low = torch.floor(phase).long().clamp_max(frames.shape[0] - 1)
    high = torch.ceil(phase).long().clamp_max(frames.shape[0] - 1)
    blend = (phase - low).unsqueeze(1)
    data = frames[low] * (1.0 - blend) + frames[high] * blend
    # Source discriminator state: q, body linear/angular velocity, dq,
    # terrain-relative root height.  Quaternion and toe fields are omitted.
    return torch.cat(
      (data[:, 7:19], data[:, 31:37], data[:, 37:49], data[:, 2:3]), dim=1
    )
