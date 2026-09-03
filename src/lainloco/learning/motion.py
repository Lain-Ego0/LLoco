"""AMP motion-trajectory loader for LainLoco tasks.

The source project stores trajectories as JSON documents with a ``.txt``
extension.  This loader intentionally keeps the format simple and dependency
free: it reads the source frames, normalizes quaternions, and samples paired
AMP observations for the discriminator.  The supplied files already use the
MJCF order, while ``joint_permutation`` supports explicit conversion for a
checkpoint or asset with another order.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# The source Isaac-Gym asset and migrated MJCF currently use this order.
# Keeping names here makes any future checkpoint/asset layout conversion
# explicit instead of relying on an undocumented slice convention.
GO2_JOINT_NAMES = (
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
)


def joint_permutation(
  source_names: Sequence[str], target_names: Sequence[str] = GO2_JOINT_NAMES
) -> tuple[int, ...]:
  """Return source indices that produce ``target_names`` ordering."""
  source = tuple(source_names)
  target = tuple(target_names)
  if len(source) != len(target) or len(set(source)) != len(source):
    raise ValueError(
      "source and target joint layouts must contain unique equal-sized names"
    )
  missing = [name for name in target if name not in source]
  if missing or len(set(target)) != len(target):
    raise ValueError(f"target joint layout is incompatible; missing={missing}")
  return tuple(source.index(name) for name in target)


@dataclass(frozen=True)
class Go2MotionTrajectory:
  """One source trajectory and its sampling metadata."""

  frames: torch.Tensor
  frame_duration: float
  weight: float
  name: str

  @property
  def duration(self) -> float:
    return max(0.0, (self.frames.shape[0] - 1) * self.frame_duration)


class Go2MotionLoader:
  """Load and sample Go2 AMP trajectories from source JSON files.

  A raw frame has 49 values in the source layout::

    root position (3), root quaternion xyzw (4), joint position (12),
    local foot position (12), base linear velocity (3), base angular
    velocity (3), and joint velocity (12).

  The discriminator input follows ``LeggedRobot.get_amp_observations`` and is
  therefore 31-dimensional: joint position, base linear/angular velocity,
  joint velocity, and root height.
  """

  RAW_FRAME_DIM = 49
  AMP_OBS_DIM = 31
  SOURCE_TRANSITION_DT = 0.02
  _ROOT_POS = slice(0, 3)
  _ROOT_QUAT = slice(3, 7)
  _JOINT_POS = slice(7, 19)
  _BASE_LIN_VEL = slice(31, 34)
  _BASE_ANG_VEL = slice(34, 37)
  _JOINT_VEL = slice(37, 49)

  def __init__(
    self,
    motion_files: Sequence[str | Path],
    device: str | torch.device = "cpu",
    joint_permutation: Sequence[int] | None = None,
    time_between_frames: float = SOURCE_TRANSITION_DT,
  ) -> None:
    self.device = torch.device(device)
    if time_between_frames <= 0.0:
      raise ValueError("time_between_frames must be positive")
    self._time_between_frames = float(time_between_frames)
    permutation = self._validate_permutation(joint_permutation)
    self.trajectories = tuple(
      self._load_file(Path(path), permutation) for path in motion_files
    )
    if not self.trajectories:
      raise ValueError("Go2MotionLoader requires at least one motion file")
    weights = torch.tensor(
      [trajectory.weight for trajectory in self.trajectories], dtype=torch.float32
    )
    if not torch.isfinite(weights).all() or float(weights.sum()) <= 0.0:
      weights = torch.ones_like(weights)
    self.weights = weights / weights.sum()

  @classmethod
  def from_directory(
    cls,
    directory: str | Path,
    device: str | torch.device = "cpu",
    joint_permutation: Sequence[int] | None = None,
    time_between_frames: float = SOURCE_TRANSITION_DT,
  ) -> "Go2MotionLoader":
    """Load all JSON-like trajectory files in a directory."""
    paths = sorted(Path(directory).glob("*.txt"))
    if not paths:
      paths = sorted(Path(directory).glob("*.json"))
    return cls(
      paths,
      device=device,
      joint_permutation=joint_permutation,
      time_between_frames=time_between_frames,
    )

  @staticmethod
  def _validate_permutation(
    permutation: Sequence[int] | None,
  ) -> tuple[int, ...] | None:
    if permutation is None:
      return None
    result = tuple(int(index) for index in permutation)
    if sorted(result) != list(range(12)):
      raise ValueError("joint_permutation must be a permutation of indices 0..11")
    return result

  @staticmethod
  def _foot_leg_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    """Derive a four-leg permutation from a contiguous 3-DOF layout."""
    legs: list[int] = []
    for target_leg in range(4):
      source_indices = permutation[target_leg * 3 : target_leg * 3 + 3]
      source_leg = source_indices[0] // 3
      if source_indices != tuple(source_leg * 3 + offset for offset in range(3)):
        raise ValueError(
          "joint_permutation must preserve each leg's hip/thigh/calf triplet"
        )
      legs.append(source_leg)
    return tuple(legs)

  @staticmethod
  def _load_file(
    path: Path, permutation: tuple[int, ...] | None = None
  ) -> Go2MotionTrajectory:
    with path.open(encoding="utf-8") as stream:
      payload = json.load(stream)
    frames = np.asarray(payload["Frames"], dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != Go2MotionLoader.RAW_FRAME_DIM:
      raise ValueError(
        f"{path} has shape {frames.shape}; expected [N, {Go2MotionLoader.RAW_FRAME_DIM}]"
      )
    # Normalize each quaternion and choose a consistent hemisphere before
    # interpolation, avoiding sign flips between adjacent source frames.
    quaternions = frames[:, Go2MotionLoader._ROOT_QUAT]
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
      raise ValueError(f"{path} contains a zero root quaternion")
    quaternions /= norms
    quaternions[quaternions[:, 3] < 0.0] *= -1.0
    if permutation is not None:
      # Convert positions, local foot positions and velocities together;
      # root pose/velocity fields are independent of joint layout.
      frames[:, 7:19] = frames[:, 7:19][:, permutation]
      leg_permutation = Go2MotionLoader._foot_leg_permutation(permutation)
      frames[:, 19:31] = (
        frames[:, 19:31].reshape(-1, 4, 3)[:, leg_permutation].reshape(-1, 12)
      )
      frames[:, 37:49] = frames[:, 37:49][:, permutation]
    return Go2MotionTrajectory(
      frames=torch.from_numpy(frames),
      frame_duration=float(payload["FrameDuration"]),
      weight=float(payload.get("MotionWeight", 1.0)),
      name=path.stem,
    )

  @property
  def num_motions(self) -> int:
    return len(self.trajectories)

  @property
  def observation_dim(self) -> int:
    return self.AMP_OBS_DIM

  @property
  def time_between_frames(self) -> float:
    """Source environment control period between AMP transition states."""
    return self._time_between_frames

  def _sample_indices(self, batch_size: int) -> torch.Tensor:
    return torch.multinomial(self.weights, batch_size, replacement=True)

  @staticmethod
  def _interpolate(
    frames: torch.Tensor, times: torch.Tensor, duration: float
  ) -> torch.Tensor:
    if frames.shape[0] == 1 or duration <= 0.0:
      return frames[:1].expand(times.shape[0], -1)
    # Preserve the legacy loader's indexing convention.  It multiplies the
    # normalized trajectory time by N (not N-1) and reserves one source frame
    # at the tail when sampling, so expert transitions retain the exact source
    # phase distribution.
    positions = (times.clamp(0.0, duration) / duration) * frames.shape[0]
    low = positions.floor().long().clamp(max=frames.shape[0] - 1)
    high = (low + 1).clamp(max=frames.shape[0] - 1)
    blend = (positions - low).unsqueeze(-1)
    result = torch.lerp(frames[low], frames[high], blend)
    quat = result[:, Go2MotionLoader._ROOT_QUAT]
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    result[:, Go2MotionLoader._ROOT_QUAT] = quat
    return result

  @classmethod
  def amp_observation_from_frame(cls, frames: torch.Tensor) -> torch.Tensor:
    """Convert raw source frames to the 31-D discriminator input."""
    return torch.cat(
      (
        frames[..., cls._JOINT_POS],
        frames[..., cls._BASE_LIN_VEL],
        frames[..., cls._BASE_ANG_VEL],
        frames[..., cls._JOINT_VEL],
        frames[..., cls._ROOT_POS.start + 2 : cls._ROOT_POS.start + 3],
      ),
      dim=-1,
    )

  def sample_transition(
    self,
    batch_size: int,
    time_between_frames: float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``(state, next_state)`` AMP observations on the loader device."""
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    step = (
      self.time_between_frames
      if time_between_frames is None
      else float(time_between_frames)
    )
    indices = self._sample_indices(batch_size)
    times = torch.empty(batch_size, dtype=torch.float32).uniform_(0.0, 1.0)
    current: list[tuple[torch.Tensor, torch.Tensor]] = []
    following: list[tuple[torch.Tensor, torch.Tensor]] = []
    for trajectory_index, trajectory in enumerate(self.trajectories):
      mask = indices == trajectory_index
      if not mask.any():
        continue
      # ``traj_time_sample_batch`` in the source subtracts both the transition
      # step and one native frame duration.  This keeps ceil(p*N) in range for
      # its N-based interpolation convention.
      sampled = (
        times[mask] * trajectory.duration - step - trajectory.frame_duration
      ).clamp_min(0.0)
      frames = trajectory.frames
      current.append((mask, self._interpolate(frames, sampled, trajectory.duration)))
      following.append(
        (mask, self._interpolate(frames, sampled + step, trajectory.duration))
      )
    current_frames = torch.empty((batch_size, self.RAW_FRAME_DIM), dtype=torch.float32)
    next_frames = torch.empty_like(current_frames)
    for mask, values in current:
      current_frames[mask] = values
    for mask, values in following:
      next_frames[mask] = values
    return (
      self.amp_observation_from_frame(current_frames).to(self.device),
      self.amp_observation_from_frame(next_frames).to(self.device),
    )
