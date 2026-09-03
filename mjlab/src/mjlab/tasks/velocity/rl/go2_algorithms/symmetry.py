"""Source-compatible left/right symmetry transforms for Go2 policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

_ACTION_PERMUTATION = (-3, 4, 5, -0.0001, 1, 2, -9, 10, 11, -6, 7, 8)

_TROT_JUMP_OBS_PERMUTATION = (
  -0.0001, -1, 2, -3, -4, -5, 6, -7, -8, 9, -10,
  -14, 15, 16, -11, 12, 13, -20, 21, 22, -17, 18, 19,
  -26, 27, 28, -23, 24, 25, -32, 33, 34, -29, 30, 31,
  -38, 39, 40, -35, 36, 37, -44, 45, 46, -41, 42, 43,
)

_SPRING_JUMP_OBS_PERMUTATION = (
  -0.0001, -1, 2, -3, 4, -5, 6, -7, -8, 9, -10,
  -14, 15, 16, -11, 12, 13, -20, 21, 22, -17, 18, 19,
  -26, 27, 28, -23, 24, 25, -32, 33, 34, -29, 30, 31,
  -38, 39, 40, -35, 36, 37, -44, 45, 46, -41, 42, 43,
)

_HANDSTAND_OBS_PERMUTATION = (
  -0.0001, 1, -2, 3, -4, 5, 6, -7, -8,
  -12, 13, 14, -9, 10, 11, -18, 19, 20, -15, 16, 17,
  -24, 25, 26, -21, 22, 23, -30, 31, 32, -27, 28, 29,
  -36, 37, 38, -33, 34, 35, -42, 43, 44, -39, 40, 41,
)


def _mirror_tensor(
  value: torch.Tensor,
  permutation: Sequence[float],
  *,
  frame_stack: int = 1,
) -> torch.Tensor:
  """Apply the source signed permutation matrix to one or more frames."""
  frame_width = len(permutation)
  expected_width = frame_width * frame_stack
  if value.shape[-1] != expected_width:
    raise ValueError(
      f"Go2 symmetry expected width {expected_width}, got {value.shape[-1]}"
    )
  destination: list[int] = []
  signs: list[float] = []
  for frame in range(frame_stack):
    offset = frame * frame_width
    for item in permutation:
      destination.append(offset + int(abs(item)))
      signs.append(-1.0 if item < 0 else 1.0)
  mirrored = torch.empty_like(value)
  destination_tensor = torch.tensor(destination, device=value.device, dtype=torch.long)
  sign_tensor = torch.tensor(signs, device=value.device, dtype=value.dtype)
  mirrored[..., destination_tensor] = value * sign_tensor
  return mirrored


def _augment(
  *,
  obs: Any | None,
  actions: torch.Tensor | None,
  observation_permutation: Sequence[float],
  frame_stack: int,
) -> tuple[Any | None, torch.Tensor | None]:
  augmented_obs = None
  if obs is not None:
    mirrored_obs = obs.clone()
    mirrored_obs["actor"] = _mirror_tensor(
      obs["actor"], observation_permutation, frame_stack=frame_stack
    )
    augmented_obs = torch.cat((obs, mirrored_obs), dim=0)
  augmented_actions = None
  if actions is not None:
    mirrored_actions = _mirror_tensor(actions, _ACTION_PERMUTATION)
    augmented_actions = torch.cat((actions, mirrored_actions), dim=0)
  return augmented_obs, augmented_actions


def trot_jump_symmetry(
  *, env: Any, obs: Any | None, actions: torch.Tensor | None
) -> tuple[Any | None, torch.Tensor | None]:
  """Mirror the 10×47 observation used by Trot and Jump."""
  del env
  return _augment(
    obs=obs,
    actions=actions,
    observation_permutation=_TROT_JUMP_OBS_PERMUTATION,
    frame_stack=10,
  )


def spring_jump_symmetry(
  *, env: Any, obs: Any | None, actions: torch.Tensor | None
) -> tuple[Any | None, torch.Tensor | None]:
  """Mirror the 10×47 observation used by Spring-Jump."""
  del env
  return _augment(
    obs=obs,
    actions=actions,
    observation_permutation=_SPRING_JUMP_OBS_PERMUTATION,
    frame_stack=10,
  )


def handstand_symmetry(
  *, env: Any, obs: Any | None, actions: torch.Tensor | None
) -> tuple[Any | None, torch.Tensor | None]:
  """Mirror the 45-D observation used by Handstand."""
  del env
  return _augment(
    obs=obs,
    actions=actions,
    observation_permutation=_HANDSTAND_OBS_PERMUTATION,
    frame_stack=1,
  )
