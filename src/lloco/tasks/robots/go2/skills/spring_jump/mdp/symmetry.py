"""The source Spring-Jump signed mirror permutation, frame by frame."""

from typing import Any, cast

import torch
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.extensions import Symmetry
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

# Decode Go2_Spring_Jump_Config.py's negative-index notation.  ``-0.0001``
# is the source convention for index zero with a negative sign.
_INDEX = (
  0,
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  14,
  15,
  16,
  11,
  12,
  13,
  20,
  21,
  22,
  17,
  18,
  19,
  26,
  27,
  28,
  23,
  24,
  25,
  32,
  33,
  34,
  29,
  30,
  31,
  38,
  39,
  40,
  35,
  36,
  37,
  44,
  45,
  46,
  41,
  42,
  43,
)
_SIGN = (
  -1,
  -1,
  1,
  -1,
  1,
  -1,
  1,
  -1,
  -1,
  1,
  -1,
  -1,
  1,
  1,
  -1,
  1,
  1,
  -1,
  1,
  1,
  -1,
  1,
  1,
  -1,
  1,
  1,
  -1,
  1,
  1,
  -1,
  1,
  1,
  -1,
  1,
  1,
  -1,
  1,
  1,
  -1,
  1,
  1,
  -1,
  1,
  1,
  -1,
  1,
  1,
)
_ACTION_INDEX = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
_ACTION_SIGN = (-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1)


def spring_jump_symmetry(
  env, obs: TensorDict | None = None, actions: torch.Tensor | None = None
):
  del env
  out_obs = None
  if obs is not None:
    mirrored = obs.clone()
    frames = obs["actor"].reshape(-1, 10, 47)
    index = torch.tensor(_INDEX, device=frames.device)
    sign = torch.tensor(_SIGN, device=frames.device, dtype=frames.dtype)
    mirrored["actor"] = (frames[:, :, index] * sign).reshape(-1, 470)
    out_obs = cast(TensorDict, TensorDict.cat((obs, mirrored), dim=0))
  out_actions = None
  if actions is not None:
    index = torch.tensor(_ACTION_INDEX, device=actions.device)
    sign = torch.tensor(_ACTION_SIGN, device=actions.device, dtype=actions.dtype)
    out_actions = torch.cat((actions, actions[:, index] * sign), dim=0)
  return out_obs, out_actions


class SourceMirrorSymmetry(Symmetry):
  """Mirror loss with gradients through both branches, as in the source PPO."""

  def compute_loss(
    self, actor: MLPModel, batch: RolloutStorage.Batch, original_batch_size: int
  ) -> torch.Tensor:
    if not self.use_data_augmentation:
      batch.observations, _ = self.data_augmentation_func(
        env=self.env, obs=batch.observations, actions=None
      )

    assert batch.observations is not None
    mean_actions = actor(batch.observations.detach().clone())
    _, mirrored_original_actions = self.data_augmentation_func(
      env=self.env,
      obs=None,
      actions=mean_actions[:original_batch_size],
    )
    assert mirrored_original_actions is not None
    loss = F.mse_loss(
      mean_actions[original_batch_size:],
      mirrored_original_actions[original_batch_size:],
    )
    return loss if self.use_mirror_loss else loss.detach()


class SourceSymmetricPPO(PPO):
  """RSL-RL PPO using the old fork's bidirectional mirror-loss gradient."""

  def __init__(
    self,
    *args: Any,
    symmetry_cfg: dict[str, Any] | None = None,
    **kwargs: Any,
  ) -> None:
    super().__init__(*args, symmetry_cfg=None, **kwargs)
    self.symmetry = SourceMirrorSymmetry(**symmetry_cfg) if symmetry_cfg else None
