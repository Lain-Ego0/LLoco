"""Source-exact symmetry transforms for Go2 skill policies."""

from typing import Any, cast

import torch
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.extensions import Symmetry
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

# Signed permutations copied from Gym's Go2_Handstand_PPO_Yu. Each output column
# selects the absolute source index and applies the encoded sign.
_REAR_STAND_OBS_INDEX = (
  0, 1, 2,
  3, 4, 5,
  6, 7, 8,
  12, 13, 14, 9, 10, 11, 18, 19, 20, 15, 16, 17,
  24, 25, 26, 21, 22, 23, 30, 31, 32, 27, 28, 29,
  36, 37, 38, 33, 34, 35, 42, 43, 44, 39, 40, 41,
)
_REAR_STAND_OBS_SIGN = (
  -1, 1, -1,
  1, -1, 1,
  1, -1, -1,
  -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1,
  -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1,
  -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1,
)
_REAR_STAND_ACTION_INDEX = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
_REAR_STAND_ACTION_SIGN = (-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1)


def _mirror(
  value: torch.Tensor, indices: tuple[int, ...], signs: tuple[int, ...]
) -> torch.Tensor:
  index = torch.tensor(indices, device=value.device)
  sign = torch.tensor(signs, dtype=value.dtype, device=value.device)
  return value[..., index] * sign


def rear_stand_symmetry(
  env,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Append the left-right mirror used by the Isaac Gym Handstand PPO."""
  del env
  augmented_obs = None
  if obs is not None:
    mirrored_obs = obs.clone()
    mirrored_obs["actor"] = _mirror(
      obs["actor"], _REAR_STAND_OBS_INDEX, _REAR_STAND_OBS_SIGN
    )
    augmented_obs = cast(TensorDict, TensorDict.cat((obs, mirrored_obs), dim=0))

  augmented_actions = None
  if actions is not None:
    mirrored_actions = _mirror(
      actions, _REAR_STAND_ACTION_INDEX, _REAR_STAND_ACTION_SIGN
    )
    augmented_actions = torch.cat((actions, mirrored_actions), dim=0)

  return augmented_obs, augmented_actions


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
