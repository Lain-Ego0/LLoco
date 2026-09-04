"""Exact signed-permutation checks for the RearStand PPO symmetry loss."""

import torch
from tensordict import TensorDict

from lloco.tasks.go2_skills.mdp.symmetry import rear_stand_symmetry


def test_rear_stand_symmetry_matches_source_permutations_and_is_involution() -> None:
  actor = torch.arange(90, dtype=torch.float32).reshape(2, 45)
  critic = torch.arange(172, dtype=torch.float32).reshape(2, 86)
  actions = torch.arange(24, dtype=torch.float32).reshape(2, 12)
  obs = TensorDict({"actor": actor, "critic": critic}, batch_size=[2])

  augmented_obs, augmented_actions = rear_stand_symmetry(None, obs, actions)
  assert augmented_obs is not None
  assert augmented_actions is not None
  assert augmented_obs.batch_size == torch.Size([4])
  assert augmented_actions.shape == (4, 12)
  assert torch.equal(augmented_obs["actor"][:2], actor)
  assert torch.equal(augmented_obs["critic"][:2], critic)
  assert torch.equal(augmented_obs["critic"][2:], critic)
  assert torch.equal(augmented_obs["actor"][2:, :3], actor[:, :3] * actor.new_tensor([-1, 1, -1]))
  assert torch.equal(augmented_obs["actor"][2:, 9:12], actor[:, 12:15] * actor.new_tensor([-1, 1, 1]))
  assert torch.equal(augmented_actions[2:, :3], actions[:, 3:6] * actions.new_tensor([-1, 1, 1]))

  mirrored_obs = TensorDict(
    {"actor": augmented_obs["actor"][2:], "critic": critic}, batch_size=[2]
  )
  mirrored_actions = augmented_actions[2:]
  twice_obs, twice_actions = rear_stand_symmetry(
    None, mirrored_obs, mirrored_actions
  )
  assert twice_obs is not None
  assert twice_actions is not None
  assert torch.equal(twice_obs["actor"][2:], actor)
  assert torch.equal(twice_actions[2:], actions)
