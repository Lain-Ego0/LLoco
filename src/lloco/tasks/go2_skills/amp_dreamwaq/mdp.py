"""AMP-DreamWaQ Manager-MDP additions."""

import torch
from mjlab.entity import Entity
from mjlab.managers import RecorderTerm

from ..shared.contacts import JOINT_NAMES


def amp_state(env) -> torch.Tensor:
  """Exact source AMP state: q, body v, body omega, dq, terrain-relative z."""
  robot: Entity = env.scene["robot"]
  ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
  if tuple(names) != JOINT_NAMES:
    raise RuntimeError(f"Go2 joint order mismatch: {names}")
  scan = env.scene["terrain_scan"].data
  terrain_z = scan.hit_pos_w[..., 2].mean(1, keepdim=True)
  return torch.cat((
    robot.data.joint_pos[:, ids],
    robot.data.root_link_lin_vel_b,
    robot.data.root_link_ang_vel_b,
    robot.data.joint_vel[:, ids],
    robot.data.root_link_pos_w[:, 2:3] - terrain_z,
  ), dim=1)


def rear_hip_limit(env) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids, names = robot.find_joints(("RL_hip_joint", "RR_hip_joint"), preserve_order=True)
  if tuple(names) != ("RL_hip_joint", "RR_hip_joint"):
    raise RuntimeError(f"Go2 rear hip joint mismatch: {names}")
  q = robot.data.joint_pos[:, ids]
  return ((q > .4) | (q < -.4)).any(dim=1).float()


class AmpTerminalStateRecorder(RecorderTerm):
  """Expose the pre-reset AMP state to the on-policy algorithm.

  Mjlab auto-reset observations contain the next episode's initial state.  The
  recorder hook runs before reset and preserves the terminal state reached by
  the action, matching the Gym AMP runner's ``terminal_amp_states`` path.
  """

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    self._env.extras["amp_terminal_env_ids"] = env_ids.clone()
    self._env.extras["amp_terminal_states"] = amp_state(self._env)[env_ids].clone()

  def record_post_step(self) -> None:
    if not torch.any(self._env.reset_buf):
      self._env.extras.pop("amp_terminal_env_ids", None)
      self._env.extras.pop("amp_terminal_states", None)
