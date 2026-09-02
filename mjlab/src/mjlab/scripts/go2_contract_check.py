"""Headless reset/step and observation-contract check for all migrated Go2 tasks.

The check deliberately performs one environment reset and one zero-action step per
source task. It verifies task registration, construction, the 12-action interface,
and actor/critic dimensions without becoming a training or performance benchmark.
"""

from __future__ import annotations

import argparse

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

_SOURCE_TASKS: tuple[tuple[str, int, int], ...] = (
  ("Mjlab-Trot-Flat-Unitree-Go2", 470, 204),
  ("Mjlab-Jump-Flat-Unitree-Go2", 470, 210),
  ("Mjlab-Spring-Jump-Flat-Unitree-Go2", 470, 195),
  ("Mjlab-Backflip-Flat-Unitree-Go2", 470, 150),
  ("Mjlab-Handstand-Flat-Unitree-Go2", 45, 86),
  ("Mjlab-Leggedstand-Flat-Unitree-Go2", 45, 86),
  ("Mjlab-DreamWaQ-Rough-Unitree-Go2", 45, 783),
  ("Mjlab-AMP-DreamWaQ-Rough-Unitree-Go2", 45, 783),
  ("Mjlab-CTS-Rough-Unitree-Go2", 45, 278),
  ("Mjlab-AMP-CTS-Rough-Unitree-Go2", 45, 281),
  ("Mjlab-AMP-TS-Rough-Unitree-Go2", 45, 309),
  ("Mjlab-AMP-TS-Student-Rough-Unitree-Go2", 45, 309),
  ("Mjlab-TS-Rough-Unitree-Go2", 45, 309),
  ("Mjlab-TS-Student-Rough-Unitree-Go2", 45, 309),
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  # Registration is performed by the package import, matching the normal CLI.
  import mjlab.tasks  # noqa: F401, PLC0415

  for task, expected_actor, expected_critic in _SOURCE_TASKS:
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
      obs, _ = env.reset()
      if env.action_manager.total_action_dim != 12:
        raise RuntimeError(
          f"{task}: expected 12 actions, got {env.action_manager.total_action_dim}"
        )
      action = torch.zeros((1, 12), device=env.device)
      obs, _reward, _terminated, _truncated, _info = env.step(action)
      actor_dim = int(obs["actor"].shape[-1])
      critic_dim = int(obs["critic"].shape[-1])
      if (actor_dim, critic_dim) != (expected_actor, expected_critic):
        raise RuntimeError(
          f"{task}: expected actor/critic {(expected_actor, expected_critic)}, "
          f"got {(actor_dim, critic_dim)}"
        )
      print(f"{task}: actions=12 actor={actor_dim} critic={critic_dim} OK")
    finally:
      env.close()

  print(f"Go2 contract check passed: {len(_SOURCE_TASKS)} source tasks")


if __name__ == "__main__":
  main()
