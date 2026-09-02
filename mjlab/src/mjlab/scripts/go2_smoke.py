"""Short zero/random-action replay for migrated Go2 tasks.

This is intentionally finite and headless so it can be used as the stage-1
asset/environment smoke entry point without opening a viewer or running a
training job.
"""

import argparse

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task", nargs="?", default="Mjlab-Velocity-Flat-Unitree-Go2")
  parser.add_argument("--agent", choices=("zero", "random"), default="random")
  parser.add_argument("--steps", type=int, default=4)
  parser.add_argument("--num-envs", type=int, default=2)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401, PLC0415

  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    obs, _ = env.reset()
    for _ in range(args.steps):
      if args.agent == "zero":
        action = torch.zeros((args.num_envs, 12), device=env.device)
      else:
        action = 2.0 * torch.rand((args.num_envs, 12), device=env.device) - 1.0
      obs, reward, terminated, truncated, _ = env.step(action)
    print(
      f"{args.task}: agent={args.agent} steps={args.steps} "
      f"actor={tuple(obs['actor'].shape)} critic={tuple(obs['critic'].shape)} "
      f"reward={tuple(reward.shape)} terminated={tuple(terminated.shape)} "
      f"truncated={tuple(truncated.shape)}"
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()
