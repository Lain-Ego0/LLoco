#!/usr/bin/env python3
"""Run a LainLoco Policy Bundle in the mjlab simulation backend."""

from __future__ import annotations

import argparse

from lainloco.workflows import run_mjlab_bundle


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("bundle", help="path to a validated LainLoco Policy Bundle")
  parser.add_argument("--steps", type=int, default=100)
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  stats = run_mjlab_bundle(
    args.bundle,
    steps=args.steps,
    num_envs=args.num_envs,
    device=args.device,
  )
  print(
    f"steps={stats.control_steps}\tpolicy_calls={stats.policy_calls}\t"
    f"resets={stats.episode_resets}\tsimulated={stats.simulated_seconds:g}s"
  )


if __name__ == "__main__":
  main()
