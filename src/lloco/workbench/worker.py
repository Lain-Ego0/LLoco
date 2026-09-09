"""Short-lived adapters to the existing training and playback interfaces."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def export(task, checkpoint, motion):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import lloco.tasks  # noqa: F401

  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  if motion:
    cfg.commands["motion"].motion_file = motion
  agent = load_rl_cfg(task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = cls(wrapped, asdict(agent), device="cpu")
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cpu")
    target = Path(checkpoint).parent / "exported" / Path(checkpoint).stem
    runner.export_policy_to_onnx(str(target), "policy.onnx")
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

    attach_metadata_to_onnx(
      str(target / "policy.onnx"), get_base_metadata(env, "workbench")
    )
    (target / "export.json").write_text(
      json.dumps(
        {
          "task": task,
          "checkpoint": str(Path(checkpoint).resolve()),
          "motion": motion,
          "device": "cpu",
          "policy": "policy.onnx",
        },
        indent=2,
      )
    )
    print(f"ONNX exported: {target / 'policy.onnx'}", flush=True)
  finally:
    env.close()


def main():
  action = sys.argv.pop(1)
  if action == "train":
    from lloco.cli import train

    train()
    return
  parser = argparse.ArgumentParser()
  if action in ("export", "viser"):
    parser.add_argument("task")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion")
    args = parser.parse_args()
    if action == "export":
      export(args.task, args.checkpoint, args.motion)
    else:
      from lloco.cli import play

      sys.argv = [
        sys.argv[0],
        args.task,
        "--checkpoint-file",
        args.checkpoint,
        "--viewer",
        "viser",
        "--num-envs",
        "1",
        "--device",
        "cpu",
      ]
      if args.motion:
        sys.argv += ["--motion-file", args.motion]
      play()
  elif action in ("gmr-convert", "gmr-retarget"):
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if action == "gmr-retarget":
      from .retarget import retarget

      retarget(Path(args.source), Path(args.output))
    else:
      from lloco.gmr_conversion import convert_gmr_to_npz

      convert_gmr_to_npz(args.source, args.output, device="cpu")
  else:
    parser.error("Unknown worker action")


if __name__ == "__main__":
  main()
