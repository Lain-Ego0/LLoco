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
  elif action == "csv-convert":
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--robot", choices=("g1", "g1_23dof"), required=True)
    args = parser.parse_args()
    from .motion_preview import Preview

    preview = Preview()
    preview.update(source=Path(args.source).name, output=args.output)
    try:
      from lloco.motion_conversion import convert_csv_to_npz

      convert_csv_to_npz(
        args.robot,
        args.source,
        args.output,
        device="cpu",
        progress=lambda stage, processed, total: preview.update(
          stage=stage, processed=processed, total=total
        ),
      )
      preview.update(stage="complete")
    except Exception as error:
      preview.update(stage="failed", error=str(error))
      raise
  elif action in ("gmr-convert", "gmr-retarget"):
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from .motion_preview import Preview

    preview = Preview()
    try:
      if action == "gmr-retarget":
        from .retarget import retarget

        retarget(Path(args.source), Path(args.output), preview)
      else:
        import pickle

        import numpy as np

        from lloco.gmr_conversion import convert_gmr_to_npz

        with Path(args.source).open("rb") as stream:
          motion = pickle.load(stream)
        positions = np.asarray(motion["root_pos"])
        rotations = np.asarray(motion["root_rot"])
        joints = np.asarray(motion["dof_pos"])
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        if (
          positions.ndim != 2
          or positions.shape[1] != 3
          or rotations.shape != (len(positions), 4)
          or joints.shape != (len(positions), 29)
          or not np.isfinite(fps)
          or fps <= 0
          or len(positions) < 2
        ):
          raise ValueError("GMR PKL 需要有效帧率和 G1 29-DoF 动作")
        preview.update(
          stage="converting",
          total=len(positions),
          fps=fps,
          bones=[],
          parents=[],
          source=Path(args.source).name,
          output=args.output,
        )
        for index in range(len(positions)):
          if index % max(1, round(fps / 15)) == 0 or index == len(positions) - 1:
            qpos = np.concatenate(
              (positions[index], rotations[index, [3, 0, 1, 2]], joints[index])
            )
            preview.frame(index, fps, qpos)
            preview.update(processed=index + 1)
        convert_gmr_to_npz(args.source, args.output, device="cpu")
      preview.update(stage="complete")
    except Exception as error:
      preview.update(stage="failed", error=str(error))
      raise

  else:
    parser.error("Unknown worker action")


if __name__ == "__main__":
  main()
