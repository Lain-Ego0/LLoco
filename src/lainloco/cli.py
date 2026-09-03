"""Command-line catalog and validation entry point for LainLoco."""

from __future__ import annotations

import argparse

from lainloco.robots.unitree.go2 import GO2, GO2_TASKS, GO2_TRAINING_PROFILES


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="lainloco", description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)

  robots = commands.add_parser("robots", help="inspect robot specifications")
  robot_commands = robots.add_subparsers(dest="robots_command", required=True)
  robot_commands.add_parser("list", help="list available robots")

  tasks = commands.add_parser("tasks", help="inspect robot skills")
  task_commands = tasks.add_subparsers(dest="tasks_command", required=True)
  tasks_list = task_commands.add_parser("list", help="list available tasks")
  tasks_list.add_argument("--robot", default="go2")

  profiles = commands.add_parser("profiles", help="inspect training profiles")
  profile_commands = profiles.add_subparsers(dest="profiles_command", required=True)
  profiles_list = profile_commands.add_parser("list", help="list training profiles")
  profiles_list.add_argument("--robot", default="go2")

  envs = commands.add_parser("envs", help="list registered mjlab environment IDs")
  envs.add_argument("--keyword", default="Go2")

  train = commands.add_parser(
    "train", help="train an explicit task/profile composition"
  )
  train.add_argument("task")
  train.add_argument("--profile", default="ppo")
  train.add_argument("--iterations", type=int)
  train.add_argument("--num-envs", type=int, default=4096)
  train.add_argument("--log-root", default="runs/rsl_rl")
  train.add_argument("--gpu-ids", default="0", help="comma-separated GPU ids, or 'cpu'")
  train.add_argument("--dry-run", action="store_true")

  play = commands.add_parser("play", help="play an explicit task/profile composition")
  play.add_argument("task")
  play.add_argument("--profile", default="ppo")
  play.add_argument("--agent", choices=("zero", "random", "trained"), default="trained")
  play.add_argument("--checkpoint")
  play.add_argument("--num-envs", type=int, default=1)
  play.add_argument("--device")
  play.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
  play.add_argument("--dry-run", action="store_true")

  distill = commands.add_parser(
    "distill", help="train a student from an explicit teacher checkpoint"
  )
  distill.add_argument("teacher_checkpoint")
  distill.add_argument("--task", default="go2/velocity-rough")
  distill.add_argument("--profile", default="ts-student")
  distill.add_argument("--iterations", type=int, default=20_000)
  distill.add_argument("--num-envs", type=int, default=1024)
  distill.add_argument("--log-root", default="runs/rsl_rl")
  distill.add_argument(
    "--gpu-ids",
    default="0",
    help="comma-separated GPU ids, or 'cpu'",
  )
  distill.add_argument("--dry-run", action="store_true")

  export = commands.add_parser(
    "export", help="export a checkpoint directly to a Policy Bundle"
  )
  export.add_argument("checkpoint")
  export.add_argument("--destination")
  export.add_argument("--task", required=True)
  export.add_argument("--profile", required=True)
  export.add_argument("--device", default="cpu")
  export.add_argument("--dry-run", action="store_true")

  bundle = commands.add_parser("bundle", help="create or validate a Policy Bundle")
  bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
  bundle_create = bundle_commands.add_parser("create", help="create a Policy Bundle")
  bundle_create.add_argument("policy", help="exported ONNX policy")
  bundle_create.add_argument("destination", help="new bundle directory")
  bundle_create.add_argument("--task", default="go2/velocity-rough")
  bundle_create.add_argument("--profile", default="ppo")
  bundle_create.add_argument("--normalization")
  bundle_validate = bundle_commands.add_parser(
    "validate", help="validate and inspect a Policy Bundle"
  )
  bundle_validate.add_argument("path")
  bundle_validate.add_argument("--task")
  bundle_validate.add_argument("--profile")
  bundle_rollout = bundle_commands.add_parser(
    "rollout", help="run a bundle in an independent mjlab control loop"
  )
  bundle_rollout.add_argument("path")
  bundle_rollout.add_argument("--steps", type=int, default=100)
  bundle_rollout.add_argument("--num-envs", type=int, default=1)
  bundle_rollout.add_argument("--device", default="cpu")

  validate = commands.add_parser("validate", help="run migration acceptance checks")
  validate_commands = validate.add_subparsers(dest="validate_command", required=True)
  asset = validate_commands.add_parser("asset", help="validate the Go2 asset")
  asset.add_argument("--task", default="Mjlab-Velocity-Flat-Unitree-Go2")
  asset.add_argument("--device", default="cpu")
  contracts = validate_commands.add_parser(
    "contracts", help="validate all source task contracts"
  )
  contracts.add_argument("--device", default="cpu")
  smoke = validate_commands.add_parser("smoke", help="run a finite rollout")
  smoke.add_argument("task", nargs="?", default="Mjlab-Velocity-Flat-Unitree-Go2")
  smoke.add_argument("--agent", choices=("zero", "random"), default="random")
  smoke.add_argument("--steps", type=int, default=4)
  smoke.add_argument("--num-envs", type=int, default=2)
  smoke.add_argument("--device", default="cpu")
  return parser


def main() -> None:
  args = _build_parser().parse_args()
  if args.command == "robots":
    print(f"{GO2.robot_id}\tactions={GO2.action_dim}\tdt={GO2.control_dt:g}")
    return
  if args.command == "tasks":
    if args.robot.lower() not in {"go2", "unitree/go2"}:
      raise SystemExit(f"Unknown robot: {args.robot}")
    for task in GO2_TASKS.values():
      print(f"{task.task_id}\t{task.family}\t{task.terrain_profile}")
    return
  if args.command == "profiles":
    if args.robot.lower() not in {"go2", "unitree/go2"}:
      raise SystemExit(f"Unknown robot: {args.robot}")
    for profile in GO2_TRAINING_PROFILES.values():
      print(f"{profile.profile_id}\t{profile.algorithm}")
    return
  if args.command == "envs":
    from mjlab.tasks.registry import list_tasks

    keyword = args.keyword.lower()
    for task_id in list_tasks():
      if not keyword or keyword in task_id.lower():
        print(task_id)
    return
  if args.command == "train":
    from lainloco.workflows import build_training_plan, launch_experiment_training

    gpu_ids = (
      None
      if args.gpu_ids.lower() == "cpu"
      else [int(value) for value in args.gpu_ids.split(",")]
    )
    plan = build_training_plan(
      task_id=args.task,
      profile_id=args.profile,
      iterations=args.iterations,
      num_envs=args.num_envs,
      log_root=args.log_root,
      gpu_ids=gpu_ids,
    )
    if args.dry_run:
      print(
        f"{plan.task_id}::{plan.profile_id}\t{plan.registry_task_id}\t"
        f"envs={plan.num_envs}\titerations={plan.iterations}"
      )
    else:
      launch_experiment_training(plan)
    return
  if args.command == "play":
    from lainloco.workflows import build_playback_plan, launch_playback

    plan = build_playback_plan(
      task_id=args.task,
      profile_id=args.profile,
      agent=args.agent,
      checkpoint=args.checkpoint,
      num_envs=args.num_envs,
      device=args.device,
      viewer=args.viewer,
    )
    if args.dry_run:
      print(
        f"{plan.task_id}::{plan.profile_id}\t{plan.registry_task_id}\t"
        f"agent={plan.agent}\tcheckpoint={plan.checkpoint}"
      )
    else:
      launch_playback(plan)
    return
  if args.command == "distill":
    from lainloco.workflows import build_distillation_plan, launch_distillation

    gpu_ids = (
      None
      if args.gpu_ids.lower() == "cpu"
      else [int(value) for value in args.gpu_ids.split(",")]
    )
    plan = build_distillation_plan(
      args.teacher_checkpoint,
      task_id=args.task,
      profile_id=args.profile,
      iterations=args.iterations,
      num_envs=args.num_envs,
      log_root=args.log_root,
      gpu_ids=gpu_ids,
    )
    if args.dry_run:
      print(
        f"{plan.task_id}::{plan.profile_id}\t{plan.registry_task_id}\t"
        f"teacher={plan.teacher_checkpoint}\tenvs={plan.num_envs}\t"
        f"iterations={plan.iterations}"
      )
    else:
      launch_distillation(plan)
    return
  if args.command == "export":
    from pathlib import Path

    from lainloco.workflows import (
      build_policy_export_plan,
      export_policy_bundle,
    )

    checkpoint = Path(args.checkpoint).expanduser()
    destination = (
      Path(args.destination).expanduser()
      if args.destination is not None
      else checkpoint.with_suffix("").with_name(checkpoint.stem + "-bundle")
    )
    plan = build_policy_export_plan(
      checkpoint,
      destination,
      task_id=args.task,
      profile_id=args.profile,
      device=args.device,
    )
    if args.dry_run:
      print(
        f"{plan.task_id}::{plan.profile_id}\t{plan.registry_task_id}\t"
        f"checkpoint={plan.checkpoint}\tdestination={plan.destination}"
      )
    else:
      loaded = export_policy_bundle(plan)
      print(
        f"{loaded.manifest.task_id}::{loaded.manifest.training_profile_id}\t"
        f"{loaded.root}"
      )
    return
  if args.command == "bundle":
    from lainloco.robots.unitree.go2.experiments import resolve_experiment
    from lainloco.runtime import (
      create_policy_bundle,
      load_policy_bundle,
    )

    if args.bundle_command == "create":
      binding = resolve_experiment(args.task, args.profile)
      loaded = create_policy_bundle(
        args.destination,
        args.policy,
        binding.experiment,
        normalization_path=args.normalization,
      )
    elif args.bundle_command == "validate":
      if (args.task is None) != (args.profile is None):
        raise SystemExit("--task and --profile must be supplied together")
      expected = (
        resolve_experiment(args.task, args.profile).experiment
        if args.task is not None
        else None
      )
      loaded = load_policy_bundle(args.path, expected_experiment=expected)
    elif args.bundle_command == "rollout":
      from lainloco.workflows import run_mjlab_bundle

      stats = run_mjlab_bundle(
        args.path,
        steps=args.steps,
        num_envs=args.num_envs,
        device=args.device,
      )
      print(
        f"steps={stats.control_steps}\tpolicy_calls={stats.policy_calls}\t"
        f"resets={stats.episode_resets}\tsimulated={stats.simulated_seconds:g}s"
      )
      return
    else:
      raise AssertionError(f"Unhandled bundle command: {args.bundle_command}")
    print(
      f"{loaded.manifest.robot_id}\t{loaded.manifest.task_id}::"
      f"{loaded.manifest.training_profile_id}\t"
      f"contract={loaded.contract.contract_version}\t{loaded.policy_path}"
    )
    return
  if args.command == "validate":
    from lainloco.validation import smoke, validate_asset, validate_contracts

    if args.validate_command == "asset":
      validate_asset(task_id=args.task, device=args.device)
    elif args.validate_command == "contracts":
      validate_contracts(device=args.device)
    elif args.validate_command == "smoke":
      smoke(
        args.task,
        agent=args.agent,
        steps=args.steps,
        num_envs=args.num_envs,
        device=args.device,
      )
    else:
      raise AssertionError(f"Unhandled validation command: {args.validate_command}")
    return
  raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
  main()
