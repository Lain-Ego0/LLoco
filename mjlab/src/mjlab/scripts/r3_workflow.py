"""R3 schema capture, trained play recording, and independent evaluation."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

TASK_ID = "robolab.motion.velocity.flat"
TASK_VERSION = "1.0.0"
PROFILE_ID = "community.firedog2_2"
PROFILE_VERSION = "1.0.0"
THRESHOLD_VERSION = "r3.velocity.flat.thresholds@1.0.0"


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _canonical(value: Any) -> str:
  return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_obj(value: Any) -> str:
  return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _jsonable(value: Any) -> Any:
  if dataclasses.is_dataclass(value):
    return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  if callable(value):
    return f"{value.__module__}.{value.__qualname__}"
  return repr(value)


def _git_revision(root: Path) -> str:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def _import_task():
  import mjlab.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  return load_env_cfg, load_rl_cfg


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def capture_schema(output: Path, device: str, num_envs: int, seed: int) -> dict[str, Any]:
  from mjlab.envs import ManagerBasedRlEnv

  load_env_cfg, _ = _import_task()
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    observations, _ = env.reset()
    fields: list[dict[str, Any]] = []
    for group, names in env.observation_manager.active_terms.items():
      tensor = observations[group]
      assert isinstance(tensor, torch.Tensor)
      offset = 0
      for order, name in enumerate(names):
        cfg_term = env.observation_manager.get_term_cfg(group, name)
        dims = env.observation_manager.group_obs_term_dim[group][order]
        width = int(np.prod(dims))
        raw_source = cfg_term.func
        field = {
          "stableName": f"{group}.{name}",
          "shape": list(dims),
          "dtype": str(tensor.dtype).replace("torch.", ""),
          "unit": "mixed_si_or_normalized",
          "frame": "base" if name not in {"command", "actions", "foot_contact"} else "task",
          "history": {
            "length": cfg_term.history_length,
            "flattened": cfg_term.flatten_history_dim,
          },
          "scale": _jsonable(cfg_term.scale) if cfg_term.scale is not None else 1.0,
          "clipping": list(cfg_term.clip) if cfg_term.clip is not None else None,
          "source": f"{raw_source.__module__}.{raw_source.__qualname__}",
          "order": order,
          "flatOffset": [offset, offset + width],
          "task": f"{TASK_ID}@{TASK_VERSION}",
          "profile": f"{PROFILE_ID}@{PROFILE_VERSION}",
        }
        field["hash"] = _hash_obj(field)
        fields.append(field)
        offset += width
      if tensor.ndim != 2 or tensor.shape[1] != offset:
        raise RuntimeError(f"Observation schema mismatch for {group}: {tensor.shape}")
      if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite {group} observation")

    observation = {
      "schema": "robolab.observation-schema.v1",
      "id": "robolab.motion.velocity.flat.observation",
      "version": "1.0.0",
      "task": f"{TASK_ID}@{TASK_VERSION}",
      "profile": f"{PROFILE_ID}@{PROFILE_VERSION}",
      "runtime": {
        "device": device,
        "numEnvs": num_envs,
        "seed": seed,
        "controlFrequencyHz": 1.0 / env.step_dt,
      },
      "groups": {
        group: {
          "shape": list(observations[group].shape[1:]),
          "dtype": str(observations[group].dtype).replace("torch.", ""),
          "finite": bool(torch.isfinite(observations[group]).all()),
          "fieldNames": [field["stableName"] for field in fields if field["stableName"].startswith(group + ".")],
        }
        for group in env.observation_manager.active_terms
      },
      "fields": fields,
    }
    observation["hash"] = _hash_obj(observation)

    action_fields = []
    offset = 0
    for order, (name, dim) in enumerate(
      zip(env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True)
    ):
      term = env.action_manager.get_term(name)
      scale = getattr(term, "_scale", 1.0)
      if isinstance(scale, torch.Tensor):
        scale = scale[0].detach().cpu().tolist()
      field = {
        "stableName": name,
        "shape": [dim],
        "dtype": str(env.action_manager.action.dtype).replace("torch.", ""),
        "unit": "normalized_policy_action",
        "frame": "joint_order",
        "history": {"length": 0, "flattened": True},
        "scale": _jsonable(scale),
        "clipping": list(term.cfg.clip.values())[0] if term.cfg.clip else None,
        "source": f"{term.__class__.__module__}.{term.__class__.__qualname__}",
        "order": order,
        "flatOffset": [offset, offset + dim],
        "jointOrder": list(getattr(term, "joint_names", getattr(term, "_target_names", []))),
        "task": f"{TASK_ID}@{TASK_VERSION}",
        "profile": f"{PROFILE_ID}@{PROFILE_VERSION}",
      }
      field["hash"] = _hash_obj(field)
      action_fields.append(field)
      offset += dim
    if offset != env.action_manager.total_action_dim:
      raise RuntimeError("Action schema does not match runtime action dimension")
    action = {
      "schema": "robolab.action-schema.v1",
      "id": "robolab.motion.velocity.flat.action",
      "version": "1.0.0",
      "task": f"{TASK_ID}@{TASK_VERSION}",
      "profile": f"{PROFILE_ID}@{PROFILE_VERSION}",
      "dimension": env.action_manager.total_action_dim,
      "dtype": str(env.action_manager.action.dtype).replace("torch.", ""),
      "finite": bool(torch.isfinite(env.action_manager.action).all()),
      "controlFrequencyHz": 1.0 / env.step_dt,
      "fields": action_fields,
    }
    action["hash"] = _hash_obj(action)
    _write_json(output / "observation_schema.json", observation)
    _write_json(output / "action_schema.json", action)
    _write_json(output / "runtime_check.json", {
      "actorShape": list(observations["actor"].shape),
      "criticShape": list(observations["critic"].shape),
      "actionDimension": env.action_manager.total_action_dim,
      "dtype": action["dtype"],
      "finite": observation["groups"]["actor"]["finite"] and observation["groups"]["critic"]["finite"],
      "device": device,
      "seed": seed,
    })
    return {"observation": observation, "action": action}
  finally:
    env.close()


def write_recipe(output: Path, device: str, num_envs: int, seed: int, iterations: int) -> None:
  _, load_rl_cfg = _import_task()
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(TASK_ID)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  agent_cfg = load_rl_cfg(TASK_ID)
  agent_cfg.seed = seed
  agent_cfg.max_iterations = iterations
  recipe = {
    "schema": "robolab.training-recipe.v1",
    "id": "robolab.motion.velocity.flat.training",
    "version": "1.0.0",
    "task": f"{TASK_ID}@{TASK_VERSION}",
    "robotProfile": f"{PROFILE_ID}@{PROFILE_VERSION}",
    "seed": seed,
    "device": device,
    "gpuId": 0 if device.startswith("cuda") else None,
    "numEnvs": num_envs,
    "simulationTimestep": env_cfg.sim.mujoco.timestep,
    "decimation": env_cfg.decimation,
    "controlFrequencyHz": 1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation),
    "episodeLengthSeconds": env_cfg.episode_length_s,
    "ppoRunner": _jsonable(agent_cfg),
    "actorNetwork": _jsonable(agent_cfg.actor),
    "criticNetwork": _jsonable(agent_cfg.critic),
    "observationNormalization": {
      "actor": agent_cfg.actor.obs_normalization,
      "critic": agent_cfg.critic.obs_normalization,
    },
    "actionClipping": agent_cfg.clip_actions,
    "rewardTerms": _jsonable(env_cfg.rewards),
    "commandRanges": _jsonable(env_cfg.commands),
    "randomization": _jsonable(env_cfg.events),
    "checkpointInterval": agent_cfg.save_interval,
    "maxIterations": iterations,
    "resolvedEnvironmentConfig": _jsonable(env_cfg),
    "resolvedAgentConfig": _jsonable(agent_cfg),
    "lineage": {
      "robolabRevision": _git_revision(_repo_root()),
      "mjlabUpstreamRevision": "0fb8a681136be94ffc636a3dd423cabb97d91f10",
      "python": platform.python_version(),
      "torch": torch.__version__,
      "cuda": torch.version.cuda,
      "mujoco": importlib.metadata.version("mujoco"),
      "warp": importlib.metadata.version("warp-lang"),
      "rslRl": importlib.metadata.version("rsl-rl-lib"),
    },
  }
  recipe["hash"] = _hash_obj(recipe)
  _write_json(output / "training_recipe.json", recipe)


def _load_policy(checkpoint: Path, device: str):
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array")
  agent_cfg = load_rl_cfg(TASK_ID)
  wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID)
  if runner_cls is None:
    raise RuntimeError("R3 task has no runner class")
  runner = runner_cls(wrapper, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  return env, wrapper, runner.get_inference_policy(device=device)


def play_rollout(checkpoint: Path, output: Path, device: str, seed: int, steps: int) -> dict[str, Any]:
  from mjlab.utils.wrappers import VideoRecorder

  env, wrapper, policy = _load_policy(checkpoint, device)
  recorder = VideoRecorder(
    env,
    video_folder=output / "video",
    step_trigger=lambda step: step == 0,
    video_length=steps,
    disable_logger=True,
  )
  wrapper.env = recorder
  try:
    wrapper.seed(seed)
    obs = wrapper.get_observations()
    action_norms = []
    commands = []
    terminations = []
    with torch.no_grad():
      for _ in range(steps):
        action = policy(obs)
        action_norms.append(float(torch.linalg.vector_norm(action, dim=1).mean().cpu()))
        command = env.command_manager.get_command("twist")
        commands.append(command[0].detach().cpu().tolist())
        obs, _, terminated, truncated, _ = wrapper.step(action)
        terminations.append({"terminated": bool(terminated[0]), "truncated": bool(truncated[0])})
    recorder.close()
    videos = sorted((output / "video").glob("*.mp4"))
    return {
      "checkpoint": str(checkpoint),
      "checkpointSha256": _sha256(checkpoint),
      "task": f"{TASK_ID}@{TASK_VERSION}",
      "profile": f"{PROFILE_ID}@{PROFILE_VERSION}",
      "seed": seed,
      "steps": steps,
      "actionNormMean": float(np.mean(action_norms)),
      "actionNormStd": float(np.std(action_norms)),
      "commandFirst": commands[0] if commands else [],
      "terminations": terminations,
      "video": str(videos[-1]) if videos else None,
    }
  finally:
    if recorder.is_recording:
      recorder.close()


def evaluate(checkpoint: Path, output: Path, device: str, episodes: int, seeds: list[int]) -> dict[str, Any]:
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  agent_cfg = load_rl_cfg(TASK_ID)
  runner_cls = load_runner_cls(TASK_ID)
  if runner_cls is None:
    raise RuntimeError("R3 task has no runner class")
  records: list[dict[str, float]] = []
  all_steps = 0
  for seed in seeds:
    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs = min(16, episodes)
    cfg.seed = seed
    env = ManagerBasedRlEnv(cfg, device=device)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = runner_cls(wrapper, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    try:
      obs = wrapper.get_observations()
      n = wrapper.num_envs
      ep_return = torch.zeros(n, device=device)
      ep_steps = torch.zeros(n, device=device)
      lin_err = torch.zeros(n, device=device)
      yaw_err = torch.zeros(n, device=device)
      tilt = torch.zeros(n, device=device)
      action_mag = torch.zeros(n, device=device)
      action_smooth = torch.zeros(n, device=device)
      joint_violation = torch.zeros(n, device=device)
      contact_steps = torch.zeros(n, device=device)
      previous_action = torch.zeros((n, env.action_manager.total_action_dim), device=device)
      finished = 0
      while finished < episodes:
        command = env.command_manager.get_command("twist")
        asset = env.scene["robot"]
        actual_lin = asset.data.root_link_lin_vel_b
        actual_yaw = asset.data.root_link_ang_vel_b[:, 2]
        gravity = asset.data.projected_gravity_b
        action = policy(obs)
        contact = env.scene["feet_ground_contact"].data.found
        assert contact is not None
        limits = asset.data.soft_joint_pos_limits
        assert limits is not None
        violation = (
          (limits[:, :, 0] - asset.data.joint_pos).clamp(min=0.0)
          + (asset.data.joint_pos - limits[:, :, 1]).clamp(min=0.0)
        ).sum(dim=1)
        obs, reward, terminated, truncated, _ = wrapper.step(action)
        done = terminated | truncated
        ep_return += reward
        ep_steps += 1
        lin_err += torch.linalg.vector_norm(command[:, :2] - actual_lin[:, :2], dim=1)
        yaw_err += torch.abs(command[:, 2] - actual_yaw)
        tilt += torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0))
        action_mag += torch.linalg.vector_norm(action, dim=1)
        action_smooth += torch.linalg.vector_norm(action - previous_action, dim=1)
        joint_violation += violation
        contact_steps += (contact > 0).float().mean(dim=1)
        previous_action = action.detach()
        all_steps += n
        for idx in torch.where(done)[0].tolist():
          if finished >= episodes:
            break
          records.append({
            "seed": float(seed),
            "episodeReturn": float(ep_return[idx] / ep_steps[idx].clamp(min=1)),
            "episodeLength": float(ep_steps[idx]),
            "linearTrackingError": float(lin_err[idx] / ep_steps[idx].clamp(min=1)),
            "angularTrackingError": float(yaw_err[idx] / ep_steps[idx].clamp(min=1)),
            "uprightAngleRad": float(tilt[idx] / ep_steps[idx].clamp(min=1)),
            "actionMagnitude": float(action_mag[idx] / ep_steps[idx].clamp(min=1)),
            "actionSmoothness": float(action_smooth[idx] / ep_steps[idx].clamp(min=1)),
            "jointLimitViolation": float(joint_violation[idx] / ep_steps[idx].clamp(min=1)),
            "footContactFraction": float(contact_steps[idx] / ep_steps[idx].clamp(min=1)),
            "fall": float(bool(terminated[idx])),
          })
          finished += 1
          ep_return[idx] = 0
          ep_steps[idx] = 0
          lin_err[idx] = 0
          yaw_err[idx] = 0
          tilt[idx] = 0
          action_mag[idx] = 0
          action_smooth[idx] = 0
          joint_violation[idx] = 0
          contact_steps[idx] = 0
    finally:
      env.close()

  def mean(name: str) -> float:
    return float(np.mean([record[name] for record in records]))

  metrics = {
    "linearTrackingError": mean("linearTrackingError"),
    "angularTrackingError": mean("angularTrackingError"),
    "uprightAngleRad": mean("uprightAngleRad"),
    "survivalEpisodeLength": mean("episodeLength"),
    "fallRate": mean("fall"),
    "actionMagnitude": mean("actionMagnitude"),
    "actionSmoothness": mean("actionSmoothness"),
    "jointLimitViolation": mean("jointLimitViolation"),
    "footContactFraction": mean("footContactFraction"),
    "episodeReturn": mean("episodeReturn"),
  }
  thresholds = {
    "linearTrackingError": {"value": 1.0, "direction": "max"},
    "angularTrackingError": {"value": 0.8, "direction": "max"},
    "uprightAngleRad": {"value": 0.9, "direction": "max"},
    "survivalEpisodeLength": {"value": 100.0, "direction": "min"},
    "fallRate": {"value": 0.6, "direction": "max"},
    "actionMagnitude": {"value": 4.0, "direction": "max"},
    "actionSmoothness": {"value": 2.0, "direction": "max"},
    "jointLimitViolation": {"value": 0.1, "direction": "max"},
    "footContactFraction": {"value": 0.05, "direction": "min"},
    "episodeReturn": {"value": 0.0, "direction": "min"},
  }
  passed = all(
    metrics[name] <= spec["value"] if spec["direction"] == "max" else metrics[name] >= spec["value"]
    for name, spec in thresholds.items()
  )
  result = {
    "schema": "robolab.independent-evaluation.v1",
    "status": "passed" if passed else "failed",
    "thresholdVersion": THRESHOLD_VERSION,
    "task": f"{TASK_ID}@{TASK_VERSION}",
    "profile": f"{PROFILE_ID}@{PROFILE_VERSION}",
    "checkpoint": str(checkpoint),
    "checkpointSha256": _sha256(checkpoint),
    "scene": "flat",
    "episodes": len(records),
    "seeds": seeds,
    "device": device,
    "metrics": {name: {"value": metrics[name], **spec, "passed": (
      metrics[name] <= spec["value"] if spec["direction"] == "max" else metrics[name] >= spec["value"]
    )} for name, spec in thresholds.items()},
    "aggregate": metrics,
    "episodesDetail": records,
    "steps": all_steps,
  }
  _write_json(output / "evaluate.json", result)
  lines = [
    "# R3 independent evaluate",
    "",
    f"Status: **{result['status']}**",
    f"Checkpoint SHA-256: `{result['checkpointSha256']}`",
    f"Task: `{result['task']}`; Profile: `{result['profile']}`",
    f"Episodes: {result['episodes']}; seeds: {seeds}; threshold: `{THRESHOLD_VERSION}`",
    "",
    "| Metric | Value | Threshold | Result |",
    "|---|---:|---:|---|",
  ]
  for name, item in result["metrics"].items():
    lines.append(f"| {name} | {item['value']:.6f} | {item['direction']} {item['value']:.6f} | {'PASS' if item['passed'] else 'FAIL'} |".replace(f"{item['value']:.6f} | {item['direction']}", f"{metrics[name]:.6f} | {item['direction']} {item['value']:.6f}"))
  (output / "evaluate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  schema = sub.add_parser("schema")
  schema.add_argument("--output", type=Path, required=True)
  schema.add_argument("--device", default="cpu")
  schema.add_argument("--num-envs", type=int, default=1)
  schema.add_argument("--seed", type=int, default=20260821)
  recipe = sub.add_parser("recipe")
  recipe.add_argument("--output", type=Path, required=True)
  recipe.add_argument("--device", default="cuda:0")
  recipe.add_argument("--num-envs", type=int, required=True)
  recipe.add_argument("--seed", type=int, required=True)
  recipe.add_argument("--iterations", type=int, required=True)
  play = sub.add_parser("play")
  play.add_argument("--checkpoint", type=Path, required=True)
  play.add_argument("--output", type=Path, required=True)
  play.add_argument("--device", default="cuda:0")
  play.add_argument("--seed", type=int, default=20260821)
  play.add_argument("--steps", type=int, default=1000)
  evaluate_cmd = sub.add_parser("evaluate")
  evaluate_cmd.add_argument("--checkpoint", type=Path, required=True)
  evaluate_cmd.add_argument("--output", type=Path, required=True)
  evaluate_cmd.add_argument("--device", default="cuda:0")
  evaluate_cmd.add_argument("--episodes", type=int, default=32)
  evaluate_cmd.add_argument("--seeds", default="101,202,303")
  args = parser.parse_args()
  if args.command == "schema":
    capture_schema(args.output, args.device, args.num_envs, args.seed)
  elif args.command == "recipe":
    write_recipe(args.output, args.device, args.num_envs, args.seed, args.iterations)
  elif args.command == "play":
    _write_json(args.output / "play.json", play_rollout(args.checkpoint, args.output, args.device, args.seed, args.steps))
  else:
    evaluate(args.checkpoint, args.output, args.device, args.episodes, [int(x) for x in args.seeds.split(",")])


if __name__ == "__main__":
  main()
