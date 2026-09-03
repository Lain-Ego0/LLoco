"""Headless robot asset, policy-contract, and finite-step checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import mujoco
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as get_g1_spec
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import get_spec as get_go2_spec
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.tasks.registry import load_env_cfg

from lainloco.experiments import resolve_robot


@dataclass(frozen=True, slots=True)
class ObservationExpectation:
  """Expected policy and privileged widths for a migrated task."""

  task_id: str
  actor_dim: int
  critic_dim: int


GO2_SOURCE_CONTRACTS = (
  ObservationExpectation("Mjlab-Trot-Flat-Unitree-Go2", 470, 204),
  ObservationExpectation("Mjlab-Jump-Flat-Unitree-Go2", 470, 210),
  ObservationExpectation("Mjlab-Spring-Jump-Flat-Unitree-Go2", 470, 195),
  ObservationExpectation("Mjlab-Backflip-Flat-Unitree-Go2", 470, 150),
  ObservationExpectation("Mjlab-Handstand-Flat-Unitree-Go2", 45, 86),
  ObservationExpectation("Mjlab-Leggedstand-Flat-Unitree-Go2", 45, 86),
  ObservationExpectation("Mjlab-DreamWaQ-Rough-Unitree-Go2", 45, 783),
  ObservationExpectation("Mjlab-AMP-DreamWaQ-Rough-Unitree-Go2", 45, 783),
  ObservationExpectation("Mjlab-CTS-Rough-Unitree-Go2", 45, 278),
  ObservationExpectation("Mjlab-AMP-CTS-Rough-Unitree-Go2", 45, 281),
  ObservationExpectation("Mjlab-AMP-TS-Rough-Unitree-Go2", 45, 309),
  ObservationExpectation("Mjlab-AMP-TS-Student-Rough-Unitree-Go2", 45, 309),
  ObservationExpectation("Mjlab-TS-Rough-Unitree-Go2", 45, 309),
  ObservationExpectation("Mjlab-TS-Student-Rough-Unitree-Go2", 45, 309),
)

G1_SOURCE_CONTRACTS = (
  ObservationExpectation("Mjlab-Velocity-Flat-Unitree-G1", 99, 111),
  ObservationExpectation("Mjlab-Velocity-Rough-Unitree-G1", 286, 298),
)


def validate_asset(
  task_id: str | None = None,
  device: str = "cpu",
  robot_id: str = "go2",
) -> None:
  """Validate MJCF dimensions, actuators, and policy-facing names."""
  robot = resolve_robot(robot_id)
  if robot.robot_id == "unitree/go2":
    label = "Go2"
    asset_model = get_go2_spec().compile()
    asset_expected = {"nq": 19, "nv": 18, "nbody": 14, "ngeom": 56}
    selected_task = task_id or "Mjlab-Velocity-Flat-Unitree-Go2"
  else:
    label = "G1"
    asset_model = get_g1_spec().compile()
    asset_expected = {"nq": 36, "nv": 35, "nbody": 31, "ngeom": 68}
    selected_task = task_id or "Mjlab-Velocity-Flat-Unitree-G1"
  asset_actual = {name: int(getattr(asset_model, name)) for name in asset_expected}
  if asset_actual != asset_expected:
    raise RuntimeError(
      f"{label} XML dimensions mismatch: expected {asset_expected}, got {asset_actual}"
    )

  cfg = load_env_cfg(selected_task, play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    model = env.sim.mj_model
    expected = {
      "nq": asset_expected["nq"],
      "nv": asset_expected["nv"],
      "nu": robot.action_dim,
    }
    actual = {name: int(getattr(model, name)) for name in expected}
    if actual != expected:
      raise RuntimeError(
        f"{label} environment dimensions mismatch: expected {expected}, got {actual}"
      )
    if model.nbody < asset_expected["nbody"] or model.ngeom < asset_expected["ngeom"]:
      raise RuntimeError(
        f"{label} environment dropped asset bodies/geoms: "
        f"nbody={model.nbody}, ngeom={model.ngeom}"
      )

    body_suffixes = _name_suffixes(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    geom_suffixes = _name_suffixes(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
    joint_suffixes = _name_suffixes(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    site_suffixes = _name_suffixes(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    for kind, required, available in (
      ("body", {robot.base_body}, body_suffixes),
      ("geom", set(robot.collision_geoms), geom_suffixes),
      ("joint", set(robot.joint_order), joint_suffixes),
      ("site", set(robot.foot_sites), site_suffixes),
    ):
      missing = sorted(required - available)
      if missing:
        raise RuntimeError(f"Missing {robot.robot_id} {kind} names: {missing}")

    action = env.action_manager.get_term("joint_pos")
    if not isinstance(action, JointPositionAction):
      raise RuntimeError(
        f"{robot.robot_id} joint_pos must be a JointPositionAction, "
        f"got {type(action).__name__}"
      )
    if tuple(action.target_names) != robot.joint_order:
      raise RuntimeError(
        f"{robot.robot_id} policy joint order does not match RobotSpec"
      )
    actual_scale = action.scale
    if isinstance(actual_scale, torch.Tensor):
      scale = actual_scale[0]
    else:
      scale = torch.full((robot.action_dim,), actual_scale, device=env.device)
    expected_scale = torch.tensor(robot.action_scale, device=env.device)
    if not torch.allclose(scale, expected_scale, rtol=1e-6, atol=1e-7):
      raise RuntimeError(f"{robot.robot_id} action scale does not match RobotSpec")
    actual_offset = action.offset
    if not isinstance(actual_offset, torch.Tensor):
      raise RuntimeError(f"{robot.robot_id} joint_pos must use the default-pose offset")
    expected_pose = torch.tensor(
      [value for _name, value in robot.default_pose], device=env.device
    )
    if not torch.allclose(actual_offset[0], expected_pose, rtol=1e-6, atol=1e-7):
      raise RuntimeError(f"{robot.robot_id} default pose does not match RobotSpec")
    print(
      f"{robot.robot_id} asset check passed: "
      + ", ".join(f"{name}={value}" for name, value in asset_actual.items())
      + f"; env_nu={model.nu}, env_nbody={model.nbody}, env_ngeom={model.ngeom}"
      + "; names, joint order, default pose, and action scale resolved"
    )
  finally:
    env.close()


def _name_suffixes(model: mujoco.MjModel, kind: mujoco.mjtObj, count: int) -> set[str]:
  names = {mujoco.mj_id2name(model, kind, index) for index in range(count)}
  return {name.rsplit("/", 1)[-1] for name in names if name is not None}


def validate_contracts(device: str = "cpu", robot_id: str = "go2") -> None:
  """Reset, step, and verify every source task for one robot."""
  robot = resolve_robot(robot_id)
  expectations = (
    GO2_SOURCE_CONTRACTS if robot.robot_id == "unitree/go2" else G1_SOURCE_CONTRACTS
  )
  for expected in expectations:
    cfg = load_env_cfg(expected.task_id, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
      obs, _ = env.reset()
      if env.action_manager.total_action_dim != robot.action_dim:
        raise RuntimeError(
          f"{expected.task_id}: expected {robot.action_dim} actions, "
          f"got {env.action_manager.total_action_dim}"
        )
      action = torch.zeros((1, robot.action_dim), device=env.device)
      obs, _reward, _terminated, _truncated, _info = env.step(action)
      tensor_obs = cast(dict[str, torch.Tensor], obs)
      actual_dims = (
        int(tensor_obs["actor"].shape[-1]),
        int(tensor_obs["critic"].shape[-1]),
      )
      expected_dims = (expected.actor_dim, expected.critic_dim)
      if actual_dims != expected_dims:
        raise RuntimeError(
          f"{expected.task_id}: expected actor/critic {expected_dims}, "
          f"got {actual_dims}"
        )
      print(
        f"{expected.task_id}: actions={robot.action_dim} actor={actual_dims[0]} "
        f"critic={actual_dims[1]} OK"
      )
    finally:
      env.close()
  print(f"{robot.robot_id} contract check passed: {len(expectations)} source tasks")


def smoke(
  task_id: str,
  *,
  agent: str = "random",
  steps: int = 4,
  num_envs: int = 2,
  device: str = "cpu",
) -> None:
  """Run a finite zero- or random-action rollout without a viewer."""
  if agent not in {"zero", "random"}:
    raise ValueError("agent must be 'zero' or 'random'")
  if steps < 0 or num_envs <= 0:
    raise ValueError("steps must be non-negative and num_envs must be positive")
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    obs, _ = env.reset()
    action_dim = env.action_manager.total_action_dim
    reward = terminated = truncated = torch.empty(0, device=env.device)
    for _ in range(steps):
      if agent == "zero":
        action = torch.zeros((num_envs, action_dim), device=env.device)
      else:
        action = 2.0 * torch.rand((num_envs, action_dim), device=env.device) - 1.0
      obs, reward, terminated, truncated, _ = env.step(action)
    tensor_obs = cast(dict[str, torch.Tensor], obs)
    print(
      f"{task_id}: agent={agent} steps={steps} "
      f"actor={tuple(tensor_obs['actor'].shape)} "
      f"critic={tuple(tensor_obs['critic'].shape)} "
      f"reward={tuple(reward.shape)} terminated={tuple(terminated.shape)} "
      f"truncated={tuple(truncated.shape)}"
    )
  finally:
    env.close()
