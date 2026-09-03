"""Headless Go2 asset, policy-contract, and finite-step checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import mujoco
import torch
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import get_spec
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


@dataclass(frozen=True, slots=True)
class ObservationExpectation:
  """Expected policy and privileged widths for a migrated task."""

  task_id: str
  actor_dim: int
  critic_dim: int


SOURCE_CONTRACTS = (
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


def validate_asset(
  task_id: str = "Mjlab-Velocity-Flat-Unitree-Go2", device: str = "cpu"
) -> None:
  """Validate Go2 MJCF dimensions, actuators, and policy-facing names."""
  asset_model = get_spec().compile()
  asset_expected = {"nq": 19, "nv": 18, "nbody": 14, "ngeom": 56}
  asset_actual = {name: int(getattr(asset_model, name)) for name in asset_expected}
  if asset_actual != asset_expected:
    raise RuntimeError(
      f"Go2 XML dimensions mismatch: expected {asset_expected}, got {asset_actual}"
    )

  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    model = env.sim.mj_model
    expected = {"nq": 19, "nv": 18, "nu": 12}
    actual = {name: int(getattr(model, name)) for name in expected}
    if actual != expected:
      raise RuntimeError(
        f"Go2 environment dimensions mismatch: expected {expected}, got {actual}"
      )
    if model.nbody < asset_expected["nbody"] or model.ngeom < asset_expected["ngeom"]:
      raise RuntimeError(
        "Go2 environment dropped asset bodies/geoms: "
        f"nbody={model.nbody}, ngeom={model.ngeom}"
      )

    body_suffixes = _name_suffixes(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    joint_suffixes = _name_suffixes(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    site_suffixes = _name_suffixes(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    required_joints = {
      f"{leg}_{kind}_joint"
      for leg in ("FL", "FR", "RL", "RR")
      for kind in ("hip", "thigh", "calf")
    }
    for label, required, available in (
      ("body", {"base_link"}, body_suffixes),
      ("joint", required_joints, joint_suffixes),
      ("site", {"FL", "FR", "RL", "RR"}, site_suffixes),
    ):
      missing = sorted(required - available)
      if missing:
        raise RuntimeError(f"Missing Go2 {label} names: {missing}")
    print(
      "Go2 asset check passed: "
      + ", ".join(f"{name}={value}" for name, value in asset_actual.items())
      + f"; env_nu={model.nu}, env_nbody={model.nbody}, env_ngeom={model.ngeom}"
      + "; key body/joint/site names resolved"
    )
  finally:
    env.close()


def _name_suffixes(model: mujoco.MjModel, kind: mujoco.mjtObj, count: int) -> set[str]:
  names = {mujoco.mj_id2name(model, kind, index) for index in range(count)}
  return {name.rsplit("/", 1)[-1] for name in names if name is not None}


def validate_contracts(device: str = "cpu") -> None:
  """Reset and step all 14 migrated source tasks and check observation widths."""
  for expected in SOURCE_CONTRACTS:
    cfg = load_env_cfg(expected.task_id, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
      obs, _ = env.reset()
      if env.action_manager.total_action_dim != 12:
        raise RuntimeError(
          f"{expected.task_id}: expected 12 actions, "
          f"got {env.action_manager.total_action_dim}"
        )
      action = torch.zeros((1, 12), device=env.device)
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
        f"{expected.task_id}: actions=12 actor={actual_dims[0]} "
        f"critic={actual_dims[1]} OK"
      )
    finally:
      env.close()
  print(f"Go2 contract check passed: {len(SOURCE_CONTRACTS)} source tasks")


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
    reward = terminated = truncated = torch.empty(0, device=env.device)
    for _ in range(steps):
      if agent == "zero":
        action = torch.zeros((num_envs, 12), device=env.device)
      else:
        action = 2.0 * torch.rand((num_envs, 12), device=env.device) - 1.0
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
