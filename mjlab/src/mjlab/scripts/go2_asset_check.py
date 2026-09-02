"""Minimal structural acceptance check for the migrated Unitree Go2 asset."""

from __future__ import annotations

import argparse

import mujoco

from mjlab.asset_zoo.robots.unitree_go2.go2_constants import get_spec
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--task", default="Mjlab-Velocity-Flat-Unitree-Go2",
    help="Go2 task used to build the injected-actuator model.",
  )
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401, PLC0415

  asset_model = get_spec().compile()
  asset_expected = {"nq": 19, "nv": 18, "nbody": 14, "ngeom": 56}
  asset_actual = {
    name: int(getattr(asset_model, name)) for name in asset_expected
  }
  if asset_actual != asset_expected:
    raise RuntimeError(
      f"Go2 XML dimensions mismatch: expected {asset_expected}, got {asset_actual}"
    )

  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device=args.device)
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

    body_names = {
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
      for i in range(model.nbody)
    }
    joint_names = {
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
      for i in range(model.njnt)
    }
    site_names = {
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
      for i in range(model.nsite)
    }
    # Entity names are namespaced as ``robot/<name>`` in an environment model;
    # compare the terminal component so the check remains stable if the scene
    # entity name changes.
    body_suffixes = {name.rsplit("/", 1)[-1] for name in body_names}
    joint_suffixes = {name.rsplit("/", 1)[-1] for name in joint_names}
    site_suffixes = {name.rsplit("/", 1)[-1] for name in site_names}
    required_bodies = {"base_link"}
    required_joints = {
      f"{leg}_{kind}_joint"
      for leg in ("FL", "FR", "RL", "RR")
      for kind in ("hip", "thigh", "calf")
    }
    required_sites = {"FL", "FR", "RL", "RR"}
    for label, required, available in (
      ("body", required_bodies, body_suffixes),
      ("joint", required_joints, joint_suffixes),
      ("site", required_sites, site_suffixes),
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


if __name__ == "__main__":
  main()
