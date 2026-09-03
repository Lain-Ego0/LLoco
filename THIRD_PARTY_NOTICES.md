# Third-party provenance and release blockers

This inventory records material incorporated into the current workspace. It is
not a substitute for the corresponding license texts and does not grant rights
that an upstream source did not grant.

## mjlab

- Upstream: <https://github.com/mujocolab/mjlab>
- Local component: `mjlab/`
- License in the local snapshot: Apache License 2.0 (`mjlab/LICENSE`)
- Use: simulation lifecycle, managers, assets framework, RSL-RL integration,
  training/play commands, and the base project modified by this repository.

The upstream license text and notices must remain with distributions of this
component and its modifications.

## Unitree RL Mjlab reference snapshot

- Canonical upstream: <https://github.com/unitreerobotics/unitree_rl_mjlab>
- Local source snapshot: `/home/lxy/下载/unitree_rl_mjlab-main`
- License: Apache License 2.0 (`LICENCE` in the local/upstream repository).
- Incorporated files: the Go2 MJCF and 16 OBJ meshes now under
  `mjlab/src/mjlab/asset_zoo/robots/unitree_go2/xmls/` are byte-identical to
  the local reference snapshot as of 2026-09-03.

The snapshot uses the British filename `LICENCE`, which was missed by the
initial `LICENSE*` audit. Its README is byte-identical to upstream `main`, and
GitHub identifies the repository license as Apache-2.0. Preserve the Apache
license text and attribution when redistributing the incorporated assets.

## My_unitree_go2_gym reference snapshot

- Canonical upstream: <https://github.com/yusongmin1/My_unitree_go2_gym>
- Local source snapshot: `/home/lxy/下载/My_unitree_go2_gym-main`
- Use: task/reward/observation semantics and reference implementations for
  DreamWaQ, AMP, CTS, and teacher/student training.
- Its README attributes ETH `legged_gym`, RSL-RL, DreamWaQ and CTS research,
  and `fan-ziqi/rl_amp`.
- License status: **unresolved**. The supplied snapshot contains no `LICENSE`,
  `COPYING`, or `NOTICE` file.

Before public release, identify the snapshot's canonical upstream repository,
review file-level provenance for migrated implementation code, and either
obtain compatible permission or replace/reimplement affected material.

## Runtime dependencies

Runtime and development packages are resolved by `uv.lock`. Important direct
or transitive projects include RSL-RL (BSD-3-Clause in installed metadata),
MuJoCo and MuJoCo Warp (Apache-2.0), ONNX Runtime (MIT), PyTorch, NumPy, and
their dependencies. Binary redistribution must include the license material
required by the exact artifacts being shipped; the lockfile is not itself a
license inventory.

## Project license status

The owner has not yet selected a top-level license for the original LainLoco
code. Apache-2.0 would align with both mjlab and Unitree RL Mjlab, but it cannot
grant rights to material derived from the unlicensed `My_unitree_go2_gym`
snapshot. This is an explicit release blocker, not an implicit “all rights
granted” declaration. Add the selected `LICENSE` and package metadata only
after that third-party review.
