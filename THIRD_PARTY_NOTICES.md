# Third-party provenance and release considerations

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
- Use: behavioral and interface reference for task/reward/observation semantics
  and for DreamWaQ, AMP, CTS, and teacher/student training.
- Its README attributes ETH `legged_gym`, RSL-RL, DreamWaQ and CTS research,
  and `fan-ziqi/rl_amp`.
- License status: the supplied snapshot contains no `LICENSE`, `COPYING`, or
  `NOTICE` file.

The project owner confirms that no implementation files from this snapshot
were copied into LainLoco. The LainLoco implementation was written for the
mjlab APIs using the source project only as a behavioral and interface
reference. The upstream repository is therefore recorded for technical
provenance and acknowledgement, not as incorporated distribution material or
as a release blocker.

## Runtime dependencies

Runtime and development packages are resolved by `uv.lock`. Important direct
or transitive projects include RSL-RL (BSD-3-Clause in installed metadata),
MuJoCo and MuJoCo Warp (Apache-2.0), ONNX Runtime (MIT), PyTorch, NumPy, and
their dependencies. Binary redistribution must include the license material
required by the exact artifacts being shipped; the lockfile is not itself a
license inventory.

## Project license status

The owner has not yet selected a top-level license for the original LainLoco
code. Apache-2.0 would align with both mjlab and Unitree RL Mjlab. The choice of
project license remains an owner decision and is independent from the
behavior-only `My_unitree_go2_gym` reference described above.
