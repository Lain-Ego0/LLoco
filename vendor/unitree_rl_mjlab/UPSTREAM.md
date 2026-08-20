# Upstream Record

## Source

- Repository: https://github.com/unitreerobotics/unitree_rl_mjlab
- License: Apache License 2.0 (`LICENCE` in this directory)
- Imported revision: [`1425b15f73bd4095f0df53709d7c389c3eb9e790`](https://github.com/unitreerobotics/unitree_rl_mjlab/commit/1425b15f73bd4095f0df53709d7c389c3eb9e790)
- Import method: copied source tree; not yet represented as a Git submodule/subtree/fork in the RoboLab history

Verification performed on 2026-08-20: every upstream blob path and Git blob SHA-1 in the original local working copy matched the commit above. This record file and the vendor manifest are RoboLab additions.

RoboLab import mode: curated vendor under `vendor/unitree_rl_mjlab/`. Upstream READMEs/docs/GIFs, bundled precompiled runtimes, and Skill-owned policies/motions are excluded according to `VENDOR_MANIFEST.yaml`.

## Recommended maintenance workflow

RoboLab will create a curated vendor tree from this verified source. Selection must be generated from the pinned upstream commit and documented in `VENDOR_MANIFEST.yaml`; do not silently replace files with fresh copies or downloaded archives.

For every upstream sync:

1. record the previous and new upstream commit;
2. import the upstream change in a dedicated commit;
3. preserve Apache-2.0 license and upstream notices;
4. resolve RoboLab modifications in separate, reviewable commits where practical;
5. update compatibility tests and `THIRD_PARTY_NOTICES.md` when dependencies/assets change.

Modified upstream files must remain traceable, and distributed source must carry the notices required by Apache License 2.0.
