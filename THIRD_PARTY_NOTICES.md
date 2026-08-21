# Third-Party Notices

This file is an initial attribution inventory and must be updated as RoboLab's dependency graph and distributed artifacts evolve.

## unitree_rl_mjlab（historical source; retirement scheduled for R0.6）

- Project: Unitree RL MJLab
- Source: https://github.com/unitreerobotics/unitree_rl_mjlab
- Historical location: `vendor/unitree_rl_mjlab/` (curated vendor import; scheduled for deletion from the active tree in R0.6)
- License: Apache License 2.0; see `vendor/unitree_rl_mjlab/LICENCE`
- Relationship: historical source of selected Unitree training, robot assets, simulation, and legacy deployment technology. It is not part of the future Customized MJLab 1.6 active toolchain.
- Imported revision: `1425b15f73bd4095f0df53709d7c389c3eb9e790`; see `vendor/unitree_rl_mjlab/UPSTREAM.md`.

## Components inside the curated vendor tree

The curated vendor tree bundles additional third-party source distributions. Any component that is redistributed must keep its original license and notice files:

- cnpy, copyright Carl Rogers: `vendor/unitree_rl_mjlab/deploy/thirdparty/cnpy/LICENSE`.
- LodePNG, copyright Lode Vandevenne: `vendor/unitree_rl_mjlab/simulate/src/lodepng/LICENSE`.
- joystick-derived code: `vendor/unitree_rl_mjlab/simulate/src/joystick/LICENSE-2.0.txt` and `readme.md`.

The historical curated vendor excluded the precompiled ONNX Runtime 1.22.0 distributions and bundled MuJoCo binaries. These notices remain relevant only while the corresponding historical files are present or redistributed. After R0.6, the current release notice must list only third-party files actually shipped by the active tree.

RoboLab's root MIT license does not replace the license or notices attached to third-party source code, assets, models, data, or documentation. RoboLab's Customized MJLab 1.6 distribution may contain downstream modifications to the upstream MJLab source; those files retain the applicable upstream license, copyright, NOTICE, and modification records. RoboLab-owned platform, task, integration, and runtime code remains governed by the root MIT license unless a file states otherwise. Acknowledgement details and the release checklist are maintained in `docs/reference/UPSTREAM_AND_ACKNOWLEDGEMENTS.md`.
