# Third-Party Notices

This file is an initial attribution inventory and must be updated as RoboLab's dependency graph and distributed artifacts evolve.

## unitree_rl_mjlab

- Project: Unitree RL MJLab
- Source: https://github.com/unitreerobotics/unitree_rl_mjlab
- Current local location: `mjlab/`; planned curated location: `vendor/unitree_rl_mjlab/`
- License: Apache License 2.0; see `mjlab/LICENCE`
- Relationship: direct source of selected Unitree training, robot assets, simulation, and legacy deployment technology. Promotional documentation, bundled runtimes, and Skill artifacts are excluded by the curated vendor manifest.
- Imported revision: `1425b15f73bd4095f0df53709d7c389c3eb9e790`; see `mjlab/UPSTREAM.md`.

## Components in the current complete working copy

The untracked, provenance-verification copy currently bundles additional third-party source or binary distributions. Curated import rules may exclude a component; any component that is redistributed must keep its original license and notice files:

- ONNX Runtime 1.22.0, x86-64 and aarch64 distributions: `mjlab/deploy/thirdparty/onnxruntime-linux-*/LICENSE` and `ThirdPartyNotices.txt`.
- cnpy, copyright Carl Rogers: `mjlab/deploy/thirdparty/cnpy/LICENSE`.
- LodePNG, copyright Lode Vandevenne: `mjlab/simulate/src/lodepng/LICENSE`.
- joystick-derived code: `mjlab/simulate/src/joystick/LICENSE-2.0.txt` and `readme.md`.

The planned active vendor excludes the precompiled ONNX Runtime distributions and other redundant runtime binaries. Their notices remain relevant only if those binaries are redistributed by a setup package or release artifact.

RoboLab's root MIT license does not replace the license or notices attached to third-party source code, assets, models, data, or documentation. Acknowledgement details and the release checklist are maintained in `docs/UPSTREAM_AND_ACKNOWLEDGEMENTS.md`.
