# LLoco Playground

Standalone browser demos for selected, verified LLoco policies.  This project
does not import the workbench, run Python, or communicate with the training
workflow at runtime.

## Run

```bash
cd playground
npm install
npm run dev
```

`npm run prepare-assets` copies the Go2 scene and meshes from the project asset
library into `public/robot/`. Policy ONNX files are versioned in
`public/policies/`; clone with Git LFS enabled before running the demo.

## Included policies

- Front handstand (48-D observation)
- Rear stand (45-D observation)
- Trot (470-D / 10-frame history)
- Jump (470-D / 10-frame history)
- Spring jump (470-D / 10-frame history)
- Arena flat walk (270-D / 6-frame history, sourced from ArenaX)
- DreamWaQ terrain gait (270-D / 6-frame history)
- AMP-CTS gait (270-D / 6-frame history)

The handstand artifact is LLoco's own exported policy from
`logs/rsl_rl/go2_handstand/`, while the other artifacts are converted from the
corresponding TorchScript policies in
`/home/lxy/下载/My_unitree_go2_gym-main`. They are never fetched by the
browser. To intentionally refresh them on a development machine that has
PyTorch and ONNX installed:

```bash
/home/lxy/miniconda3/envs/mjlab/bin/python scripts/export_policies.py \
  --source /home/lxy/下载/My_unitree_go2_gym-main
```

The script copies the selected LLoco handstand export and converts the six
remaining Gym models whose input/output contracts are implemented by this
browser demo. It does not run as part of the normal build.

## Terrain editor

The **Terrain** button opens a browser-native editor implemented as part of
the playground UI; it has no PyQt, Python server, or workbench dependency.
It provides flat, slope, stairs, and seeded obstacle profiles, plus placed
platforms, stairs, ramps, stepping stones, and low walls. Selecting **Apply
to scene** recompiles only the active MuJoCo terrain geometry set; the robot
mesh assets remain cached in the browser VFS so collision topology is current
without reloading those assets.

**Export JSON** creates a portable playground scene draft for future online
publishing. **Import** accepts that JSON and ArenaX-style `scene.json` files;
it also imports simple MuJoCo XML files containing box geoms. Heightfields and
external mesh assets intentionally remain out of scope for browser import.

The current release is a public-facing interactive demo, not a replacement for
native MuJoCo/Viser acceptance checks.
