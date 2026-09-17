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

The artifacts are converted from the corresponding TorchScript policies in
`/home/lxy/下载/My_unitree_go2_gym-main`, rather than being fetched by the
browser. To intentionally refresh them on a development machine that has
PyTorch and ONNX installed:

```bash
/home/lxy/miniconda3/envs/mjlab/bin/python scripts/export_gym_policies.py \
  --source /home/lxy/下载/My_unitree_go2_gym-main
```

The script exports only the five models whose input/output contracts are
implemented by this browser demo. It does not run as part of the normal build.

The current release is a public-facing interactive demo, not a replacement for
native MuJoCo/Viser acceptance checks.
