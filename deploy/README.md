# Deployment programs

Executable sim-to-sim and hardware integrations belong here. Shared policy
bundle, inference, safety-state-machine, and robot-contract code remains in
`src/lainloco` so deployment programs and tests use the same implementation.

Run a validated Go2 bundle in the local mjlab backend with:

```bash
MJLAB_WARP_QUIET=1 uv run --package lainloco --extra cpu \
  python deploy/go2_sim2sim.py /path/to/go2-policy-bundle \
  --steps 100 --num-envs 1 --device cpu
```

No hardware program is provided yet: adding one requires the Unitree SDK,
an explicit hardware joint mapping, and a physical safety review.
