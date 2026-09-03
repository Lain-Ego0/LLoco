# Contributing to Lain's LocoLab

LainLoco extends mjlab with robot-owned tasks, reusable learning components,
and versioned deployment contracts. Keep changes inside those boundaries and
preserve the legacy `Mjlab-*` interfaces during the compatibility period.

## Development setup

```bash
uv sync --package lainloco --extra cpu
```

Use `--extra cu128` only for GPU training or capacity checks. Generated
checkpoints, ONNX files, videos, and local bundles do not belong in commits.

## Where changes belong

- New robot facts go in a single `RobotSpec` under `robots/<vendor>/<robot>/`.
- A skill is a `TaskSpec` plus a thin environment factory in that robot's
  `tasks/` tree. Do not encode an algorithm in a new task ID.
- Reusable algorithms, models, storage, exporters, and runners go under
  `lainloco.learning`; robot-specific parameters stay under the robot's
  `training/` directory.
- Training/deployment boundaries must be represented by `PolicyContract` and
  exported as a complete Policy Bundle.
- Generic simulation lifecycle changes belong upstream in `mjlab`, not in a
  robot task.

## Adding a robot

1. Add one authoritative `RobotSpec`, including ordered joints, action scale,
   default pose, physics/control periods, and asset factory.
2. Add task and training Catalogs without import-time registration side
   effects.
3. Compose explicit `ExperimentSpec` entries and register canonical IDs; add
   legacy aliases only when compatibility requires them.
4. Add contract tests for asset dimensions, observation/action widths, factory
   isolation, bundle reload, and timing rejection.

Do not set a hardware joint permutation by assumption. It requires an SDK and
physical safety review independent of simulation acceptance.

## Adding a task or training profile

For a task, document its family, terrain, commands, observations, rewards,
terminations, randomization, and episode length. Reuse shared MDP terms and
override only genuine skill differences.

For a training profile, identify the algorithm, actor/critic, storage, runner,
optimizer, observation groups, auxiliary losses, and exporter. Teacher/student
training must use an explicit teacher checkpoint workflow.

Any observation order, action scale, joint order, history/reset behavior,
normalization, recurrent state, or control-period change is a contract change.
Version it and add a rejection/migration test.

## Required checks

```bash
uv run --package lainloco --extra cpu ruff check .
uv run --package lainloco --extra cpu ty check
uv run --package lainloco --extra cpu pyright
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --package mjlab --extra cpu pytest -q \
  tests/contracts tests/integration vendor/mjlab/tests/test_task_configs.py \
  vendor/mjlab/tests/test_velocity_task.py \
  vendor/mjlab/tests/test_velocity_rewards.py \
  vendor/mjlab/tests/test_foot_height_sensor.py \
  vendor/mjlab/tests/test_runner.py
MJLAB_WARP_QUIET=1 uv run --package lainloco --extra cpu \
  lainloco validate contracts --device cpu
uv build --package lainloco
```

When training math, rewards, observations, or physics change, also run the
smallest relevant real runner update. Contract or smoke success must not be
reported as policy convergence.

## Documentation and provenance

Update `PROJECT_PROGRESS.md` with reproducible evidence and add user-facing
changes to `vendor/mjlab/docs/source/changelog.rst`. Record every incorporated code,
asset, or dataset source in `THIRD_PARTY_NOTICES.md` before committing it.

The repository's top-level code license and the redistribution status of the
imported Go2 assets are not yet finalized. Do not publish a release until both
items marked unresolved in `THIRD_PARTY_NOTICES.md` have been cleared.
