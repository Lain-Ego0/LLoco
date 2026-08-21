# MJLab Upstream Record

- Repository: https://github.com/mujocolab/mjlab
- Upstream tag: `v1.6.0`
- Upstream commit: `0fb8a681136be94ffc636a3dd423cabb97d91f10`
- Package version: `1.6.0`
- License: Apache License 2.0; see `LICENSE`
- Target RoboLab source location: `mjlab/`
- Transitional location before R0.4: `vendor/mjlab/`
- Verified at: 2026-08-21

## Verification

The RoboLab tree was compared recursively against a fresh shallow checkout of upstream tag `v1.6.0` at
`0fb8a681136be94ffc636a3dd423cabb97d91f10`, excluding only the upstream checkout's `.git` directory. No file or
content differences were reported at the time of verification.

This file establishes the starting point for RoboLab's Customized MJLab 1.6 downstream distribution. Future upstream
imports and RoboLab-specific changes must be made in separate commits and recorded in `ROBOLAB_CHANGES.md`.

## Update Rules

1. Resolve the exact target upstream commit before changing source files.
2. Record the old and new upstream commits and the synchronization method.
3. Keep upstream-import changes separate from RoboLab-specific behavior changes.
4. Run the MJLab smoke, contract and representative task regression suites.
5. Record conflicts, intentionally retained downstream behavior and the rollback revision.
