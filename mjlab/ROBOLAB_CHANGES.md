# RoboLab Changes to MJLab

This ledger tracks downstream modifications made to the MJLab 1.6 source under target path `mjlab/`.

The initial baseline exactly matches upstream `mujocolab/mjlab@0fb8a681136be94ffc636a3dd423cabb97d91f10`
(`v1.6.0`) as recorded in `UPSTREAM.md`. No RoboLab source modification had been recorded when this ledger was created.

## Required Entry Format

Every behavior-changing downstream modification must add an entry with all fields below:

```text
ID: RMJ-XXXX
Date: YYYY-MM-DD
RoboLab commit: <commit or PENDING before commit>
Files: <exact paths>
Purpose: <user-visible or architectural reason>
Type: robolab-specific | upstreamable | temporary-compatibility
Upstream issue/PR: <URL or none>
Compatibility impact: <config/API/checkpoint/runtime impact>
Tests: <exact commands and evidence paths>
Rollback: <commit or explicit revert instructions>
```

Documentation-only edits that do not change MJLab behavior may be grouped, but source, configuration, CLI, dependency and
public API changes must be recorded individually.

## Change Entries

ID: RMJ-0001
Date: 2026-08-21
RoboLab commit: PENDING
Files: mjlab/src/mjlab/asset_zoo/robots/unitree_g1/, mjlab/src/mjlab/asset_zoo/robots/unitree_go1/, mjlab/src/mjlab/tasks/velocity/config/g1/, mjlab/src/mjlab/tasks/velocity/config/go1/, mjlab/src/mjlab/tasks/tracking/config/g1/, related MJLab tests and scripts
Purpose: Retire Unitree-specific upstream assets, task registrations and legacy demonstrations from the active RoboLab tree.
Type: robolab-specific
Upstream issue/PR: none
Compatibility impact: Unitree task IDs, asset aliases and related demos are no longer available in the active MJLab 1.6 checkout; R2 must reintroduce any robot only through a reviewed public Robot Profile contract.
Tests: `python -m pytest -q mjlab/tests/smoke_test.py mjlab/tests/test_task_configs.py`; `python -m mjlab.scripts.list_envs`
Rollback: Revert this entry and restore the deleted paths from the fixed upstream commit; do not restore the retired Unitree legacy stack.
