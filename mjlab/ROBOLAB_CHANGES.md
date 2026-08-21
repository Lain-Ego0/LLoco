# RoboLab Changes to MJLab

This ledger tracks downstream modifications made to the MJLab 1.6 source under target path `mjlab/`. Before the R0.4
pure path-move commit, the same files temporarily remain under `vendor/mjlab/`.

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

No downstream behavior changes recorded yet.
