# Development tools

Repository-level validation and development scripts belong here. Product code
must remain importable from `src/lainloco`; reusable validation logic belongs
in the package and may be exposed here through a thin script.

`run_go2_validation.sh` runs the long-form migrated-task training matrix and
writes its generated checkpoints and status files beneath `runs/` by default.
