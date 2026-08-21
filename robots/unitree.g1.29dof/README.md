# Unitree G1 29DoF RobotProfile

This is RoboLab's first simulation-only RobotProfile. It references the curated
MJLab vendor model rather than copying XML, meshes, or runtime dependencies.

- MJCF: `model/g1.xml`, linked to the curated vendor asset directory.
- Provenance: `bindings/simulation.yaml` records the upstream revision, license,
  and SHA-256 of `g1.xml`.
- Current compatibility task bindings: `Unitree-G1-Flat` and `Unitree-G1-Rough`.
  These are `unitree_compat` implementation IDs. They will be hidden behind stable RoboLab
  `TaskDefinition` IDs before `mjlab_native` becomes the default backend.
- Canonical mapping: `g1.29dof.canonical.v1`, currently represented by the B1
  contract fixture at `tests/contract/fixtures/joint_set.g1_29dof.yaml`.

`physicalDeployment` is deliberately disabled. This package establishes L0
model identity and contract validation only; it does not assert hardware safety
or deployment readiness.
