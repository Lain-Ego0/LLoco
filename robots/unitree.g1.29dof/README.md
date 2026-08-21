# Unitree G1 29DoF RobotProfile

Status: deprecated and scheduled for deletion in R0.6.

This was RoboLab's first historical simulation-only RobotProfile. It references the curated
Unitree legacy model rather than copying XML, meshes, or runtime dependencies. It is one commercial-robot
example, not the definition of the general RoboLab robot contract.

- MJCF: `model/g1.xml`, linked to the curated vendor asset directory.
- Provenance: `bindings/simulation.yaml` records the upstream revision, license,
  and SHA-256 of `g1.xml`.
- Current legacy task bindings: `Unitree-G1-Flat` and `Unitree-G1-Rough`.
  These are Unitree implementation IDs and must remain private to the legacy adapter. The Customized MJLab 1.6
  path will bind G1 through stable RoboLab `TaskDefinition` IDs.
- Canonical mapping: `g1.29dof.canonical.v1`, currently represented by the B1
  contract fixture at `tests/contract/fixtures/joint_set.g1_29dof.yaml`.

`physicalDeployment` is deliberately disabled. This package establishes L0
model identity and contract validation only; it does not assert hardware safety
or deployment readiness.

The profile and its vendor symlinks will not be copied into the Customized MJLab 1.6 path. Any future G1 support must be
implemented as a new adapter through the common RoboLab contracts.
