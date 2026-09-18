# OpenDoge (OpenDog V1.1)

OpenDoge is the OpenDog V1.1 quadruped migrated from
`OpenDoge_train/resources/robots/OpenDogV1.1`.

- `urdf/opendoge.urdf` preserves the original SolidWorks URDF as the source of
  truth.
- `xmls/opendoge.xml` is the MuJoCo model used by LainLab. It keeps the URDF
  link inertials, body offsets, joint axes/limits, and collision shapes, and
  adds an `imu` site plus the sensors expected by mjlab.
- `xmls/assets/` contains the original STL meshes.
- Foot collisions are named `FR_foot_collision`, `FL_foot_collision`,
  `RR_foot_collision`, and `RL_foot_collision`; foot sites are `FR`, `FL`,
  `RR`, and `RL`.

The flat velocity profile scales height, foot clearance, command ranges, and
debug visualization to the smaller OpenDoge body instead of reusing Go2-sized
values directly. OpenDoge is an in-house robot and belongs to the `LainLab`
task group; both flat and rough velocity tasks are registered.

## Regenerating the MJCF

`xmls/opendoge.xml` is generated from `urdf/opendoge.urdf` with the committed
conversion script. Do not edit the XML by hand for source-geometry changes.

```bash
# From the repository root
.venv/bin/python -m src.assets.robots.opendoge.convert_urdf
```

The script uses MuJoCo's URDF importer for inertias, joint transforms, limits,
and collision primitives, then applies the mjlab-specific post-processing
(base link/freejoint, visual geoms, `imu` site, foot sites, collision names,
and IMU sensors). Run it after any URDF or mesh update.
