# L0 Validation Record

Validated on 2026-08-20 with the local `mjlab` Conda environment and MuJoCo.

```text
nq=36  nv=35  nu=0  njnt=30  nsite=6
root body: pelvis
IMU site: imu_in_pelvis
foot sites: left_foot, right_foot
```

The model has one free joint and 29 articulated joints. `nu=0` is expected:
the source MJCF contains no actuators, while MJLab attaches
`BuiltinPositionActuator` instances during task construction. The canonical
29-DoF actuator mapping is therefore recorded in
`tests/contract/fixtures/joint_set.g1_29dof.yaml`, not fabricated in the XML.

The source XML SHA-256 is
`56539bc76eadb05dd439c47de94df52130ea8fa243d08bdddd9cbc32dd4c78a0`.
