# FireDog 2.2 — RoboLab R2 simulation-only Profile

公共 Profile ID：`community.firedog2_2`，版本 `1.0.0`。原始用户资产保留在本目录；目录名不是公共 Registry ID。

入口文件：

- `profile.yaml` — `RobotProfile v1beta1`；
- `joint_set.yaml` — 16 个 canonical joints，四腿顺序 `RF/RR/LR/LF`；
- `model/firedog2_2.xml` — 由原始 URDF 确定性生成的 MJCF；
- `bindings/actuator_sensor_mapping.yaml` — 16 actuator、34 sensor 的单位/方向/限位/索引映射；
- `bindings/task_binding.yaml` — 通用 `velocity_tracking` task binding；
- `snapshots/resolved_config.snapshot.json` — Profile + task + MJLab/toolchain 的排序快照；
- `provenance.yaml` — 原始文件大小、SHA-256、SW2URDF exporter、授权边界和派生链。

当前能力严格为 simulation-only：physical target disabled，motor ID 为 null，不包含 Driver、SDK 或真实策略 checkpoint。
`package.xml` 声明 BSD 仅记录为来源声明；由于作者/维护者是 TODO，CAD/STL/URDF/CSV 的再分发授权仍未确认。

