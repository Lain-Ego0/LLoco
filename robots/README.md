# Robot Profiles

状态：`unitree.g1.29dof@1.0.0` 是首个正式 simulation-only Profile；模型与任务资产引用 `vendor/unitree_rl_mjlab/` 中已有内容，不复制 vendor 资产。

## 职责

存放与 Skill 解耦的 RoboLab Robot Profile：描述“这台机器人是什么、关节如何映射、怎样通信、怎样保证安全”。

- 每个 Profile 包含：MJCF 引用与诊断、canonical joint 映射、控制频率与默认姿态、仿真/实机 target、安全边界和 capabilities 声明；
- Profile 描述型号；某一台具体设备的网络/地址等机器相关值属于本地 Device Instance 配置，不进入版本化 Profile；
- 只有 `physicalDeployment` 及必需 capability 全部就绪，WebUI 才允许激活 physical target。

## 边界

- 本目录只放 RoboLab 自有 Profile，不放具体 Skill（策略/动作产物属于 MotionSkill），也不放厂商 Driver 代码（属于 [`integrations/`](../integrations/)）；
- 机器人 MJCF/mesh 等上游资产保留在 vendor 命名空间，Profile 通过引用与哈希固定，不复制；
- Profile schema（`RobotProfile v1alpha1`）已在 B1 冻结；使用 `robolab check robots/unitree.g1.29dof/profile.yaml` 验证 manifest。

字段规范见 [ROBOT_ADAPTATION](../docs/ROBOT_ADAPTATION.md)。
