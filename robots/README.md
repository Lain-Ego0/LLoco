# Robot Profiles

状态：R2 当前固定接入 FireDog 2.2，公共 Profile ID 为 `community.firedog2_2`。目录名保留用户提供的原始资产命名，
不作为公共 Registry ID，也不向 Skill 或公共 API 暴露路径。

## 职责

存放与 Skill 解耦的 RoboLab Robot Profile：描述“这台机器人是什么、关节如何映射、怎样通信、怎样保证安全”。

- 每个 Profile 包含：MJCF 引用与诊断、canonical joint 映射、控制频率与默认姿态、仿真/实机 target、安全边界和 capabilities 声明；
- Profile 描述型号；某一台具体设备的网络/地址等机器相关值属于本地 Device Instance 配置，不进入版本化 Profile；
- 只有 `physicalDeployment` 及必需 capability 全部就绪，WebUI 才允许激活 physical target。

## 边界

- 本目录只放 RoboLab 自有 Profile，不放具体 Skill（策略/动作产物属于 MotionSkill），也不放厂商 Driver 代码（属于 [`integrations/README.md`](../integrations/README.md)）；
- 机器人 MJCF/mesh 等外部资产可以保留在来源命名空间，Profile 通过来源、revision 和哈希固定，不复制不明许可证的内容；
- Profile schema（`RobotProfile v1alpha1`）已在 B1 冻结；R0.6 后测试使用明确标记为 fixture 的中性 Profile，直到 R2 选定真实参考机器人。

字段规范见 [Robot Profile 规范](../docs/specifications/ROBOT_PROFILE.md)。
