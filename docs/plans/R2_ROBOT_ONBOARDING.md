# R2：自研机器人接入

状态：`⬜ NOT ACTIVE`。

## 目标

选择一个来源和许可证清晰的自研、爱好者或社区可复现参考机器人，证明 RoboLab 的机器人抽象适用于自研、爱好者和比赛机器人。

## 进入条件

- R1 退出条件全部通过；
- Robot/Task Registry 和配置生成接口稳定；
- `plans/README.md` 将 `ACTIVE_STAGE` 明确切换为 R2。

## 本阶段不做

- 不把测试占位符当成真实机器人；
- 不因方便重新导入 G1 或其他厂商项目目录；
- 不要求真实硬件 Driver；
- 不训练高质量策略。

## 工作项

| ID | 工作项 | 交付物 | 验收 |
|---|---|---|---|
| R2.1 | 选择参考机器人 | 来源、revision、许可证、模型和目标 task 记录 | 机器人来源可复现，资产使用边界明确 |
| R2.2 | MJCF Inspector | body/joint/actuator/sensor/frame/mesh 诊断 | 错误定位到名称和路径 |
| R2.3 | Robot Profile vNext | description、simulation、task、runtime binding | simulation-only 不要求厂商 SDK |
| R2.4 | Actuator/Sensor mapping | 类型、单位、方向、limit、gear、frequency、latency | schema 正反例通过 |
| R2.5 | Config generation | Profile/MJCF 到 MJLab config | snapshot 可复现 |
| R2.6 | 最小仿真 | load、reset、named action、observation、viewer | 无厂商代码和路径依赖 |

## 退出条件

真实参考机器人通过 Profile 加载到 Customized MJLab 1.6，完成 reset、动作驱动、观测生成、配置快照和 Viewer 证据；不得把
厂商 SDK 或品牌目录作为 simulation-first 前置依赖。
