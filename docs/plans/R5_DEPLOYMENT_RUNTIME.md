# R5：部署 Runtime 与适配

状态：`⬜ NOT ACTIVE`。

## 目标

完成从已验证 PolicyArtifact 到安全 simulation deployment 的独立数据面。

## 进入条件

- R4 已证明 MotionSkill、PolicyArtifact 与 ValidationRun 的固定关联；
- observation/action/deployment schema、控制频率和 joint order 可由 Artifact 读取；
- 当前阶段已在 `plans/README.md` 激活。

## 本阶段不做

- 不把 WebUI、API 或 WebSocket 当成安全控制环；
- 不接入未完成 Driver、Calibration 和 SafetyProfile 的真实机器人；
- 不在通用 FSM、Policy runner 或 DeploymentPlan 中导入厂商 SDK 类型；
- 不因 simulation deployment 通过就宣称 hardware-approved。

## 工作项

| ID | 工作项 | 交付物 | 验收 |
|---|---|---|---|
| R5.1 | Runtime protocol | prepare/start/status/stop/safe、heartbeat、token | 非法转换和幂等 stop 测试 |
| R5.2 | ONNX runner | observation、inference、action、frequency、shape | 不一致时 fail closed |
| R5.3 | 通用 FSM | Passive/Stand/RL/Stopping/Safe | 异常和人工停止进入 Safe |
| R5.4 | Simulation driver API | RobotState/RobotCommand、timestamp、health | 不导入厂商 SDK 类型 |
| R5.5 | 自研机器人 sim-to-sim | simulator/driver、轨迹、指标和视频 | 使用 R3 同一 Artifact |
| R5.6 | DeploymentPlan | target、Artifact、runtime、gates、fallback | 输入固定可 diff |
| R5.7 | DeploymentSession | PREPARING/ARMED/ACTIVE/STOPPING/SAFE/FAILED | 终态可恢复 |
| R5.8 | Fault injection | heartbeat loss、shape error、disconnect、timeout | 自动 SAFE 证据 |
| R5.9 | Physical gate | Driver/Calibration/Safety/Profile | 缺任一项不可激活 |

## 退出条件

平台能够启动自研机器人 simulation DeploymentPlan，进入 ACTIVE，并通过 stop 或故障注入进入 SAFE；WebUI/API 断开不影响
Runtime 进入安全终态，且完整 session、事件和证据可追溯。
