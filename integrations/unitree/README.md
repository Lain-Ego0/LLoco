# Unitree Integration（接口边界，未实现）

状态：本目录目前只声明所有权与接口边界，没有平台代码。Unitree 是 RoboLab 的首个厂商适配器，不是通用 API 的命名来源。

## 职责

承载 Unitree 专用的 Driver 与 Robot Profile 适配：

- Unitree 专用 DDS topic、mode check、motor ID 和 message type；
- `unitree_sdk2` / CycloneDDS 通信的接入与连接健康度；
- Unitree 机型（G1、Go2、H1 等）的 Profile 绑定与硬件映射参考。

## 边界

- Unitree 训练/仿真/legacy 部署的**上游源码**在 [`vendor/unitree_rl_mjlab/`](../../vendor/unitree_rl_mjlab/)，本目录不放上游代码副本；
- 与厂商无关的 FSM、推理、安全和遥测放 [`runtime/`](../../runtime/)；
- 机器人型号描述（simulation-only Profile）放 [`robots/`](../../robots/)；
- 迁移路径：vendor `deploy/robots/*` 中的 Unitree 专用部分逐步迁入本目录，共享部分迁入 `runtime/`，见 [ROBOT_ADAPTATION §10](../../docs/ROBOT_ADAPTATION.md)。

设计细节见 [ARCHITECTURE §4.3](../../docs/ARCHITECTURE.md) 与 [MJLAB_MAINTENANCE §6](../../docs/MJLAB_MAINTENANCE.md)。
