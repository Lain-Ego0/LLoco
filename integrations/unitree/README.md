# Unitree Integration（具体厂商适配接口，未实现）

状态：已废止，等待 R0.6 删除。本目录只有接口说明，没有平台代码，不保留空的 Unitree 扩展承诺。

## 职责

承载 Unitree 专用的 Driver 与 Robot Profile 适配：

- Unitree 专用 DDS topic、mode check、motor ID 和 message type；
- `unitree_sdk2` / CycloneDDS 通信的接入与连接健康度；
- Unitree 机型（G1、Go2、H1 等）的 Profile 绑定与硬件映射参考。

## 边界

- Unitree 训练/仿真/legacy 部署的**上游源码**在 [`vendor/unitree_rl_mjlab/`](../../vendor/unitree_rl_mjlab/)，本目录不放上游代码副本；
- 与厂商无关的 FSM、推理、安全和遥测放 [`runtime/`](../../runtime/)；
- 机器人型号描述（simulation-only Profile）放 [`robots/`](../../robots/)；
- 未来如重新支持 Unitree，应重新创建 integration，并通过当时已经稳定的通用 Runtime/Robot Profile 契约实现；不得从本占位目录延续旧接口。

设计细节见 [ARCHITECTURE §4.3](../../docs/reference/ARCHITECTURE.md) 与 [MJLAB_MAINTENANCE §6](../../docs/history/MJLAB_MAINTENANCE.md)。
