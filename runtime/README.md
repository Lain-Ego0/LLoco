# RoboLab Runtime（接口边界，未实现）

状态：本目录目前只声明所有权与接口边界，没有平台代码。实现见 [DEVELOPMENT_PLAN](../docs/project/DEVELOPMENT_PLAN.md) R5。

## 职责

与厂商无关的部署运行时（数据面）：

- 共享 C++ FSM：Passive / FixStand / Damping / RL / E-stop 等通用生命周期与安全转换；
- ONNX 推理 runner、observation/action 构造、参数加载；
- watchdog、命令超时、限幅、姿态/速度边界与故障降级；
- 遥测记录与 session 管理；`drivers/` 子目录承载 `unitree_sdk2` 等硬件驱动插件。

## 边界

- 本目录不依赖任何厂商 SDK 类型；厂商专用逻辑通过 driver 插件和 [`integrations/`](../integrations/) 接入；
- Unitree legacy deploy 在 R0.6 从 active tree 删除；Runtime 设计只依据通用部署契约，必要历史实现从 Git 查看；
- Edge Runtime 必须独立进程运行，实机控制循环不依赖 WebUI/WebSocket 在线。

设计细节见 [ARCHITECTURE §2、§5](../docs/reference/ARCHITECTURE.md) 与 [ROBOT_ADAPTATION §6](../docs/reference/ROBOT_ADAPTATION.md)。
