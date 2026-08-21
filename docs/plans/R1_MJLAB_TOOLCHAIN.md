# R1：MJLab 定制工具链基础

状态：`🔶 ACTIVE`。

## 目标

建立由 RoboLab 控制的 MJLab 1.6 motion toolchain，使 train/play/evaluate/export、Robot/Task Registry 和 Artifact metadata
不依赖已删除的 Unitree 代码。

## 进入条件

- R0 完成；
- `mjlab/` upstream、环境、smoke 和 change ledger 可用；
- contract suite 通过；
- active code 不含 Unitree legacy 入口。

## 本阶段不做

- 不选择或接入真实 R2 参考机器人；
- 不启动 GPU 训练；
- 不实现 WebUI Train/Validate 完整页面；
- 不实现部署 Runtime；
- 不恢复旧 `mjlab tasks/play` vendor 行为。

## 工作项

| ID | 工作项 | 交付物 | 验收 |
|---|---|---|---|
| R1.1 | Toolchain identity | `robolab_mjlab@1`、upstream revision、RoboLab revision | Job/Artifact 可记录三者 |
| R1.2 | Robot Registry | 稳定 robot/profile 到 MJLab entity/config 的绑定 | 不含厂商路径或类型 |
| R1.3 | Task Registry | 稳定 task ID、版本、能力和配置 schema | task 与具体机器人 binding 分离 |
| R1.4 | Train entry | resolved config、seed、resume、资源和输出目录 | CPU 命令构造与配置测试通过 |
| R1.5 | Play entry | checkpoint/agent、viewer、录制和确定性配置 | 直接使用 Customized MJLab 1.6 |
| R1.6 | Evaluate entry | 场景、episode、metric、threshold 和证据结构 | 生成机器可读结果 |
| R1.7 | Export entry | checkpoint/ONNX 和 metadata | schema、频率、scale、joint order、hash 完整 |
| R1.8 | 回归测试 | import、registry、config、commands、minimal rollout | CPU suite；GPU 使用独立 marker |

## 机器验收

- Platform/CLI 能发现 RoboLab 稳定 task，而不是直接暴露上游环境 ID；
- train/play/evaluate/export 都能构造结构化 JobCommand；
- Toolchain revision 和最终 resolved config 可持久化；
- observation/action/deployment metadata 有明确生产者；
- contract suite 和 MJLab smoke 通过；
- `rg` 不出现已退役 Unitree adapter、vendor path 或 G1 task ID。

## 退出条件

RoboLab 可以只依赖根目录 `mjlab/` 和自己的 toolchain 契约执行 motion jobs。完成证据记录后，才能将
`ACTIVE_STAGE` 切换为 R2。
