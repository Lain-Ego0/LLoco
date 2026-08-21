# RoboLab 下一阶段详细执行计划

状态：v0.2，2026-08-21 启用。本文是当前批次、依赖、完成状态与剩余工作的
**唯一权威来源**。

历史 B1–B7 计划已归档至
[`../history/PHASE_0_1_DEVELOPMENT_PLAN.md`](../history/PHASE_0_1_DEVELOPMENT_PLAN.md)。
本阶段以 [`MAINLINE_ALIGNMENT_REVIEW.md`](MAINLINE_ALIGNMENT_REVIEW.md) 的主线审查为范围约束，
以 [`B7_ACCEPTANCE_RECORD.md`](B7_ACCEPTANCE_RECORD.md) 记录 N0 的验收证据。

## 1. 阶段目标

下一阶段不再扩展横向平台功能，集中完成一条真实运动控制纵向闭环：

```text
G1 trained MotionSkill
  -> MJLab play
  -> ValidationRun
  -> unitree_mujoco sim-to-sim
  -> DeploymentPlan
  -> DeploymentSession
  -> Runtime stop/safe
```

阶段退出时，RoboLab 必须能够从同一个 Skill、Robot Profile 和固定 Artifact hash 出发，
完成真实策略回放、验证、仿真部署、会话监控和安全停止，并保存完整 lineage。

## 2. 当前状态与资源门禁

- ✅ Phase 0–1 平台骨架：schema、Skill、CLI、Worker、API、WebUI 与 Artifact Store 已实现；
- ✅ CPU contract suite：2026-08-21 实测 `113 passed, 1 warning`；
- ✅ G1 zero-policy 已启动 MJLab/Viser，adapter 布尔参数与 PlayConfig 缺陷已修复；
- ⏸️ 当前 GPU 正在运行其他 IsaacLab 训练，不启动 RoboLab/G1 GPU 任务，不抢占或终止该进程；
- ⬜ 外部训练完成后，先训练或取得可复现的 G1 velocity checkpoint；
- ⬜ 在真实 G1 trained play 通过前，N1–N5 只保留计划，不进入实现。

当前等待链：

```text
现有 GPU 训练结束
  -> 训练/取得 G1 velocity policy
  -> 固定 checkpoint + config + revision
  -> N0 trained MotionSkill 验收
  -> 开始 N1
```

## 3. 批次总览

| 批次 | 目标 | 依赖 | 状态 |
|---|---|---|---|
| N0 | 关闭 B7：真实 G1 trained MotionSkill 验收 | 当前 GPU 任务结束、G1 checkpoint | ⏸️ 等待外部训练资源 |
| N1 | 冻结验证与部署领域契约 | N0 | ⬜ 未开始 |
| N2 | 最小通用 Runtime 与 sim-to-sim target | N1 | ⬜ 未开始 |
| N3 | Validation/Deployment 控制面与安全门禁 | N2 | ⬜ 未开始 |
| N4 | Validate/Deploy WebUI 纵向入口 | N3 | ⬜ 未开始 |
| N5 | 端到端验收、回归与文档收口 | N4 | ⬜ 未开始 |

依赖原则：N0 未通过，不以 zero-policy 代替真实 MotionSkill；N2 未具备独立 Runtime 的
`stop/safe`，不开放 DeploymentSession；N3 安全门禁未通过，不显示可启动的 Deploy 操作。

## 4. N0：真实 G1 MotionSkill 与 B7 关闭门禁

目标：证明平台承载的是可运行运动能力，而不仅是 task discovery 和 Job 管理。

| # | 工作项 | 交付物 | 验收标准 | 状态 |
|---|---|---|---|---|
| N0.1 | 等待当前 IsaacLab 训练自然结束 | 可用 GPU 时间窗 | 不终止、不降速、不抢占当前训练 | ⏸️ |
| N0.2 | 训练或取得 G1 velocity policy | checkpoint/WandB run、训练配置、代码 revision、seed、环境版本 | 来源可复现，文件可读取，任务与 G1 29DoF Profile 匹配 | ⬜ |
| N0.3 | 固定 MotionSkill 产物 | policy artifact、deploy params、SHA-256、manifest revision | `robolab check` 与 compatibility 全部通过 | ⬜ |
| N0.4 | 从平台运行真实 trained play | Job input/events/logs/result、Viser 证据、可选视频 | 使用 `agent=trained` 成功启动并产生有效机器人运动，不能使用 zero-policy 代替 | ⬜ |
| N0.5 | 验证 Job 生命周期与 lineage | cancel/terminal status、config snapshot、artifact hashes | API/CLI 至少一条路径可复现，取消后无残留进程组 | ⬜ |
| N0.6 | 完成远程 catalog/CI 与 B7 记录 | 远程 CI 链接、最终验收记录 | B7.1/B7.2/B7.3 全部有证据后关闭 | ⬜ |

N0 退出条件：真实 G1 Velocity MotionSkill 从固定版本 catalog 安装后，可由 RoboLab 创建
trained play Job，运行、停止并归档完整结果。

## 5. N1：验证与部署领域契约

目标：在写 Runtime 和页面前，冻结从“已验证策略”到“仿真部署会话”的机器契约。

| # | 工作项 | 主要内容 | 验收标准 | 状态 |
|---|---|---|---|---|
| N1.1 | `ValidationRun` schema | Skill/Profile/Artifact hash、target、gate、metrics、evidence、status | schema 正反例与版本兼容测试通过 | ⬜ |
| N1.2 | `DeploymentPlan` schema | 固定输入、runtime config、target、required gates、fallback | 配置变化产生新 revision，不允许浮动 Skill 或 Artifact | ⬜ |
| N1.3 | `DeploymentSession` 状态机 | CREATED/PREPARING/ARMED/ACTIVE/STOPPING/SAFE/FAILED | 非法转换被机器拒绝并给出原因 | ⬜ |
| N1.4 | Runtime 协议 | prepare/start/status/stop/safe、heartbeat、session token、事件格式 | contract test 覆盖超时、重复 stop 和失联降级 | ⬜ |
| N1.5 | Safety Gate 判定 | offline、mjlab_play、sim_to_sim、target capability | 每项通过/失败均可解释且关联证据 hash | ⬜ |

N1 退出条件：CLI/测试可以在不启动仿真器的情况下构造计划、执行状态转换并验证所有安全门禁。

## 6. N2：最小 Runtime 与 sim-to-sim

目标：从 Unitree legacy deploy 中抽出最小可验证的数据面，不追求一次性重构全部机器人。

| # | 工作项 | 主要内容 | 验收标准 | 状态 |
|---|---|---|---|---|
| N2.1 | Runtime 骨架 | 独立进程、配置加载、结构化事件、退出码 | 不依赖 WebUI 在线，API 崩溃不保持 active 命令 | ⬜ |
| N2.2 | ONNX policy runner | observation、inference、action、frequency、shape 检查 | 使用 N0 同一 policy artifact，维度和频率异常时 fail closed | ⬜ |
| N2.3 | 通用 FSM | Passive/FixStand/Damping/RL/Safe 生命周期 | stop、超时、异常均进入 Safe/Damping | ⬜ |
| N2.4 | Simulation driver | unitree_mujoco transport、RobotState/RobotCommand 转换 | 厂商类型不泄漏到通用 runner/FSM | ⬜ |
| N2.5 | Heartbeat/watchdog | deadline、状态超时、命令超时、故障事件 | 人工断开控制面后 runtime 自动安全降级 | ⬜ |
| N2.6 | G1 sim-to-sim 样板 | G1 Profile binding、deploy config、启动脚本 | N0 policy 在独立 simulator/runtime 中完成稳定会话 | ⬜ |

N2 退出条件：命令行能够启动一条 G1 simulation DeploymentPlan，进入 ACTIVE，监控状态，
并通过显式 stop 或故障注入进入 SAFE。

## 7. N3：Validation/Deployment 控制面

| # | 工作项 | 主要内容 | 验收标准 | 状态 |
|---|---|---|---|---|
| N3.1 | Validation 服务与存储 | 创建/查询 ValidationRun，登记指标、日志、视频和证据 | 验证记录固定 Skill/Profile/Artifact hash | ⬜ |
| N3.2 | Deployment 服务与存储 | 创建 plan/session、prepare/start/stop/safe | 重启 API 后可恢复终态与审计记录 | ⬜ |
| N3.3 | Edge 管理面 | 启动/监控 Runtime、heartbeat、session token | 旧 token、错误 target、缺失 heartbeat 被拒绝 | ⬜ |
| N3.4 | 自动门禁 | compatibility + required validations + capability | 未通过 sim-to-sim 的 Artifact 不能启动 simulation deployment | ⬜ |
| N3.5 | Artifact lineage | checkpoint、ONNX、deploy config、validation、session 关联 | 任一 session 可追溯到代码、配置、Skill 和训练来源 | ⬜ |

N3 退出条件：REST/CLI 可以完成 validate、plan、prepare、start、status、stop/safe 全流程，
且绕过门禁的请求被明确拒绝。

## 8. N4：Validate/Deploy WebUI

| # | 工作项 | 主要内容 | 验收标准 | 状态 |
|---|---|---|---|---|
| N4.1 | Validate 页面 | 选择固定 Skill/Profile/Artifact、显示 gates/evidence | 不使用含糊的“已验证”，逐项显示结果和原因 | ⬜ |
| N4.2 | Deployment Plan 页面 | 显示 target、policy hash、配置 diff、fallback | 启动前可核对等价 CLI 和全部固定输入 | ⬜ |
| N4.3 | Session 页面 | FSM 状态、heartbeat、告警、事件时间线、stop/safe | 页面断开不影响 runtime watchdog | ⬜ |
| N4.4 | 危险操作交互 | simulation/physical 明确分离，physical 保持禁用 | 缺 Driver/Calibration/Safety 时不能激活 physical target | ⬜ |

N4 退出条件：用户无需拼接命令即可完成一次仿真部署，同时任何关键状态和失败原因都可见。

## 9. N5：端到端验收与阶段收口

| # | 工作项 | 验收标准 | 状态 |
|---|---|---|---|
| N5.1 | Happy path | 安装 G1 Skill -> trained play -> validate -> sim-to-sim -> stop/safe 全部通过 | ⬜ |
| N5.2 | Safety fault injection | API 断开、runtime heartbeat 超时、无效 token、错误 shape 均安全失败 | ⬜ |
| N5.3 | Reproducibility | 在干净环境依据 snapshot/hash 重放同一流程 | ⬜ |
| N5.4 | Regression suite | CPU contract tests 与必要 GPU/manual suite 分层记录 | ⬜ |
| N5.5 | 文档收口 | 架构、运行手册、验收记录、已知限制与回滚步骤同步 | ⬜ |

阶段退出条件：实现“一键仿真部署”的准确含义，并以真实 G1 policy 给出可复现证据；
physical target 仍保持不可激活，直到 Phase 4 的 Driver/Calibration/Safety 验收完成。

## 10. 范围冻结

本阶段明确不做：

- 多租户、云端训练、远端 Worker、Docker 平台化；
- 内置聊天 Agent、Claude/DeepSeek adapter 扩展；
- 第二机器人、完整 Robot Onboarding Wizard；
- CompositeSkill、签名市场和通用插件生态；
- 实机 motor command 激活。

只有 N5 关闭后，才重新评估训练平台、第二机器人和受控实机阶段。

## 11. 状态维护规则

1. 状态只使用 `⬜ 未开始`、`🔶 进行中`、`⏸️ 等待依赖`、`✅ 完成`；
2. 每次状态变化记录日期和证据路径，不以代码存在代替验收通过；
3. GPU/manual 验收与 CPU contract tests 分开记录；
4. 新增横向功能前必须说明它服务于上述纵向链路的哪一个节点；
5. 与主线审查冲突的工作，先更新决策记录和主线审查，再进入本计划。
