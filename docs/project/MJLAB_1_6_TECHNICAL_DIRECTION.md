# MJLab 1.6 深度定制技术主线

状态：2026-08-21 生效。本文是 RoboLab 仿真、运动控制工具链和机器人适配路线的最高级技术决策之一。

## 1. 项目定义

RoboLab 是一个基于深度定制 MJLab 1.6 工具链、面向商业成品机器人、实验室自研机器人、爱好者机器人和比赛机器人
的一站式运动控制开发与部署平台。

项目愿景的完整英文表述为：

> RoboLab: A One-Stop Motion Control Platform Built on a Heavily Customized MJLab Toolchain, with Robot Deployment Adaptation and Skill Integration.

这里的“深度定制 MJLab”是项目的技术主线，不是宣传用语。它表示 RoboLab 将在固定的上游 MJLab 1.6 基础上维护
自己的下游版本、补丁和扩展，并对任务、机器人、训练、回放、评测、导出和部署元数据拥有明确的工程所有权。

## 2. 本决策解决的问题

项目早期选择 `unitree_rl_mjlab` 作为第一个实现入口，便于快速验证 G1 资产、任务和部署经验，但这一路径不能定义
RoboLab 的长期架构。Unitree RL MJLab 建立在较旧的 MJLab 版本和 Unitree 特定工程布局之上，适合作为历史参考、
G1 适配来源和兼容路径，不适合作为面向自研机器人的平台核心。

因此，RoboLab 的正式技术基座调整为 MJLab 1.6。Unitree 相关代码在过渡期只作为退役参考，不再决定公共 task ID、
Skill manifest、平台目录、训练入口或 Runtime 协议；R0.6 完成后不再存在于 active working tree。

## 3. 术语的严格含义

### 3.1 One-Stop Motion Control Platform

“一站式”指同一套可追踪、可复现的工作流覆盖：

1. 机器人模型和 Robot Profile 接入；
2. 运动控制任务、MDP、观测和动作定义；
3. 训练、回放、评测和策略导出；
4. Policy Artifact、配置、日志、指标和视频的 lineage 管理；
5. Skill 安装、兼容性检查和能力复用；
6. 面向目标机器人的部署适配、仿真部署和安全 Runtime。

它不是云训练、多用户平台或通用低代码系统。首版仍然是本地优先的工程工作台。

### 3.2 Heavily Customized MJLab Toolchain

“深度定制”不要求重写 MuJoCo、MuJoCo-Warp 或 RSL-RL，也不以修改文件数量作为标准。它要求 RoboLab 对运动控制
工具链拥有实质控制权，至少包括：

- Robot Registry 与 MJLab robot/config 的统一绑定；
- 可扩展的 task、MDP、observation、action、reward 和 termination 体系；
- schema 化的 train/play/evaluate/export 入口；
- 可生成并验证 observation/action/deployment schema 的策略产物格式；
- 面向自研执行器、传感器和控制周期的适配点；
- 结构化指标、回归测试和验证证据；
- 与 Skill、Robot Profile、Artifact 和 DeploymentPlan 的稳定关联。

RoboLab 可以修改 MJLab 源码，但 WebUI、Skill Registry、Artifact Store、Job Worker 和 Edge Runtime 不应全部塞入
MJLab 核心包。MJLab 是运动控制仿真和训练工具链基座，RoboLab Platform 是围绕它构建的控制面与部署系统。

### 3.3 Robot Deployment Adaptation

部署适配包括但不限于：

- joint、actuator、sensor、frame 和坐标系映射；
- observation/action 的运行时构造与校验；
- 控制频率、动作缩放、限幅和默认姿态；
- ONNX/策略推理与机器人 Runtime 的连接；
- 仿真、sim-to-sim 和实机 Driver 的边界；
- heartbeat、watchdog、stop、safe 和故障回退。

它必须同时服务成品机器人和自研机器人。Unitree 只是其中一个 Driver/Robot Profile 实现。

## 4. 基座和代码所有权

RoboLab 使用固定上游 revision 的 MJLab 1.6 作为下游定制基座。该基座位于仓库一级目录 `mjlab/`，而不是
`vendor/` 命名空间。`mjlab/` 保持自己的 `pyproject.toml`、源码、测试、upstream record 和 RoboLab change ledger；
它与 RoboLab Platform 同仓库但拥有独立的代码和发布边界。后续可在边界稳定后拆分为独立的 `RoboLab-MJLab` 仓库，
但拆仓不是当前阶段的前置条件。

维护要求：

- 记录上游仓库、commit、版本和同步范围；
- 上游导入、RoboLab 补丁和平台扩展保持可区分；
- 每项核心修改标记为 RoboLab-specific 或候选 upstreamable；
- 保留 MJLab 及其依赖的原许可证、NOTICE 和版权声明；
- 用 smoke test、contract test 和代表性任务回归测试保护上游同步；
- 不把 Unitree 专用目录结构复制为通用平台接口。

## 5. Unitree 的最终定位

`vendor/unitree_rl_mjlab/` 的历史定位曾经是：

- G1 资产、参数和任务的历史来源；
- 早期 G1 兼容路径和 checkpoint 诊断工具；
- Unitree sim-to-sim 参考实现；
- Unitree Driver/Robot Profile 的技术资料。

该目录将在 Unitree retirement 批次中从 active tree 删除。删除后，历史 Git 记录和上游 commit 仍可审计和恢复，但它不再是：

- RoboLab 的默认仿真训练基座；
- 公共 API 或 Skill 的命名来源；
- MJLab 1.6 定制工作的前置条件；
- native 工具链发布的行为裁判。

## 6. 验收原则

RoboLab 不要求 MJLab 1.6 与 Unitree 旧路径逐帧或逐 reward 等价。两者版本、物理实现和配置栈不同，合理的行为差异
应通过迁移报告记录，而不是阻止新工具链发展。

必须分别验收：

1. **基座回归**：MJLab 1.6 基础环境、viewer、训练入口和依赖可运行；
2. **工具链能力**：RoboLab 定制的 task/train/play/evaluate/export 契约有效；
3. **机器人接入**：至少一个自研或非 Unitree 机器人可以完成 simulation-first 接入；
4. **策略质量**：使用版本化运动指标验收，不要求复刻旧 checkpoint；
5. **部署安全**：Runtime 的状态机、watchdog、stop/safe 和故障注入通过；
6. **可复现性**：Skill、Profile、Artifact、配置和代码 revision 可追溯。

Unitree 路径只参加兼容性和迁移回归测试。

## 7. 对后续文档和计划的强制约束

任何新文档不得再把以下表述作为当前主线：

- “先完成 Unitree G1 黄金基线，才能开始 MJLab 1.6 主线”；
- “双后端等价后才允许 native 开发或成为产品主线”；
- “RoboLab 不 fork 或修改 MJLab”；
- “Unitree 是长期平台 backend”；
- “自研机器人属于后期可选扩展”。

历史文档可以保留这些表述，但必须注明它们属于已废止的 Unitree-first 方案。

Unitree 整栈退役和 MJLab 根目录迁移的具体删除清单见
[`UNITREE_RETIREMENT_AND_MJLAB_RELOCATION.md`](UNITREE_RETIREMENT_AND_MJLAB_RELOCATION.md)。
