# RoboLab MJLab 1.6 主线开发计划

状态：v1.0，2026-08-21 启用。本文是当前批次、依赖、完成状态和剩余工作的唯一权威来源。

旧版 Unitree-first N0–N5 计划已经被本计划取代。Unitree 整栈退役和 MJLab 根目录迁移的强制执行说明见
[`UNITREE_RETIREMENT_AND_MJLAB_RELOCATION.md`](UNITREE_RETIREMENT_AND_MJLAB_RELOCATION.md)。历史 B1–B7 实现和验收事实继续保留在
[`../history/PHASE_0_1_DEVELOPMENT_PLAN.md`](../history/PHASE_0_1_DEVELOPMENT_PLAN.md) 与
[`B7_ACCEPTANCE_RECORD.md`](B7_ACCEPTANCE_RECORD.md)，但不再作为 MJLab 1.6 主线的串行前置门禁。

最高级技术决策见 [`MJLAB_1_6_TECHNICAL_DIRECTION.md`](MJLAB_1_6_TECHNICAL_DIRECTION.md)。

## 1. 阶段目标

本阶段将项目从 Unitree/MJLab 1.2 兼容入口切换到 RoboLab 深度定制 MJLab 1.6 主线，并完成一条同时适用于自研和
成品机器人的运动控制纵向闭环：

```text
Robot MJCF / Robot Profile
  -> RoboLab Customized MJLab 1.6
  -> Task / Train / Play / Evaluate / Export
  -> MotionSkill / PolicyArtifact
  -> ValidationRun / sim-to-sim
  -> DeploymentPlan / Runtime
  -> stop / safe / reproducible evidence
```

阶段退出时必须证明：

1. 平台默认运动工具链不依赖 `vendor/unitree_rl_mjlab`；
2. 至少一个非 Unitree 参考机器人完成 simulation-first 接入；
3. 至少一个真实运动策略通过训练或可信 checkpoint 回放、评测和导出；
4. 同一 Artifact 可以进入 Skill、Validation 和 simulation deployment；
5. Runtime 在显式停止和故障条件下进入 SAFE；
6. G1 如被接入，只作为使用相同公共契约的商业机器人适配器。

## 2. 当前事实

- ✅ B1–B6 平台骨架已实现：schema、Skill、CLI、Worker、API、WebUI 和 Artifact Store；
- ✅ R0 开始前 CPU contract baseline：`113 passed, 1 warning`；R0.6 退役后重新运行结果为 `105 passed, 1 warning`；
- ✅ 当前 `mjlab/` 已包含 MJLab 1.6.0 源码；R0.4 路径迁移已完成；
- ✅ Unitree 旧栈已从 active tree 退役；历史来源、许可证事实和 Git 历史仍保留；
- ✅ 历史 G1 zero-policy 曾启动旧 MJLab/Viser 路径；该证据随 Unitree 栈退役，仅作为历史记录，不是当前能力验收；
- ✅ MJLab 1.6 upstream commit、RoboLab 修改记录和默认环境已正式固定；
- ⬜ RoboLab 定制的 Robot/Task/Train/Play/Evaluate/Export 工具链尚未实现；
- ⬜ 非 Unitree 自研参考机器人尚未选定；
- ⬜ Runtime、ValidationRun、DeploymentSession 和通用 sim-to-sim 尚未实现；
- ⏸️ 当前 GPU 被其他 IsaacLab 训练占用，不启动 RoboLab GPU 训练；CPU 可完成工作继续推进。

## 3. 执行规则

### 3.1 依赖规则

- R0 完成前，不修改 MJLab 核心行为；先固定 revision、环境、许可证和变更记录方式；
- R1 与 R2 可以在 R0 后交叉推进，但公共 schema 变更必须先有 contract test；
- R3 的真实策略质量验收依赖 GPU，但 R3 的配置、入口、指标和 CPU smoke test不依赖 GPU；
- R4 平台整合可以在 R3 训练等待期间推进，但不能把 zero-policy 标记为真实 MotionSkill；
- R5 Runtime 骨架可以与 R3/R4 并行，ACTIVE/SAFE 端到端验收需要固定 PolicyArtifact；
- R6 必须等待至少一个自研机器人闭环和一个成品机器人适配完成；
- Unitree G1 checkpoint、旧 play 和双后端等价报告不再阻塞 R0–R5。

### 3.2 完成规则

工作项只有同时具备代码、测试和证据路径才能标记 `✅ 完成`。禁止用以下内容代替完成：

- 只有 README 或接口占位；
- 只有 schema，没有生产者和消费者；
- 只有 zero-policy 或随机 action；
- 只有 WebUI 页面，没有真实 Job/Runtime；
- 只有命令成功退出，没有 Artifact、指标或状态证据；
- 只在 Unitree vendor 脚本中可运行。

### 3.3 状态值

只使用：`⬜ 未开始`、`🔶 进行中`、`⏸️ 等待外部依赖`、`✅ 完成`。

## 4. 批次总览

| 批次 | 目标 | 关键退出证据 | 状态 |
|---|---|---|---|
| R0 | 技术主线和 MJLab 1.6 基座固定 | upstream revision、环境锁、smoke test、修改记录 | 🔶 进行中 |
| R1 | RoboLab 定制 MJLab 工具链基础 | 不调用 Unitree 脚本的 train/play/evaluate/export | ⬜ 未开始 |
| R2 | 通用自研机器人接入 | 非 Unitree Profile 加载、诊断、reset、action、observation | ⬜ 未开始 |
| R3 | 真实运动策略工作流 | 训练/回放、指标、导出和 PolicyArtifact | ⬜ 未开始 |
| R4 | Skill、平台控制面和验证整合 | Skill 到 ValidationRun 的完整 lineage | ⬜ 未开始 |
| R5 | 部署 Runtime 和 sim-to-sim | DeploymentSession、watchdog、stop/safe | ⬜ 未开始 |
| R6 | 多机器人证明和发布收口 | 自研机器人 + 成品机器人共享同一契约 | ⬜ 未开始 |

## 5. R0：技术主线与基座固定

目标：在修改 MJLab 核心之前，消除版本、来源、环境和变更管理歧义。

| # | 工作项 | 交付物 | 验收标准 | 状态 |
|---|---|---|---|---|
| R0.1 | 同步项目决策和活动文档 | 技术方向、架构、路线图、环境、适配和 Skill 文档 | 活动文档已统一为 MJLab 1.6 主线；旧 B7/MVP 明确标记历史 | ✅ 完成（2026-08-21） |
| R0.2 | 固定 MJLab 1.6 上游来源 | `mjlab/UPSTREAM.md`（迁移前为 `vendor/mjlab/UPSTREAM.md`） | 已确认 `v1.6.0`/`0fb8a681...`，并与 fresh checkout 递归比较无差异 | ✅ 完成（2026-08-21） |
| R0.3 | 建立 RoboLab 修改账本 | `mjlab/ROBOLAB_CHANGES.md`（迁移前为 `vendor/mjlab/ROBOLAB_CHANGES.md`） | 已建立强制字段；当前基线尚无 RoboLab 行为修改 | ✅ 完成（2026-08-21） |
| R0.4 | 迁移 MJLab 到根目录 | `vendor/mjlab/` -> `mjlab/` 的纯路径迁移 | 内容和一级目录结构保持、历史可追踪、路径引用更新、无 nested `.git`；验收命令通过 | ✅ 完成（2026-08-21） |
| R0.5 | 建立默认 MJLab 1.6 环境 | `robolab-mjlab16` 环境、`mjlab/uv.lock`、环境安装说明 | 默认安装 `mjlab/`；Python/Torch/Warp/MuJoCo-Warp/RSL-RL 版本按锁文件固定 | ✅ 完成（2026-08-21） |
| R0.6 | 退役 Unitree 整栈 | 删除 vendor、adapter、G1 Profile、Integration、默认参数和专用测试 | 无破损 symlink；active code/docs/install 不依赖 Unitree；contract suite 通过 | ✅ 完成（2026-08-21） |
| R0.7 | MJLab 1.6 smoke test | CPU import/list-envs/最小模型加载证据；GPU 检查单独记录 | import、registry、最小模型加载成功；GPU 未执行时明确等待外部资源 | ✅ 完成（2026-08-21） |
| R0.8 | 建立修改与同步规则 | `mjlab/UPSTREAM.md`、`mjlab/ROBOLAB_CHANGES.md` | 上游同步有基线、冲突记录、回归和回滚步骤；行为修改使用强制登记字段 | ✅ 完成（2026-08-21） |

R0 退出条件：任何开发者都能确定 RoboLab 使用哪个 MJLab commit、哪些代码由 RoboLab 修改、如何安装默认环境、如何运行
最小测试，并且 active tree 中已经不存在 Unitree legacy 依赖。

## 6. R1：RoboLab 定制 MJLab 工具链基础

目标：建立真正由 RoboLab 控制的运动任务执行入口，不再把 vendor CLI 包装成平台核心。

| # | 工作项 | 主要内容 | 验收标准 | 状态 |
|---|---|---|---|---|
| R1.1 | Toolchain identity | `robolab_mjlab@1`、MJLab upstream revision、RoboLab revision | 每个 Job 和 Artifact 都记录三者 | ⬜ 未开始 |
| R1.2 | Robot Registry | Profile/robot ID 到 MJLab entity/config 的稳定绑定 | registry 不包含 Unitree vendor path | ⬜ 未开始 |
| R1.3 | Task Registry | 稳定 task ID、版本、能力、配置 schema 和 robot binding | task ID 不使用某厂商项目的环境名作为公共 ID | ⬜ 未开始 |
| R1.4 | Train entry | schema 化训练配置、seed、resume、资源和输出目录 | 命令直接运行定制 MJLab 1.6，输出结构化事件 | ⬜ 未开始 |
| R1.5 | Play entry | checkpoint/agent、viewer、录制、确定性配置 | 不调用 `vendor/unitree_rl_mjlab/scripts/play.py` | ⬜ 未开始 |
| R1.6 | Evaluate entry | 场景、episode、metrics、阈值和证据 | 结果为机器可读 Validation 片段 | ⬜ 未开始 |
| R1.7 | Export entry | checkpoint 到 ONNX/部署包和 metadata | 输入输出 schema、频率、缩放、joint order 和 hash 完整 | ⬜ 未开始 |
| R1.8 | 回归测试 | import、config、registry、命令构造、最小 rollout | CPU suite 可运行；GPU 测试有独立 marker | ⬜ 未开始 |

R1 退出条件：RoboLab 可以只依赖 Customized MJLab 1.6 构造并运行 train/play/evaluate/export Job，公共 API 不接收
`vendor_root` 或 vendor script path。

## 7. R2：通用自研机器人接入

目标：证明 RoboLab 的机器人抽象不是从 G1 反推出来的厂商接口。

### 7.1 参考机器人选择约束

R2 开始时必须选择并记录一个非 Unitree 参考机器人。它可以是：

- 项目所有者或比赛队已有的自研机器人；
- 许可证清晰、结构足够简单的公开机器人；
- 专门为接入测试建立的最小腿式机器人。

不能只把 G1 改名为“通用机器人”。模型许可证、关节定义和资产来源必须可追踪。

| # | 工作项 | 主要内容 | 验收标准 | 状态 |
|---|---|---|---|---|
| R2.1 | 选择参考机器人 | 选择记录、模型来源、许可证、目标 task | 非 Unitree，资产可分发或有明确使用边界 | ⬜ 未开始 |
| R2.2 | MJCF Inspector 扩展 | body/joint/actuator/sensor/frame/mesh 诊断 | 错误定位到名称和路径，不只报告加载失败 | ⬜ 未开始 |
| R2.3 | Robot Profile vNext | description、simulation、task、runtime binding | simulation-only Profile 不要求厂商 SDK 字段 | ⬜ 未开始 |
| R2.4 | Actuator/Sensor mapping | 类型、单位、方向、limit、gear、frequency、latency | 显式 schema 和正反例测试 | ⬜ 未开始 |
| R2.5 | MJLab config generation | Profile/MJCF 到 robot entity/config | 生成结果可复现且有 config snapshot | ⬜ 未开始 |
| R2.6 | 最小仿真运行 | load、reset、named action、observation、viewer | 无 Unitree 代码或目录依赖 | ⬜ 未开始 |
| R2.7 | G1 重新绑定设计 | G1 如何使用相同 Profile/Task 契约 | 设计不能给公共对象增加 Unitree-only 必填字段 | ⬜ 未开始 |

R2 退出条件：非 Unitree 机器人可以通过 Profile 加载到 Customized MJLab 1.6，完成 reset、动作驱动、观测生成、配置快照
和 Viewer 证据。此阶段不要求已经训练出高质量策略。

## 8. R3：真实运动策略工作流

目标：让定制工具链产生可信运动能力，而不仅是模型加载和随机动作。

| # | 工作项 | 主要内容 | 验收标准 | 状态 |
|---|---|---|---|---|
| R3.1 | 首个通用任务 | 建议 `motion.velocity.flat@1` 或明确选择的 tracking task | task 与 robot binding 分离，MDP term 可枚举 | ⬜ 未开始 |
| R3.2 | Observation/Action schema | 名称、shape、dtype、单位、frame、history、scale | 由任务配置生成并进入 Artifact | ⬜ 未开始 |
| R3.3 | TrainingRecipe | seed、runner、network、reward、randomization、资源 | 最终 resolved config 保存且可重放 | ⬜ 未开始 |
| R3.4 | GPU 训练或可信 checkpoint | 真实策略、训练日志、代码和依赖 revision | 来源可复现，不能使用 zero-policy 代替 | ⏸️ 等待外部依赖 |
| R3.5 | Trained play | 固定 Artifact 在 Viewer/录制中运行 | 产生有效运动和结构化 episode 结果 | ⬜ 未开始 |
| R3.6 | Independent evaluation | 速度误差、姿态、接触、终止、扰动等 | 阈值版本化；不以 Unitree 旧 reward 等价验收 | ⬜ 未开始 |
| R3.7 | Export | checkpoint 到 ONNX 与 deploy metadata | ONNX 输出与原策略在固定输入上按阈值一致 | ⬜ 未开始 |
| R3.8 | PolicyArtifact | checkpoint/ONNX/config/schema/hash/lineage | lint、hash、compatibility 和复现测试通过 | ⬜ 未开始 |

R3 退出条件：至少一个真实策略在非 Unitree 参考机器人或明确选择的主线机器人上完成 train/play/evaluate/export，并形成固定
PolicyArtifact。若 GPU 暂不可用，其他批次可以继续，但 R3 不得虚假关闭。

## 9. R4：Skill、平台控制面和验证

目标：把现有平台骨架真正连接到 Customized MJLab 1.6，而不是继续包装 Unitree 脚本。

| # | 工作项 | 主要内容 | 验收标准 | 状态 |
|---|---|---|---|---|
| R4.1 | MotionSkill vNext binding | robot/task/toolchain/artifact/evaluate/export | manifest 不含 vendor path 和 Unitree 环境 ID | ⬜ 未开始 |
| R4.2 | Worker Job | train/play/evaluate/export 隔离执行 | 日志、事件、取消、进程组和输出完整 | ⬜ 未开始 |
| R4.3 | Platform API | Task、Recipe、Artifact、Validation 资源 | API 只接受稳定 ID、revision 和 schema 配置 | ⬜ 未开始 |
| R4.4 | ValidationRun | fixed inputs、metrics、thresholds、evidence、status | 任一结果可追溯到 Skill/Profile/Artifact hash | ⬜ 未开始 |
| R4.5 | WebUI Train/Validate | schema 表单、raw config、真实状态和错误原因 | UI 操作产生与 CLI 等价的 Job | ⬜ 未开始 |
| R4.6 | Artifact lineage | checkpoint、ONNX、config、video、report 关联 | 不允许浮动 `main` 或无 hash 产物进入验证 | ⬜ 未开始 |
| R4.7 | Legacy isolation | 旧 G1 action 明确标记 `unitree_legacy` | legacy 结果不被显示为 MJLab 1.6 主线验收 | ⬜ 未开始 |

R4 退出条件：用户可以从安装 MotionSkill 开始，通过 API/CLI/WebUI 创建 Customized MJLab 1.6 Job，并得到固定
PolicyArtifact 和 ValidationRun。

## 10. R5：部署 Runtime 与 sim-to-sim

目标：完成从已验证 PolicyArtifact 到安全 simulation deployment 的数据面。

| # | 工作项 | 主要内容 | 验收标准 | 状态 |
|---|---|---|---|---|
| R5.1 | Runtime protocol | prepare/start/status/stop/safe、heartbeat、session token | contract test 覆盖非法转换和幂等 stop | ⬜ 未开始 |
| R5.2 | ONNX runner | observation、inference、action、frequency、shape | schema/hash/frequency 不一致时 fail closed | ⬜ 未开始 |
| R5.3 | 通用 FSM | Passive/Stand/RL/Stopping/Safe | 超时、异常和人工停止均进入 Safe | ⬜ 未开始 |
| R5.4 | Simulation driver API | RobotState/RobotCommand、时间戳和健康度 | 通用 runner 不导入厂商 SDK 类型 | ⬜ 未开始 |
| R5.5 | 自研机器人 sim-to-sim | 独立 simulator/driver、配置、轨迹和视频 | 使用 R3 同一 Artifact 完成稳定 session | ⬜ 未开始 |
| R5.6 | DeploymentPlan | target、Artifact、runtime config、gates、fallback | 输入固定且可 diff，缺 gate 不能 start | ⬜ 未开始 |
| R5.7 | DeploymentSession | PREPARING/ARMED/ACTIVE/STOPPING/SAFE/FAILED | API 重启后终态和审计可恢复 | ⬜ 未开始 |
| R5.8 | Fault injection | heartbeat loss、shape error、driver disconnect、timeout | 每项都有自动 SAFE 证据 | ⬜ 未开始 |
| R5.9 | Physical target gate | Driver/Calibration/Safety/Profile 检查 | 缺任一项时 physical target 保持不可激活 | ⬜ 未开始 |

R5 退出条件：命令行和平台能够启动一条自研机器人 simulation DeploymentPlan，进入 ACTIVE，并通过显式 stop 或故障注入
进入 SAFE；WebUI 断开不能破坏 Runtime 安全行为。

## 11. R6：多机器人证明和发布收口

目标：用不同来源机器人证明 RoboLab 是通用运动控制平台，而不是某个机器人项目的包装。

| # | 工作项 | 验收标准 | 状态 |
|---|---|---|---|
| R6.1 | 自研机器人完整闭环 | Profile -> train/play/evaluate/export -> Skill -> sim deploy 全通过 | ⬜ 未开始 |
| R6.2 | 成品机器人适配 | 到 R6 再选择合适机型；使用相同公共对象和 Runtime 接口，不预设必须是 G1 | ⬜ 未开始 |
| R6.3 | 抽象复用审查 | 第二机器人不复制完整 task、runner、FSM 或 Skill 工作流 | ⬜ 未开始 |
| R6.4 | 早期选型迁移报告 | 记录已删除依赖、可复用经验、历史来源和重新接入条件 | ⬜ 未开始 |
| R6.5 | Upstream sync rehearsal | 模拟同步较新 MJLab revision，记录冲突和回归 | ⬜ 未开始 |
| R6.6 | End-to-end acceptance | 干净环境按锁定依赖重放两类机器人代表流程 | ⬜ 未开始 |
| R6.7 | 文档和发布 | 用户指南、开发指南、已知限制、许可证和回滚同步 | ⬜ 未开始 |

R6 退出条件：至少一个自研机器人和一个成品机器人共享 RoboLab 的 Task、Artifact、Skill、Validation 和 Runtime 契约，
并分别给出可复查证据。

## 12. Unitree 退役前历史复现

以下工作只允许在 R0.6 删除前用于补充已有历史记录，不应继续投资为兼容产品：

- 获取或训练旧 G1 velocity checkpoint；
- 运行旧 MJLab 1.2 trained play；
- 保存 Unitree sim-to-sim 轨迹和视频；
- 诊断旧 checkpoint 能否迁移到 MJLab 1.6；
- 从 vendor 提取 G1 模型、参数和部署经验。

如确需执行，必须标记：

```text
toolchain: unitree_legacy_mjlab_1_2
product_default: false
blocks_robolab_mjlab_1_6: false
```

R0.6 完成后不再提供这些运行入口。历史代码可从 Git 恢复，但 active tree 不保留兼容环境。

## 13. 当前范围冻结

R0–R6 期间明确不做：

- 多租户、云端训练、远端 Worker 或 Kubernetes；
- 在线商业 Skill 市场、签名 PKI 或复杂计费；
- 内置通用聊天 Agent；
- 在安全 Runtime 验收前激活 physical motor command；
- 为追求“深度定制”而无目的重写 MuJoCo、RSL-RL 或 MJLab 已满足需求的模块；
- 把 Unitree SDK、DDS message 或 motor ID 加入公共 Robot/Task/Skill schema；
- 未记录 upstream revision 和回归证据就同步或覆盖 `mjlab/`。

## 14. 状态维护

1. 每次状态变化记录日期、代码 revision、测试命令和证据路径；
2. GPU/manual 验收与 CPU contract tests 分开记录；
3. 历史 Unitree 结果必须标记 legacy toolchain；
4. 新增横向功能前说明它服务 R0–R6 的哪个退出条件；
5. 若实现发现本计划与最高级技术决策冲突，先停止实现并修订决策，不得自行恢复 Unitree-first 路线；
6. 小模型或自动 Agent 执行本计划时，不得把“保留 Unitree 兼容”解释为“优先完成 Unitree 再开始 MJLab 1.6”。
