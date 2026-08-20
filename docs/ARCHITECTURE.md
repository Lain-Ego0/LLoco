# RoboLab 总体架构

状态：设计基线 v0.3；Q1-Q26 已完成，仓库管理与 MVP 范围已冻结。

## 1. 设计目标

RoboLab 的目标不是重新实现一个训练框架，而是在定制 MJLab 之上增加稳定的平台边界：

- 同一套界面覆盖机器人接入、训练、仿真、Skill 管理和部署。
- 训练产物从产生到实机运行全程可追踪、可校验、可回滚。
- 新增机器人主要编写 Robot Profile 和 Driver，而不是复制整套任务与部署程序。
- 新增能力主要交付 Skill Package，而不是修改平台核心代码。
- WebUI 与硬实时控制隔离，网络或浏览器故障不能破坏底层安全状态机。

首个版本只面向个人本机使用，不建设服务器、多用户、云端训练、在线商店或 Docker 运行环境。可执行 Skill 是正式能力，但必须通过 Worker 独立进程运行，不能无隔离导入 API 进程。

## 2. 分层架构

```text
┌──────────────────────────────────────────────────────────────┐
│ WebUI: Workspace / Robots / Skills / Jobs / Deploy / Audit   │
└──────────────────────────────┬───────────────────────────────┘
                               │ REST + WebSocket
┌──────────────────────────────▼───────────────────────────────┐
│ Platform API                                                  │
│ Registry · Workflow · Compatibility · Artifact · Safety Gate │
└──────────────┬───────────────────────────────┬───────────────┘
               │ jobs                          │ deploy session
┌──────────────▼──────────────┐  ┌─────────────▼───────────────┐
│ Worker / MJLab Adapter      │  │ Edge Runtime                │
│ train · play · export · sim │  │ FSM · inference · watchdog  │
└──────────────┬──────────────┘  │ SDK/DDS driver · e-stop     │
               │ artifacts       └─────────────┬───────────────┘
┌──────────────▼──────────────┐                │
│ Artifact Store + Metadata   │                ▼
│ config · logs · ONNX · hash │              Robot
└─────────────────────────────┘
```

### 2.1 Control Plane 与 Data Plane

Platform API 和 WebUI 属于控制面：创建任务、修改配置、授权部署和查看状态。Edge Runtime 属于数据面：读取传感器、执行推理、发送电机命令并处理超时。二者必须是独立进程；实机控制循环不能依赖 WebSocket 是否在线。

### 2.2 核心领域对象

| 对象 | 作用 | 稳定标识示例 |
|---|---|---|
| `RobotProfile` | 模型、关节语义、仿真参数、驱动与安全边界 | `unitree.g1.29dof@1.0.0` |
| `TaskDefinition` | MJLab 环境、算法和导出规则 | `velocity.flat@1` |
| `SkillPackage` | 可安装能力及兼容性契约 | `dance.subject2@1.2.0` |
| `Artifact` | 模型、动作、配置、日志或报告 | content SHA-256 |
| `Job` | train/play/export/evaluate 等异步任务 | UUID |
| `ValidationRun` | 某 Skill + Robot 的验证结果 | UUID |
| `DeploymentSession` | 一次 sim 或 real 部署及审计 | UUID |

所有运行记录都要固定代码 revision、Robot Profile 版本、Skill 版本、配置快照、随机种子、依赖环境和产物哈希。

## 3. 目标目录结构

在不立即移动现有 `mjlab/` 的前提下，推荐逐步演进为：

```text
RoboLab/
├── apps/
│   └── web/                  # Web 前端
├── services/
│   ├── api/                  # 平台 API 与领域服务
│   ├── worker/               # 本地异步任务执行器
│   └── edge/                 # 非实时管理面；启动/监控 C++ runtime
├── packages/
│   ├── core/                 # 领域模型、registry、兼容性和状态机
│   ├── schemas/              # manifest/profile/API JSON Schema
│   ├── mjlab_adapter/        # 将平台 Job 映射到现有 scripts
│   └── sdk/                  # Skill/Robot 扩展开发工具
├── skills/
│   ├── builtin/              # 随平台发布的 Skill
│   ├── installed/            # catalog 安装的固定版本
│   └── dev/                  # 本地 Skill 开发链接
├── integrations/
│   └── unitree/              # Unitree Driver/Profile，首个厂商适配器
├── runtime/
│   ├── core/                 # 共享 C++ FSM、推理、安全与遥测
│   └── drivers/              # unitree_sdk2 等硬件驱动插件
├── robots/                   # RoboLab Robot Profile；不放具体 Skill
├── vendor/
│   └── unitree_rl_mjlab/     # 精选上游技术来源，不含宣传媒体/冗余二进制
├── docs/
├── tests/
│   ├── contract/
│   ├── integration/
│   └── hardware/
├── var/                      # 运行数据，gitignore
│   ├── artifacts/
│   ├── skill-cache/
│   ├── skill-envs/
│   └── runs/
└── THIRD_PARTY_NOTICES.md
```

不要在初期同时进行所有内部重构和功能开发。精选 vendor 导入完成后，第一阶段可让 `mjlab_adapter` 以受控子进程调用 vendor 中现有 `scripts/train.py`、`play.py` 和必要部署程序，等接口稳定后再重构内部实现。

上游 `doc/`、演示 GIF、预编译 runtime 和策略产物不进入 active vendor。精选后的 `vendor/unitree_rl_mjlab/deploy/` 只保留必要 Unitree 部署源码和迁移参考；RoboLab 通用 runtime 放根目录 `runtime/`，平台文档放根目录 `docs/`。

## 4. 扩展边界

### 4.1 MJLab Adapter

Adapter 负责：

- 从现有 registry 发现 task，而不是在 WebUI 手写任务列表。
- 把经过 schema 校验的配置转换为 CLI 参数或配置文件。
- 启动隔离的 train/play/export 进程，采集 stdout、指标和退出码。
- 将 checkpoint、ONNX、deploy.yaml、视频和日志登记为 Artifact。
- 不把训练进程直接嵌入 API 进程，避免 GPU 上下文和崩溃相互影响。

### 4.2 Skill Manager

Skill Manager 只消费明确的 manifest，但正式支持三种运行形态：

1. `MotionSkill`：策略、动作和训练/部署参数；
2. `PlatformSkill`：Python、CLI 或 C++ 平台功能；
3. `AgentSkill`：Agent 操作说明、工作流和平台工具白名单。

可执行 Skill 由 Worker 以独立进程启动，使用统一 Job 输入/事件/结果协议。安装阶段不自动执行仓库里的 `setup.sh`；Conda 环境准备、权限确认和首次 contract test 是显式步骤。WebUI、CLI 与 Agent 共享同一个 action registry。

AgentSkill 以根目录 `SKILL.md` 为公共核心，通过 adapter 导出给 Codex、Claude、DeepSeek 等外部开发 Agent。首版不强制内置对话模型；平台内 Agent 后续也必须复用同一个 action registry。

### 4.3 Robot Adapter

机器人适配按 simulation-first 拆为稳定接口：

1. `RobotDescription`：MJCF、mesh、frames、关节和传感器语义。
2. `SimulationConfig`：执行器、碰撞、初始姿态、控制周期和域随机化边界。
3. `MotorBus/RuntimeDriver`：状态读取、命令写入、SDK/DDS/自制总线映射与连接健康度。
4. `SensorAdapter/StateEstimator`：统一时间戳传感器与机器人状态。
5. `CalibrationProvider`：零位、方向、外参和可追踪标定报告。
6. `SafetyProfile/Controller`：限位、增益上限、超时、降级状态和实机检查清单。

只有前两项也可以形成 simulation-only Profile。训练任务和 Skill 只能依赖稳定 capability，不能直接依赖某个厂商电机 ID。

## 5. Job 与部署状态机

通用 Job：

```text
CREATED -> VALIDATING -> QUEUED -> RUNNING -> SUCCEEDED
                     \-> REJECTED        \-> FAILED/CANCELLED
```

实机部署必须使用更严格的状态机：

```text
DRAFT -> COMPATIBLE -> OFFLINE_VALIDATED -> SIM_VALIDATED
      -> OPERATOR_ARMED -> EDGE_CONFIRMED -> ACTIVE
      -> STOPPING -> SAFE
```

任一阶段检测到遥测超时、姿态越界、命令超时、驱动断连或人工急停，都由 Edge Runtime 直接进入 `SAFE`，不等待 WebUI 响应。

## 6. 存储与通信建议

- 平台只监听本机 loopback；MVP 不提供远程服务器入口和多人认证。
- MVP 元数据使用 SQLite；产物使用本地 content-addressed 目录。
- API 使用 REST；任务日志与低频遥测使用 WebSocket/SSE。
- 高频完整遥测写入本地记录文件，UI 只接收降采样数据。
- Worker 初期使用本地进程队列；不要为单机 MVP 过早引入 Kubernetes。
- Edge 与 API 的协议应有版本号、心跳和幂等 session token。
- 机器人网卡、设备地址和本机配置不能写入可发布 Skill manifest。

## 7. 技术栈基线

- Backend：Python + FastAPI + Pydantic；与 MJLab/Python 生态一致。
- Frontend：React + TypeScript；组件库可在视觉原型后确定。
- Worker：本地 subprocess + 轻量队列，同时承载 MJLab Job 和可执行 Skill。
- Environment：Conda；平台主环境 + 可选的按 Skill 隔离环境，暂不使用 Docker。
- Metadata：本地 SQLite；当前不为 PostgreSQL/多用户增加额外复杂度。
- Edge Runtime：保留 C++、ONNX Runtime、unitree_sdk2/CycloneDDS，并把现有重复机器人目录抽成共享 runtime + driver/profile。

## 8. 架构约束

- UI 展示的兼容不等于可部署；只有验证记录能解锁实机门禁。
- Skill 版本、Robot Profile 版本和 Artifact 哈希一旦用于部署就不可原地覆盖。
- 所有关节映射必须按名称和显式索引校验，禁止仅靠数组长度推断。
- 平台后端不能以 root 运行训练或第三方 Skill 代码。
- 可执行 Skill 只能访问 manifest 已声明并由用户确认的文件、网络、子进程和机器人能力。
- `main`、未固定的 URL 和没有哈希的模型不能成为可复现部署输入。
