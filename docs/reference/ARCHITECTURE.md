# RoboLab 总体架构

状态：设计基线 v0.5，2026-08-21 生效。本文描述长期架构边界；当前批次和实施状态以
[`DEVELOPMENT_PLAN.md`](../project/DEVELOPMENT_PLAN.md) 为唯一权威。

## 1. 产品与技术目标

RoboLab 是基于深度定制 MJLab 1.6 工具链的一站式运动控制开发与部署平台，服务商业成品机器人、实验室自研机器人、
爱好者机器人和比赛机器人。完整技术决策见
[`MJLAB_1_6_TECHNICAL_DIRECTION.md`](../project/MJLAB_1_6_TECHNICAL_DIRECTION.md)。

项目不是重新实现物理引擎或强化学习算法，而是在 MJLab、MuJoCo、MuJoCo-Warp、Warp 和 RSL-RL 之上建立并维护：

- RoboLab 定制的机器人、任务、训练、回放、评测和导出工具链；
- Robot Profile、Skill、Policy Artifact 和 DeploymentPlan 的统一契约；
- 面向不同机构、执行器、传感器和通信方式的机器人适配流程；
- 与 WebUI 隔离的部署 Runtime、安全状态机和 Driver；
- 从机器人接入到策略部署的可追踪、可复现、可回滚工作流。

首版是个人本机工程工作台，不建设服务器、多用户、云端训练、在线商店或 Docker 运行环境。

## 2. 分层架构

```text
┌──────────────────────────────────────────────────────────────┐
│ WebUI / CLI / Agent                                          │
│ Robots · Skills · Train · Jobs · Validate · Deploy · Audit   │
└──────────────────────────────┬───────────────────────────────┘
                               │ REST + WebSocket
┌──────────────────────────────▼───────────────────────────────┐
│ RoboLab Platform Control Plane                               │
│ Registry · Workflow · Artifact · Compatibility · Safety Gate │
└──────────────┬────────────────────────────────┬──────────────┘
               │ motion jobs                    │ deployment plan/session
┌──────────────▼─────────────────────────┐  ┌───▼────────────────────────┐
│ RoboLab Customized MJLab 1.6 Toolchain │  │ Edge Runtime               │
│ Robot · Task · MDP · Train · Play      │  │ Policy · FSM · Watchdog    │
│ Evaluate · Export · Metrics · Viewer   │  │ Telemetry · Stop · Safe    │
└──────────────┬─────────────────────────┘  └───┬────────────────────────┘
               │                                │ driver interface
┌──────────────▼────────────────────────────────▼───────────────┐
│ Robot Adaptation                                             │
│ Robot Profile · MJCF · Actuator · Sensor · Calibration       │
│ Custom Robot Driver · Unitree Driver · Other Drivers          │
└───────────────────────────────────────────────────────────────┘
```

Skill 和 Artifact 是跨层对象：Skill 描述可安装能力和权限，Artifact 固定策略、配置、schema、hash 和来源。它们不直接
替代 MJLab task 或 Runtime driver。

## 3. 核心组件

| 组件 | 职责 | 当前状态 |
|---|---|---|
| Customized MJLab 1.6 | 机器人、任务、MDP、训练、回放、评测、导出和仿真指标 | 上游源码已存在；RoboLab 定制尚未开始 |
| Platform Core | Schema、兼容性、Action、Job、Artifact 和部署门禁 | B1–B6 骨架已实现 |
| Skill System | MotionSkill、PlatformSkill、AgentSkill 的安装、权限和调用 | 当前完成度最高 |
| Robot Adaptation | MJCF/Profile、执行器、传感器、控制周期、Driver 和标定 | G1 simulation-only 样板已存在；通用接入待实现 |
| Edge Runtime | ONNX 推理、FSM、watchdog、遥测和 stop/safe | 仅有接口边界 |
| Unitree Legacy Reference | 历史 G1 旧任务、模型、部署和 sim-to-sim 来源 | R0.6 从 active tree 删除，Git 历史保留 |

## 4. 仓库所有权

```text
RoboLab/
├── apps/
│   ├── cli/                         # 本地 CLI
│   └── web/                         # React WebUI
├── services/
│   ├── api/                         # FastAPI 控制面
│   └── worker/                      # 隔离 Job 执行
├── packages/
│   ├── schemas/                     # 公共版本化 schema
│   ├── core/                        # Registry、compatibility、artifact、action
│   └── mjlab_tasks/                 # 目标：RoboLab MJLab task/robot 扩展
├── mjlab/                           # MJLab 1.6 下游定制基座（一级目录）
├── robots/                          # Robot Profile 与模型 binding
├── skills/                          # builtin/installed/dev Skill 工作区
├── runtime/                         # 厂商和仿真器无关的部署数据面
├── integrations/
│   └── unitree/                     # 一个具体厂商 Driver/SDK adapter
├── tests/                           # contract/integration/hardware 测试
├── docs/
└── var/                             # 本地运行数据，gitignore
```

`mjlab/` 是仓库一级目录，但仍与 RoboLab Platform 保持独立的 pyproject、源码、测试、上游记录和修改账本边界。
当同步和发布边界稳定后，再评估是否拆成独立 `RoboLab-MJLab` 仓库。Unitree 旧栈退役和迁移细节见
[`UNITREE_RETIREMENT_AND_MJLAB_RELOCATION.md`](../project/UNITREE_RETIREMENT_AND_MJLAB_RELOCATION.md)。

## 5. Customized MJLab 工具链

RoboLab 对 MJLab 1.6 的定制边界包括：

1. **Robot layer**：从 Robot Profile/MJCF 构造 robot config，校验 joint、actuator、sensor 和 frame；
2. **Task layer**：稳定 task ID、配置 schema、MDP term、能力要求和 robot binding；
3. **Execution layer**：统一 train/play/evaluate/export 命令和结构化 Job 输出；
4. **Artifact layer**：checkpoint/ONNX 与 observation、action、control、训练来源和 hash 绑定；
5. **Validation layer**：版本化指标、场景、阈值、视频和迁移报告；
6. **Deployment binding**：把仿真策略输入输出映射为 Runtime 可校验的部署配置。

工具链可以修改 MJLab 核心源码，也可以使用 `packages/mjlab_tasks/` 等 RoboLab 扩展包。判断修改位置的原则是：

- 通用于 MJLab 运动控制任务、且需要与其 registry/config 生命周期集成的能力，可以进入定制 MJLab；
- 只属于 RoboLab 平台控制面的能力留在 `packages/core`、API 或 Worker；
- 只属于具体机器人通信的能力留在 `integrations/` 和 Runtime driver；
- 只属于某个可安装能力的内容留在 Skill。

## 6. Robot Adaptation

机器人适配不假设机器人来自某个厂商，按 simulation-first 拆分为：

1. `RobotDescription`：MJCF、mesh、joint、frame 和 sensor 语义；
2. `SimulationConfig`：actuator、碰撞、初始姿态、控制周期和随机化边界；
3. `TaskBinding`：机器人如何实例化到 velocity、tracking 等通用任务；
4. `RuntimeBinding`：observation/action、关节顺序、缩放、单位和频率；
5. `MotorBus/RuntimeDriver`：状态读取、命令写入和连接健康度；
6. `CalibrationProvider`：零位、方向、外参和报告；
7. `SafetyProfile`：限位、增益、超时和安全回退。

只有前三项即可形成 simulation-only Profile。自研机器人是主线能力，不能要求使用 Unitree motor ID、DDS topic 或工程目录。

## 7. Skill、Job 与 Artifact

Skill 支持三种形态：

- `MotionSkill`：任务、策略、动作和训练/部署参数；
- `PlatformSkill`：Python、CLI 或 C++ 平台功能；
- `AgentSkill`：Agent 操作说明、工作流和工具白名单。

可执行 Skill 和 MJLab Job 都由 Worker 独立进程运行，不导入 API 进程。每个 motion Job 必须保存：

- toolchain ID、MJLab upstream revision 和 RoboLab revision；
- Robot Profile、TaskDefinition 和最终配置 snapshot；
- argv、cwd、允许的环境变量和输出目录；
- checkpoint、ONNX、日志、指标、视频和 hash；
- observation/action/deployment schema。

## 8. 部署 Runtime 与安全边界

WebUI 不承担硬实时关节控制。Edge Runtime 必须能在 WebUI/API 断开时独立进入安全状态。

通用部署状态机：

```text
DRAFT -> COMPATIBLE -> OFFLINE_VALIDATED -> SIM_VALIDATED
      -> OPERATOR_ARMED -> EDGE_CONFIRMED -> ACTIVE
      -> STOPPING -> SAFE
```

任一阶段发生遥测超时、姿态越界、命令超时、Driver 断连或人工急停，都由 Runtime 直接进入 `SAFE`。厂商 SDK 类型不得
泄漏到通用 policy runner、FSM 或 DeploymentPlan。

## 9. Unitree 退役边界

- R0.6 删除 `vendor/unitree_rl_mjlab/`、`packages/mjlab_adapter/`、G1 Profile 和 Unitree Integration；
- CLI/API 同步删除 `vendor_root`、旧 discovery/play 和 vendor health check；
- 当前发布物不再提供 Unitree legacy checkpoint 或 sim-to-sim 兼容承诺；
- 未来重新支持 G1 时，必须通过和自研机器人相同的 Robot Profile、Task、Artifact 和 Runtime 契约接入；
- 历史来源、许可证事实和 commit 继续保留在 Git 与历史文档中。

## 10. 架构约束

- 新增机器人不得复制整套 task、runner 或 Runtime；
- 公共 API 不接受 vendor root、vendor script path 或厂商 task ID；
- 任务、Profile、Skill 和 Artifact 版本一旦用于验证或部署，不可原地覆盖；
- 所有关节映射按名称和显式索引校验，禁止仅靠数组长度推断；
- 未经仿真验证的 Artifact 默认不能激活 physical target；
- 缺少 Driver、Calibration 或 SafetyProfile 时必须 fail closed；
- GPU 资源只能阻塞对应训练/评测验收，不得阻塞 CPU 可完成的契约、工具链和 Runtime 工作；
- MJLab 上游同步必须有依赖锁定、回归报告和回滚 revision。
