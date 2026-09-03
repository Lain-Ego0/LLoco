# Lain's LocoLab 架构设计指南

> 状态：已固化，待实施  
> 版本：0.1  
> 更新时间：2026-09-03

本文档定义 Lain's LocoLab（工程简称 **LainLoco**）的目标架构、模块边界、依赖方向和实施约束。它是后续重构的设计依据，不表示文中目录已经全部落地。实际进度见 [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md)。

## 1. 目标与非目标

### 1.1 目标

1. 以机器人作为业务能力的第一归属，使开发者能够从一个目录看到某个机器人支持的全部技能。
2. 将“机器人做什么”和“使用什么算法训练”分离，避免把 DreamWaQ、CTS 等算法伪装成环境任务。
3. 保留 mjlab 的 Manager-based 环境、MuJoCo Warp、任务注册和 RSL-RL 接口。
4. 让通用 MDP、算法和运行时能力保持单一来源，避免为每个任务复制完整环境。
5. 用显式 PolicyContract 连接训练、ONNX、sim-to-sim 和实机部署。
6. 在重构期间保持现有 `Mjlab-*` Task ID、checkpoint 和策略输入输出兼容。

### 1.2 非目标

- 不重写 MuJoCo、MuJoCo Warp 或 mjlab 的通用仿真生命周期。
- 不把 Isaac Gym 的 `LeggedRobot` 类原样复制到新项目。
- 不在架构重构阶段同时改变奖励权重、观测顺序或训练数学语义。
- 不以目录拆分代替行为级训练验收。
- 当前不把真实 Unitree Go2 硬件闭环作为迁移完成的必要条件。

## 2. 核心架构决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 项目所有权 | 机器人优先 | Go2 的技能、控制契约和部署能力应集中可见 |
| 底层复用 | 能力优先 | 通用 MDP 和算法不能绑定到单个机器人 |
| 环境与算法 | 分离 | Task 描述目标，TrainingProfile 描述学习方法 |
| 配置生成 | 显式组合 | 避免 Rough → Flat → Skill 的深层修改链 |
| 注册方式 | 显式 Catalog | 避免依靠大型 `__init__.py` 和裸元组维护任务 |
| 部署产物 | Policy Bundle | 模型必须携带可校验的机器人和输入输出契约 |
| 上游关系 | 扩展包依赖 mjlab | 降低同步上游时的冲突和项目身份混淆 |

该架构可以概括为：

```text
机器人优先的领域组织
        +
可复用的通用内核
        +
显式的实验组合
```

## 3. 系统边界

```text
┌─────────────────────────────────────────────┐
│                 LainLoco CLI                │
│       catalog / train / play / export       │
└──────────────────────┬──────────────────────┘
                       │ 组合 ExperimentSpec
┌──────────────────────▼──────────────────────┐
│             Robot-owned domains             │
│      Go2 robot / tasks / training / deploy  │
└──────────────┬───────────────────┬──────────┘
               │                   │
┌──────────────▼──────────┐ ┌──────▼──────────┐
│ Shared MDP & Learning   │ │ Policy Runtime  │
│ rewards / algorithms   │ │ ONNX / bundle   │
└──────────────┬──────────┘ └──────┬──────────┘
               │                   │
┌──────────────▼───────────────────▼──────────┐
│                    mjlab                    │
│ simulation / scene / sensors / managers    │
└─────────────────────────────────────────────┘
```

## 4. 目标仓库结构

```text
Lains-LocoLab/
├── pyproject.toml                 # uv workspace；仓库级工具配置
├── README.md
├── ARCHITECTURE.md
├── PROJECT_PROGRESS.md
├── GO2_MIGRATION_PLAN.md
│
├── mjlab/                         # 上游框架或最小化 fork
│
├── packages/
│   └── lainloco/
│       ├── pyproject.toml
│       └── src/lainloco/
│           ├── core/
│           ├── mdp/
│           ├── learning/
│           ├── runtime/
│           ├── robots/
│           └── bootstrap.py
│
├── deploy/
│   └── unitree/go2/
│       ├── sim2sim/
│       ├── hardware/
│       └── fsm/
│
├── tests/
│   ├── contracts/
│   ├── integration/
│   └── training/
│
└── runs/                          # 本地生成，不纳入源码
```

仓库改名和目录移动可以晚于 Python 包边界落地。任何阶段都不应通过一次性重命名破坏当前可运行基线。

## 5. LainLoco 包结构

```text
lainloco/
├── core/
│   ├── robot_spec.py
│   ├── task_spec.py
│   ├── training_spec.py
│   ├── experiment_spec.py
│   ├── policy_contract.py
│   ├── catalog.py
│   └── compose.py
│
├── mdp/                           # 与具体机器人无关
│   ├── observations.py
│   ├── rewards.py
│   ├── commands.py
│   ├── events.py
│   └── terminations.py
│
├── learning/                      # 与具体机器人无关
│   ├── algorithms/
│   │   ├── amp/
│   │   ├── cts/
│   │   ├── dreamwaq/
│   │   └── distillation/
│   ├── models/
│   ├── storage/
│   ├── runners/
│   └── exporters/
│
├── runtime/
│   ├── policy_bundle.py
│   ├── onnx_runtime.py
│   └── history_buffer.py
│
└── robots/
    └── unitree/
        └── go2/
            ├── robot.py
            ├── contract.py
            ├── assets/
            ├── mdp/
            ├── tasks/
            ├── training/
            └── deploy/
```

## 6. Go2 领域结构

```text
robots/unitree/go2/
├── robot.py                       # Go2 RobotSpec
├── contract.py                    # 关节、动作和部署契约
│
├── mdp/                           # Go2 专用、可被多个 Go2 技能复用
│   ├── actions.py
│   ├── observations.py
│   ├── rewards.py
│   ├── commands.py
│   └── events.py
│
├── tasks/
│   ├── locomotion/
│   │   ├── velocity.py
│   │   └── trot.py
│   ├── aerial/
│   │   ├── jump.py
│   │   ├── spring_jump.py
│   │   └── backflip.py
│   └── balance/
│       ├── handstand.py
│       └── leggedstand.py
│
├── training/                      # Go2 对通用算法的参数绑定
│   ├── ppo.py
│   ├── dreamwaq.py
│   ├── cts.py
│   ├── amp.py
│   └── teacher_student.py
│
└── deploy/
    ├── joint_mapping.py
    ├── safety.py
    └── exporter.py
```

默认一个技能使用一个文件。只有当单个技能确实包含大量独有观测、奖励或状态机时，才将其升级为子包。不要为了形式整齐制造大量只有几行的文件。

## 7. 核心领域对象

### 7.1 RobotSpec

RobotSpec 描述机器人本体和不会随任务改变的控制事实：

```text
robot_id
asset_factory
joint_names
joint_order
base_body
foot_sites
collision_geoms
default_pose
action_scale
physics_dt
control_dt
hardware_joint_mapping
```

Go2 的 12 关节顺序、足端名称和默认姿态只能在 RobotSpec 或其明确引用的资产模块中拥有一个权威来源。

### 7.2 TaskSpec

TaskSpec 描述机器人要完成什么：

```text
task_id
family
command_profile
terrain_profile
observation_profile
reward_profile
termination_profile
randomization_profile
episode_length
```

TaskSpec 不保存 PPO learning rate、模型类或 optimizer 参数。

### 7.3 TrainingSpec

TrainingSpec 描述如何学习：

```text
profile_id
algorithm
actor_model
critic_model
storage
runner
optimizer
auxiliary_losses
required_observation_groups
exporter
```

AMP 应表达为辅助目标或训练能力；CTS、DreamWaQ 和 Teacher-Student 通过 TrainingSpec 绑定模型、storage 和 runner，而不是伪装成新的运动技能。

### 7.4 PolicyContract

PolicyContract 是训练和部署之间的版本化接口：

```text
contract_version
robot_id
task_id
joint_order
action_dim
action_scale
observation_fields
observation_order
history_length
history_order
history_reset
normalization
recurrent_state
control_dt
```

### 7.5 ExperimentSpec

ExperimentSpec 是最终可启动单元：

```text
ExperimentSpec
├── RobotSpec
├── TaskSpec
├── TrainingSpec
└── PolicyContract
```

它可以被固化为兼容 Task ID，也可以由 LainLoco CLI 在运行时组合。

## 8. 任务与训练方案重新分层

### 8.1 机器人技能

- Velocity Locomotion
- Trot
- Jump
- Spring Jump
- Backflip
- Handstand
- Leggedstand

### 8.2 训练方案

- PPO
- DreamWaQ
- AMP + DreamWaQ
- CTS
- AMP + CTS
- Teacher PPO
- AMP + Teacher PPO
- Student Distillation

例如，旧入口：

```text
Mjlab-DreamWaQ-Rough-Unitree-Go2
```

在新模型中解释为：

```text
robot   = go2
task    = locomotion
terrain = rough
profile = dreamwaq
```

Teacher-Student 的 student 是训练工作流阶段，不应被解释成一种新的环境行为。

## 9. 配置构造规则

禁止继续扩展以下模式：

```text
rough_cfg()
  → 删除地形得到 flat_cfg()
    → 修改奖励得到 jump_cfg()
      → 再修改观测得到某算法 cfg()
```

目标模式是显式组合：

```python
compose_env(
  robot=GO2,
  terrain=FLAT_TERRAIN,
  task=BACKFLIP,
  observations=BACKFLIP_OBSERVATIONS,
  randomization=BACKFLIP_RANDOMIZATION,
)
```

约束如下：

1. Flat 和 Rough 从共同基础构造，彼此不继承。
2. 配置继承或补丁深度不超过两层。
3. Factory 每次返回新对象，不共享可变配置。
4. Task 文件只设置与该技能相关的差异。
5. 训练方案通过明确的 observation profile 请求额外输入，不直接修改任意环境内部字段。
6. 行数只作为代码异味提示，不作为机械拆分指标；一个模块只应有一个主要变化原因。

## 10. 依赖方向

```text
mjlab
  ↑
lainloco.core / lainloco.mdp / lainloco.learning
  ↑
lainloco.robots.unitree.go2
  ↑
go2.tasks / go2.training
  ↑
catalog / CLI
```

必须遵守：

- `learning` 不得导入 Go2。
- 通用 `mdp` 不得出现 `go2_*` 名称或 Go2 实体名称。
- Go2 MDP 可以依赖通用 MDP。
- Task 可以依赖 RobotSpec，RobotSpec 不得依赖具体 Task。
- Deployment 只能依赖 RobotSpec、PolicyContract 和 runtime。
- Deployment 不得导入 PPO、训练 Runner 或训练环境。
- mjlab 不得反向依赖 LainLoco。
- 只有至少两个机器人真正共享同一语义时，才将机器人实现提升到公共层。

## 11. Catalog、注册与命令行

任务和训练方案分别维护类型化 Catalog：

```text
TASK_CATALOG
├── go2/velocity-flat
├── go2/velocity-rough
├── go2/trot
├── go2/jump
├── go2/spring-jump
├── go2/backflip
├── go2/handstand
└── go2/leggedstand

TRAINING_PROFILES
├── ppo
├── dreamwaq
├── amp-dreamwaq
├── cts
├── amp-cts
├── ts-teacher
├── amp-ts-teacher
└── ts-student
```

禁止在注册文件中继续维护没有字段名的长元组。mjlab 注册表作为兼容适配层，新入口由 Catalog 组合。

目标命令：

```bash
lainloco robots list
lainloco tasks list --robot go2
lainloco profiles list --robot go2
lainloco train go2/backflip --terrain flat --profile ppo
lainloco train go2/locomotion --terrain rough --profile dreamwaq
lainloco distill go2/locomotion --profile ts-student --teacher model.pt
lainloco validate contracts --all
lainloco export /path/to/model.pt
```

新 Task ID 不包含算法：

```text
LainLoco-Go2-Velocity-Flat-v0
LainLoco-Go2-Locomotion-Rough-v0
LainLoco-Go2-Trot-Flat-v0
LainLoco-Go2-Backflip-Flat-v0
```

现有 `Mjlab-*` ID 在至少一个正式兼容周期内保留。

## 12. 部署架构

每次导出生成完整 Policy Bundle：

```text
go2-backflip-ppo/
├── policy.onnx
├── contract.json
├── normalization.npz
├── robot.yaml
├── task.yaml
└── manifest.json
```

部署加载器必须拒绝以下不匹配：

- Robot ID 不匹配；
- 关节名称或顺序不匹配；
- action dimension、scale 或 control dt 不匹配；
- observation 字段、顺序、history 或 normalization 不匹配；
- recurrent hidden/cell 形状不匹配；
- 不支持的 contract version。

sim-to-sim 和真实硬件控制程序共享同一个 Policy Bundle 读取器和 Go2 joint mapping。真实硬件层额外负责 FSM、安全限幅和通信，不重新实现策略预处理。

## 13. 测试与验收分层

```text
tests/
├── contracts/                     # CPU；每次提交
├── integration/                   # reset/step/ONNX；按需或 CI
└── training/                      # GPU 短训、容量和收敛验收
```

### Contract

- Catalog ID 唯一；
- factory 无可变对象泄漏；
- Go2 action dimension 为 12；
- actor、critic、history 和 auxiliary group 维度正确；
- Teacher 与 Student 契约兼容；
- 旧 ID 与新 ExperimentSpec 生成等价配置；
- ONNX 元数据与 PolicyContract 一致。

### Integration

- 所有任务可以构建、reset 和 step；
- checkpoint 可以保存和重新加载；
- 普通、多输入和 recurrent ONNX 均可重新加载前向；
- Policy Bundle 可以被 sim-to-sim runtime 消费。

### Training

- 最小 runner update：证明梯度和 optimizer 链路可执行；
- 容量测试：证明目标环境数和 rollout 长度可运行；
- 固定预算训练：比较 reward、episode length 和任务指标；
- 行为验收：视频、成功率、速度误差、落地稳定性等。

不得用 Contract 或短训通过替代收敛结论。

## 14. 当前代码迁移映射

| 当前路径 | 目标归属 |
|---|---|
| `tasks/velocity/config/go2/env_cfgs.py` | `robots/unitree/go2/tasks/*` |
| `tasks/velocity/config/go2/rl_cfg.py` | `robots/unitree/go2/training/*` |
| `tasks/velocity/mdp/go2_actions.py` | `robots/unitree/go2/mdp/actions.py` |
| `tasks/velocity/mdp/go2_commands.py` | `robots/unitree/go2/mdp/commands.py` |
| `tasks/velocity/mdp/go2_events.py` | `robots/unitree/go2/mdp/events.py` |
| `observations.py` 中 Go2 函数 | 对应 Go2 task 或 `go2/mdp/observations.py` |
| `rewards.py` 中 Go2 函数 | 对应 Go2 task 或 `go2/mdp/rewards.py` |
| `rl/go2_algorithms/algorithms.py` | `learning/algorithms/*` |
| `rl/go2_algorithms/models.py` | `learning/models/*` 与各算法模块 |
| `rl/go2_algorithms/storage.py` | `learning/storage/*` |
| `rl/go2_algorithms/deployment.py` | `runtime/*` 与 `go2/deploy/*` |
| Go2 注册 `__init__.py` | `core/catalog.py` 与兼容适配器 |
| Go2 资产 | `robots/unitree/go2/assets/` 或稳定资产包 |

## 15. 实施阶段

### A0：冻结现有行为契约

- 保存当前 16 个 Go2 注册入口。
- 固化 14 个迁移任务的观测维度。
- 固化 12 关节顺序、动作缩放和 ONNX v1 契约。

### A1：建立包边界

- 新建独立 `lainloco` 包。
- 通过 mjlab task entry point 注册扩展。
- 将 Go2 专用 CLI 从 mjlab 脚本迁出。

### A2：建立 RobotSpec 与 Catalog

- 建立 Go2 RobotSpec、TaskSpec、TrainingSpec 和 ExperimentSpec。
- 建立新 ID 和旧 ID 映射。
- 生成契约测试矩阵。

### A3：按技能迁移环境

- 先迁移 Velocity/Trot；
- 再迁移 Aerial；
- 再迁移 Balance；
- 最后删除旧 `env_cfgs.py` 的实现，仅保留临时兼容导出。

### A4：迁移训练能力

- 将算法实现从 Go2 命名空间提取到 `learning`；
- 在 `go2/training` 中保留参数和 observation binding；
- 将 student 改为显式 distillation workflow。

### A5：部署闭环

- 生成 Policy Bundle；
- 接通 sim-to-sim；
- 验证 recurrent state、history 和控制频率；
- 真实硬件接入保持独立验收。

### A6：清理与公开发布

- 移除过期兼容路径；
- 完成仓库和包改名；
- 完成许可证、第三方资产声明、CI 和贡献指南。

## 16. 重构纪律

1. 每次结构迁移只改变一个边界，不同时调整训练数值。
2. 每个阶段开始前记录基线，结束后执行对应 Contract 和 smoke test。
3. 优先使用兼容导入和旧 ID alias，不做一次性全局破坏性重命名。
4. 不复制 14 套完整环境。
5. 不把所有 Go2 技能继续归类为 velocity。
6. 不把算法名称永久编码进 Task ID。
7. 不在 `mjlab` 中新增 Go2 专用基础设施；真正通用的能力应有跨机器人测试。
8. 所有完成状态必须在 [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) 中附带证据。

## 17. 架构决策记录

后续重大变化应在 `docs/architecture/decisions/` 中添加 ADR：

```text
0001-robot-first-ownership.md
0002-separate-task-and-training-profile.md
0003-mjlab-extension-boundary.md
0004-versioned-policy-contract.md
0005-legacy-task-id-compatibility.md
```

ADR 必须记录背景、决定、替代方案、后果和迁移影响。不得只记录最终目录树。
