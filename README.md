# Lain's LocoLab

> **LainLoco** — 面向腿式机器人的可组合运动学习实验场，从 MuJoCo 仿真训练到策略部署。

Lain's LocoLab 是一个建立在 [mjlab](https://github.com/mujocolab/mjlab) 之上的机器人运动学习项目。Unitree Go2 的多种运动技能、自定义强化学习算法和部署契约现已迁入独立扩展包；后续工作重点是策略质量、公开发布准备和新增机器人能力。

仓库当前仍沿用 `RoboLab` 目录名；`lainloco` Python 包和 CLI 已建立，原有
`Mjlab-*` 任务 ID 作为兼容别名继续可用。项目后续改名时计划使用：

- 展示名称：**Lain's LocoLab**
- 工程简称：**LainLoco**
- Python 包与 CLI：`lainloco`
- 视觉简称：`LLoco` 或 `L³`

## 文档导航

- [架构设计指南](ARCHITECTURE.md)：目标目录、领域对象、依赖规则、注册方式和实施顺序。
- [项目进度与验收表](PROJECT_PROGRESS.md)：迁移完整性、策略成熟度和架构实施度。
- [Go2 全量迁移计划](GO2_MIGRATION_PLAN.md)：源项目语义、任务基线和详细迁移记录。
- [架构决策记录](docs/architecture/decisions/)：已接受的关键边界及其迁移影响。
- [mjlab 原始说明](vendor/mjlab/README.md)：本地后端 fork 的安装、训练和开发文档。

## 当前状态

Go2 代码迁移与 A1–A5 架构重构已完成；A6 公开发布仍等待仓库名和项目自身许可证的选择。

- 已接入 Unitree Go2 MJCF、12 关节动作接口、接触与地形传感器。
- 已注册原项目的 14 个 Go2 训练入口。
- 已接入 PPO、AMP、CTS、DreamWaQ 和 Teacher-Student 相关训练流程。
- 已提供资产检查、观测契约检查和有限步无界面 smoke test。
- 已产生短训 checkpoint 和 ONNX 导出，用于验证训练及导出链路。
- 已建立独立 `lainloco` 扩展包、mjlab task entry point、Robot/Task/Training/
  Experiment Spec 和类型化 Catalog。
- 已建立完整 Policy Bundle、严格契约加载器、有状态 ONNX runtime、独立
  sim-to-sim 控制循环与 Go2 Passive/Stand/Policy 安全状态机。

这些结果说明任务能够完成构建、reset、step、短训和策略导出，但**不代表全部技能已经完成收敛或通过实机性能验收**。

详细迁移范围和观测维度见 [GO2_MIGRATION_PLAN.md](GO2_MIGRATION_PLAN.md)，当前验收状态见 [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md)。

## Go2 任务

### 基础速度任务

- `Mjlab-Velocity-Flat-Unitree-Go2`
- `Mjlab-Velocity-Rough-Unitree-Go2`

### 技能任务

- `Mjlab-Trot-Flat-Unitree-Go2`
- `Mjlab-Jump-Flat-Unitree-Go2`
- `Mjlab-Spring-Jump-Flat-Unitree-Go2`
- `Mjlab-Backflip-Flat-Unitree-Go2`
- `Mjlab-Handstand-Flat-Unitree-Go2`
- `Mjlab-Leggedstand-Flat-Unitree-Go2`

### 自定义训练方案

- `Mjlab-DreamWaQ-Rough-Unitree-Go2`
- `Mjlab-AMP-DreamWaQ-Rough-Unitree-Go2`
- `Mjlab-CTS-Rough-Unitree-Go2`
- `Mjlab-AMP-CTS-Rough-Unitree-Go2`
- `Mjlab-AMP-TS-Rough-Unitree-Go2`
- `Mjlab-AMP-TS-Student-Rough-Unitree-Go2`
- `Mjlab-TS-Rough-Unitree-Go2`
- `Mjlab-TS-Student-Rough-Unitree-Go2`

`Mjlab-*` Task ID 为迁移兼容接口；新的 `LainLoco-*` ID 只描述技能，训练算法通过
TrainingProfile/ExperimentSpec 组合。

## 快速开始

进入 workspace 根目录：

```bash
cd /home/lxy/RoboLab
```

安装开发环境。NVIDIA GPU 环境使用：

```bash
uv sync --package lainloco --extra cu128
```

仅进行 CPU 结构检查时可以使用：

```bash
uv sync --package lainloco --extra cpu
```

列出已经注册的环境：

```bash
uv run --package lainloco --extra cpu lainloco envs
```

查看机器人、技能和训练方案 Catalog：

```bash
uv run --package lainloco --extra cpu lainloco robots list
uv run --package lainloco --extra cpu lainloco tasks list
uv run --package lainloco --extra cpu lainloco profiles list
```

显式组合 task/profile 训练；先用 dry-run 检查解析结果：

```bash
uv run --package lainloco --extra cpu lainloco train go2/backflip \
  --profile ppo --iterations 1000 --num-envs 4096 --gpu-ids cpu --dry-run
uv run --package lainloco --extra cu128 lainloco train go2/backflip \
  --profile ppo --iterations 1000 --num-envs 4096 --gpu-ids 0
```

通过同一组合回放随机策略或本地 checkpoint：

```bash
uv run --package lainloco --extra cpu lainloco play go2/trot \
  --profile ppo --agent random --dry-run
uv run --package lainloco --extra cu128 lainloco play go2/trot \
  --profile ppo --agent trained --checkpoint /path/to/model.pt
```

显式启动 teacher→student 蒸馏：

```bash
uv run --package lainloco --extra cu128 lainloco distill /path/to/teacher.pt \
  --task go2/velocity-rough --profile ts-student --num-envs 1024
```

先检查解析结果而不启动训练，可追加 `--dry-run`。

```bash
uv run --package lainloco --extra cpu lainloco distill /path/to/teacher.pt \
  --dry-run --gpu-ids cpu
```

### Policy Bundle 与 sim-to-sim

从 checkpoint 严格加载 task/profile 对应 actor，并直接生成完整 Bundle：

```bash
uv run --package lainloco --extra cpu lainloco export /path/to/model.pt \
  --destination /path/to/go2-policy-bundle \
  --task go2/velocity-rough --profile ts-student
```

旧版 checkpoint 中存在的 actor running normalization 会被显式恢复并写入 ONNX；
旧版本缺少的 profile 固定 `min_std` buffer 会从当前组合补齐。除此之外的结构不兼容
仍会在 `strict=True` 加载时失败。

将已导出的 ONNX 与明确的 task/profile 组合封装为完整部署产物：

```bash
uv run --package lainloco --extra cpu lainloco bundle create \
  /path/to/policy.onnx /path/to/go2-policy-bundle \
  --task go2/velocity-flat --profile ppo
```

加载器会校验六个标准文件、SHA-256、ONNX 输入输出、robot/task、关节顺序、
动作缩放、观测/history/recurrent state、normalization 和控制周期。已有目标目录
不会被覆盖。可单独重载，或在新的 headless mjlab 进程中执行连续控制：

```bash
uv run --package lainloco --extra cpu lainloco bundle validate \
  /path/to/go2-policy-bundle --task go2/velocity-flat --profile ppo
MJLAB_WARP_QUIET=1 uv run --package lainloco --extra cpu \
  lainloco bundle rollout /path/to/go2-policy-bundle \
  --steps 100 --num-envs 2 --device cpu
```

真实 Go2 SDK、硬件 joint mapping 和物理安全验收仍是独立范围；当前 FSM 只提供
硬件无关的命令与安全回退边界。

### 验证 Go2 迁移契约

检查机器人资产、执行器数量和关键实体名称：

```bash
uv run --package lainloco --extra cpu lainloco validate asset
```

逐一构建 14 个迁移任务，并检查动作及 actor/critic 观测维度：

```bash
uv run --package lainloco --extra cpu lainloco validate contracts
```

执行有限步无界面回放：

```bash
uv run --package lainloco --extra cpu lainloco validate smoke \
  Mjlab-Trot-Flat-Unitree-Go2 \
  --agent random \
  --steps 4
```

### 训练和回放

训练示例：

```bash
uv run --package mjlab --extra cu128 train Mjlab-Trot-Flat-Unitree-Go2 \
  --env.scene.num-envs 4096
```

使用随机策略检查环境：

```bash
uv run --package mjlab --extra cu128 play \
  Mjlab-Trot-Flat-Unitree-Go2 --agent random
```

加载 checkpoint 回放：

```bash
uv run --package mjlab --extra cu128 play Mjlab-Trot-Flat-Unitree-Go2 \
  --checkpoint-file /path/to/model.pt
```

训练结果默认保存在根目录 `runs/`（Git 忽略）。不同任务的显存占用和稳定批量不同，正式训练前建议先用较小的 `num_envs` 完成 smoke test。

## 当前代码结构

```text
RoboLab/
├── README.md
├── pyproject.toml                              # 主产品元数据、workspace 与工具配置
├── src/lainloco/                              # 唯一主产品
│   ├── core/                                   # Spec 与 Catalog
│   ├── robots/unitree/go2/                     # Go2 任务、MDP、训练配置与部署契约
│   ├── learning/                               # CTS/DreamWaQ/AMP/Teacher-Student
│   ├── runtime/                                # Policy Bundle、ONNX 与 sim-to-sim
│   ├── workflows/                              # train/play/distill/export 编排
│   ├── bootstrap.py                           # mjlab task entry point
│   └── validation.py                          # Go2 验收入口
├── tests/
│   ├── contracts/                              # 每次提交的结构与契约拒绝
│   ├── integration/                            # ONNX、状态和连续控制边界
│   └── training/                               # GPU/容量/收敛验收规则与记录
├── tools/                                      # 仓库级验证和开发脚本
├── deploy/                                     # Go2 sim2sim 入口；硬件进程待 SDK/安全验收
├── vendor/mjlab/                               # 明确的本地后端 fork
└── runs/                                       # 本地训练输出，Git 忽略
```

包边界、领域对象、环境/训练能力和部署闭环均已迁出；`tasks/legacy.py` 仅保留旧
导入兼容。当前实现结构为：

```text
src/lainloco/
├── robots/unitree/go2/
│   ├── robot.py
│   ├── contract.py
│   ├── deploy/                                  # Policy adapters 与 Passive/Stand/Policy FSM
│   ├── tasks/
│   │   ├── locomotion/
│   │   ├── aerial/
│   │   └── balance/
│   └── training/                                # Go2 参数、observation binding 与 runner
├── learning/                                    # 通用训练实现
├── runtime/                                     # Bundle/ONNX/sim-to-sim 运行时
├── workflows/                                   # 显式 task/profile 工作流
└── bootstrap.py                                 # 注册与旧路径兼容
```

架构原则：

1. 机器人拥有技能和部署契约。
2. 任务描述环境目标，算法描述训练方法，二者不混为同一概念。
3. 通用观测、奖励、事件和算法实现保持单一来源。
4. 训练 checkpoint、ONNX 和部署端共享显式的 observation/action contract。
5. 重构期间保持现有 Task ID 和策略接口兼容。

## 与 mjlab 的关系

本项目使用 mjlab 提供的 Manager-based 环境、MuJoCo Warp 仿真、任务注册和 RSL-RL 训练接口。`vendor/mjlab/` 明确表示包含迁移所需本地改动的后端 fork；LainLoco 的机器人、任务和算法扩展保持在根级主产品中，以降低后续同步 mjlab 的维护成本。

mjlab 原始说明和开发文档见 [vendor/mjlab/README.md](vendor/mjlab/README.md)，贡献流程见
[CONTRIBUTING.md](CONTRIBUTING.md)。底层组件的许可证见 [vendor/mjlab/LICENSE](vendor/mjlab/LICENSE)，
第三方来源、使用方式和许可证说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 项目愿景

Lain's LocoLab 希望成为一个可读、可组合、可验证的腿式机器人运动学习工作台：既保留单个机器人技能项目的直观性，也保留 mjlab 在仿真组件、任务组合和批量训练方面的复用能力。

> **Lain's LocoLab — Composable locomotion learning, from simulation to hardware.**
