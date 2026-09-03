# Lain's LocoLab

> **LainLoco** — 面向腿式机器人的可组合运动学习实验场，从 MuJoCo 仿真训练到策略部署。

Lain's LocoLab 是一个建立在 [mjlab](https://github.com/mujocolab/mjlab) 之上的机器人运动学习项目。目前工作重点是将 Unitree Go2 的多种运动技能和自定义强化学习算法迁移到 MuJoCo / MuJoCo Warp，同时保持训练、回放、ONNX 导出和部署接口之间的行为契约。

仓库当前仍沿用 `RoboLab` 目录名，Python 包和任务 ID 也暂时保留 `mjlab` / `Mjlab-*` 命名。项目后续改名时计划使用：

- 展示名称：**Lain's LocoLab**
- 工程简称：**LainLoco**
- Python 包与 CLI：`lainloco`
- 视觉简称：`LLoco` 或 `L³`

## 文档导航

- [架构设计指南](ARCHITECTURE.md)：目标目录、领域对象、依赖规则、注册方式和实施顺序。
- [项目进度与验收表](PROJECT_PROGRESS.md)：迁移完整性、策略成熟度和架构实施度。
- [Go2 全量迁移计划](GO2_MIGRATION_PLAN.md)：源项目语义、任务基线和详细迁移记录。
- [mjlab 原始说明](mjlab/README.md)：底层框架安装、训练和开发文档。

## 当前状态

项目处于 Go2 全量迁移与架构整理阶段。

- 已接入 Unitree Go2 MJCF、12 关节动作接口、接触与地形传感器。
- 已注册原项目的 14 个 Go2 训练入口。
- 已接入 PPO、AMP、CTS、DreamWaQ 和 Teacher-Student 相关训练流程。
- 已提供资产检查、观测契约检查和有限步无界面 smoke test。
- 已产生短训 checkpoint 和 ONNX 导出，用于验证训练及导出链路。

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

当前 Task ID 为迁移兼容接口。后续架构会进一步区分机器人技能、环境、观测契约和训练算法，并为旧 ID 保留兼容别名。

## 快速开始

进入当前 mjlab 工作目录：

```bash
cd /home/lxy/RoboLab/mjlab
```

安装开发环境。NVIDIA GPU 环境使用：

```bash
uv sync --extra cu128
```

仅进行 CPU 结构检查时可以使用：

```bash
uv sync --extra cpu
```

列出已经注册的环境：

```bash
uv run list-envs
```

### 验证 Go2 迁移契约

检查机器人资产、执行器数量和关键实体名称：

```bash
uv run go2-asset-check
```

逐一构建 14 个迁移任务，并检查动作及 actor/critic 观测维度：

```bash
uv run go2-contract-check
```

执行有限步无界面回放：

```bash
uv run go2-smoke Mjlab-Trot-Flat-Unitree-Go2 \
  --agent random \
  --steps 4
```

### 训练和回放

训练示例：

```bash
uv run train Mjlab-Trot-Flat-Unitree-Go2 \
  --env.scene.num-envs 4096
```

使用随机策略检查环境：

```bash
uv run play Mjlab-Trot-Flat-Unitree-Go2 --agent random
```

加载 checkpoint 回放：

```bash
uv run play Mjlab-Trot-Flat-Unitree-Go2 \
  --checkpoint-file /path/to/model.pt
```

训练结果默认保存在 `mjlab/logs/`。不同任务的显存占用和稳定批量不同，正式训练前建议先用较小的 `num_envs` 完成 smoke test。

## 当前代码结构

```text
RoboLab/
├── README.md
├── GO2_MIGRATION_PLAN.md
└── mjlab/
    ├── src/mjlab/asset_zoo/robots/unitree_go2/  # Go2 资产
    ├── src/mjlab/tasks/velocity/config/go2/    # 任务注册与配置
    ├── src/mjlab/tasks/velocity/mdp/           # 观测、奖励、命令和事件
    ├── src/mjlab/tasks/velocity/rl/            # Runner 与自定义算法
    ├── src/mjlab/scripts/go2_*.py              # 迁移检查工具
    └── logs/                                   # 本地训练输出
```

当前布局服务于迁移验证，但 Go2 专用代码仍然集中在 mjlab 的 `velocity` 任务内部。后续重构目标是让机器人拥有自己的技能目录，同时让 MDP 原语和算法实现保持共享：

```text
packages/lainloco/src/lainloco/
├── robots/unitree/go2/
│   ├── robot.py
│   ├── contract.py
│   ├── tasks/
│   │   ├── locomotion/
│   │   ├── aerial/
│   │   └── balance/
│   ├── training/
│   └── deploy/
├── learning/
└── registry.py
```

架构原则：

1. 机器人拥有技能和部署契约。
2. 任务描述环境目标，算法描述训练方法，二者不混为同一概念。
3. 通用观测、奖励、事件和算法实现保持单一来源。
4. 训练 checkpoint、ONNX 和部署端共享显式的 observation/action contract。
5. 重构期间保持现有 Task ID 和策略接口兼容。

## 与 mjlab 的关系

本项目使用 mjlab 提供的 Manager-based 环境、MuJoCo Warp 仿真、任务注册和 RSL-RL 训练接口。`mjlab/` 目录目前包含迁移所需的本地改动；长期目标是把 LainLoco 的机器人、任务和算法扩展从上游框架代码中分离，降低后续同步 mjlab 的维护成本。

mjlab 原始说明和开发文档见 [mjlab/README.md](mjlab/README.md)。底层组件的许可证见 [mjlab/LICENSE](mjlab/LICENSE)。在公开发布前，还需要补齐项目自身的许可证选择和第三方资产声明。

## 项目愿景

Lain's LocoLab 希望成为一个可读、可组合、可验证的腿式机器人运动学习工作台：既保留单个机器人技能项目的直观性，也保留 mjlab 在仿真组件、任务组合和批量训练方面的复用能力。

> **Lain's LocoLab — Composable locomotion learning, from simulation to hardware.**
