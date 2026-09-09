# LLoco

LLoco 是基于 **mjlab 1.6.0** 的 Unitree 强化学习与部署项目。mjlab
作为固定版本依赖使用，LLoco 只维护机器人资产、任务差异和部署代码，避免复制整个
仿真框架。

## 目录结构

```text
LLoco/
├── src/lloco/
│   ├── assets/          # Unitree MJCF、网格和示例动作
│   ├── tasks/           # 基于 mjlab 1.6 的薄任务适配层
│   └── cli.py           # train / play / list-envs 入口
├── tests/               # LLoco 自身的兼容性测试
├── deploy/              # 实机部署源码（不提交模型和预编译运行库）
└── simulate/            # unitree_mujoco 桥接源码
```

分层原则：

- `mjlab` 负责仿真、manager、通用 MDP、runner 和 viewer。
- `lloco.assets` 负责本项目机器人描述。
- `lloco.tasks` 只表达机器人名称、接触点、动作缩放等差异。
- CLI 先注册 LLoco 任务，再复用 mjlab 1.6 的训练和回放实现。

## 安装

需要 Python 3.10–3.13；训练需要 NVIDIA GPU。

```bash
uv sync --extra cu128
```

仅做 CPU 配置检查时：

```bash
uv sync --extra cpu
```

## 使用

```bash
# 查看 Unitree 任务
uv run list-envs --keyword Unitree

# 训练
uv run train Unitree-Go2-Flat --env.scene.num-envs 4096

# 用随机动作做配置冒烟测试
uv run play Unitree-Go2-Flat --agent random --num-envs 1

# 回放本地策略
uv run play Unitree-Go2-Flat --checkpoint-file logs/.../model_1000.pt

# 把 G1 CSV 动作转换为 mjlab 跟踪格式
uv run csv-to-npz --input-file src/lloco/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1-subject2 --robot g1
```

`csv-to-npz` 直接在本地生成 NPZ，不需要 Weights & Biases。`--output-name`
只传文件名时，输出会保存到输入 CSV 的同一目录；也可传入完整路径。
G1-23DoF 动作使用 `--robot g1_23dof`。

任务命名为 `Unitree-<Robot>-Flat` / `Unitree-<Robot>-Rough`。速度任务支持
A2、As2、Go2、G1、G1-23Dof、H1_2、H2 和 R1；动作跟踪任务支持 G1 与
G1-23Dof。

## 运行验证与已知限制

截至 2026-09-09，项目的 CPU 单环境 smoke 验证覆盖了已注册任务的环境构建、
`reset` 与一步零动作执行，并检查了观测和奖励是否为有限值。所有 Go2 任务（含
Go2 skill）、基础速度任务、Cartpole、Yam 操作任务和 29-DoF G1 跟踪任务均已通过。

- 跟踪任务必须提供与机器人自由度匹配的 NPZ 动作文件。命令行训练/回放时请设置
  `motion_file`；仓库本地的 29-DoF G1 示例不能用于 `G1-23Dof` 跟踪任务。
- `Unitree-H2-Rough` 和 `Unitree-R1-Rough` 当前的 MuJoCo-Warp 接触缓冲容量不足：
  分别至少需要 `nconmax=80` 和 `nconmax=141`。提高该配置值后，两项任务可完成
  smoke；在修复默认配置前请勿直接用默认设置启动训练。
- Smoke 只验证初始化和单步数值稳定性，不代表策略已经收敛，也不替代长时间训练、
  checkpoint 回放或真机验收。

## 开发

```bash
make format
make check
```

机器人差异集中在 `src/lloco/tasks/velocity.py` 的 `PROFILES`。新增同类机器人时，
通常只需增加资产常量和一个 profile，无需复制整套 MDP。

## 部署

`deploy/` 与 `simulate/` 保留参考项目中的源码，但不再内置 ONNX Runtime、MuJoCo
二进制或训练好的策略。请按 [deploy/README.md](deploy/README.md) 配置系统依赖并把
导出的策略放到对应机器人目录。

## 上游与许可

本项目参考 `unitree_rl_mjlab` 的项目边界和部署代码，并基于 mjlab 1.6 API
重新组织。代码使用 Apache-2.0 许可；第三方组件保留各自许可。

---

# LLoco (English)

LLoco is a Unitree reinforcement-learning and deployment project built on
**mjlab 1.6.0**. mjlab is kept as a pinned dependency; LLoco maintains only
robot assets, task-specific differences, and deployment code instead of
vendoring the full simulation framework.

## Layout

```text
LLoco/
├── src/lloco/
│   ├── assets/          # Unitree MJCF files, meshes, and example motions
│   ├── tasks/           # Thin task adapters built on mjlab 1.6
│   └── cli.py           # train / play / list-envs entry points
├── tests/               # LLoco compatibility tests
├── deploy/              # Real-robot deployment source (no models or bundled runtimes)
└── simulate/            # unitree_mujoco bridge source
```

Layering principles:

- `mjlab` provides simulation, managers, common MDPs, runners, and viewers.
- `lloco.assets` owns project-specific robot descriptions.
- `lloco.tasks` expresses only task differences such as robot names, contacts,
  and action scales.
- The CLI registers LLoco tasks before reusing mjlab 1.6 training and playback.

## Installation

Python 3.10–3.13 is required. Training requires an NVIDIA GPU.

```bash
uv sync --extra cu128
```

For CPU-only configuration checks:

```bash
uv sync --extra cpu
```

## Usage

```bash
# List Unitree tasks
uv run list-envs --keyword Unitree

# Train a task
uv run train Unitree-Go2-Flat --env.scene.num-envs 4096

# Run a random-action configuration smoke test
uv run play Unitree-Go2-Flat --agent random --num-envs 1

# Replay a local policy
uv run play Unitree-Go2-Flat --checkpoint-file logs/.../model_1000.pt

# Convert a G1 CSV motion into mjlab tracking format
uv run csv-to-npz --input-file src/lloco/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1-subject2 --robot g1
```

`csv-to-npz` creates an NPZ locally and does not require Weights & Biases. If
`--output-name` is only a filename, the output is written next to the input
CSV; an absolute or relative output path may also be used. Use
`--robot g1_23dof` for G1-23DoF motions.

Tasks follow the `Unitree-<Robot>-Flat` / `Unitree-<Robot>-Rough` convention.
Velocity tasks support A2, As2, Go2, G1, G1-23Dof, H1_2, H2, and R1. Motion
tracking tasks support G1 and G1-23Dof.

## Validation status and known limitations

As of 2026-09-09, CPU single-environment smoke checks cover environment
construction, `reset`, and one zero-action step for registered tasks, while
checking that observations and rewards remain finite. All Go2 tasks (including
Go2 skills), baseline velocity tasks, Cartpole, Yam manipulation tasks, and
29-DoF G1 tracking tasks passed.

- Tracking tasks need an NPZ motion file whose degrees of freedom match the
  robot. Set `motion_file` for training or playback. The local 29-DoF G1
  example cannot be used for `G1-23Dof` tracking.
- The default MuJoCo-Warp contact-buffer capacity is currently insufficient for
  `Unitree-H2-Rough` and `Unitree-R1-Rough`: they require at least
  `nconmax=80` and `nconmax=141`, respectively. Both tasks pass smoke checks
  after this temporary configuration adjustment; do not start training with
  their defaults until the configuration is fixed.
- Smoke tests validate initialization and single-step numerical stability only.
  They do not demonstrate policy convergence and do not replace long training,
  checkpoint playback, or real-robot validation.

## Development

```bash
make format
make check
```

Robot differences are centralized in `src/lloco/tasks/velocity.py` through
`PROFILES`. Adding a similar robot typically requires only asset constants and
a profile, rather than a duplicate MDP implementation.

## Deployment

`deploy/` and `simulate/` retain source from the reference projects but do not
bundle ONNX Runtime, MuJoCo binaries, or trained policies. Follow
[deploy/README.md](deploy/README.md) to configure system dependencies, then
place exported policies in the corresponding robot directory.

## Upstream and license

This project follows the project boundary and deployment source of
`unitree_rl_mjlab`, reorganized around the mjlab 1.6 API. LLoco is licensed
under Apache-2.0; third-party components retain their own licenses.
