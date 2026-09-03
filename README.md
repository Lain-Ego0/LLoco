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
  --output-name dance1-subject2
```

任务命名为 `Unitree-<Robot>-Flat` / `Unitree-<Robot>-Rough`。速度任务支持
A2、As2、Go2、G1、G1-23Dof、H1_2、H2 和 R1；动作跟踪任务支持 G1 与
G1-23Dof。

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
