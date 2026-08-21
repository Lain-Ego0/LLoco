# RoboLab

> **A One-Stop Motion Control Platform Built on a Heavily Customized MJLab Toolchain, with Robot Deployment Adaptation and Skill Integration.**

RoboLab 是一个基于深度定制 MJLab 1.6 工具链、面向商业成品机器人、实验室自研机器人、爱好者机器人和比赛机器人的
一站式运动控制开发与部署平台。

平台计划把 Robot Profile、运动任务、强化学习训练、回放、评测、策略导出、Skill、sim-to-sim 和安全部署组织成一条
可追踪、可复现、可回滚的工作流。Unitree G1 是已退役的早期样板；未来成品机器人选型在 R6 前置工作中决定，不预设品牌。

## 当前状态

当前仓库已经完成 Phase 0–1 的平台控制面骨架，包括 schema、Skill、CLI、Worker、API、WebUI 和 Artifact Store。
历史 Unitree/MJLab 1.2 路径曾能够进行 task discovery 和受控 play，但相关 active 栈已退役；它只作为 legacy/reference 保留在历史记录中。

新的开发主线正在切换到仓库中的 MJLab 1.6 源码，并将建立 RoboLab 定制的 Robot/Task Registry、
train/play/evaluate/export、通用机器人接入和部署 Runtime。当前尚不能宣称以下能力已完成：

- MJLab 1.6 深度定制工具链；
- 自研机器人完整纵向样板；
- 真实策略的训练、验证和 sim-to-sim 闭环；
- DeploymentSession、watchdog 和 Runtime `stop/safe`；
- 受控实机部署。

当前状态和工作顺序以 [`docs/plans/README.md`](docs/plans/README.md) 为唯一权威。

## 启动现有本地平台

```bash
cd /home/lxy/RoboLab
conda activate robolab
python -m pip install -e mjlab -e packages/schemas -e packages/core \
  -e services/api -e services/worker -e apps/cli
robolab serve
```

服务默认监听 `127.0.0.1`，端口自动选择。首次创建环境或运行 MJLab/Viser 前，先阅读
[`docs/guide/ENVIRONMENT.md`](docs/guide/ENVIRONMENT.md)。Unitree/MJLab 1.2 旧栈已从 active tree 删除，
当前不提供其安装或 play 命令。

## 核心边界

- **Customized MJLab 1.6 Toolchain**：RoboLab 的正式运动仿真和训练基座，负责机器人、任务、MDP、训练、回放、评测、导出和指标。
- **Robot Profile**：描述机器人模型、关节、执行器、传感器、控制周期、任务和部署绑定；不要求机器人来自某个厂商。
- **Skill Package**：统一承载运动能力、平台功能和 Agent 工作流，包含入口、依赖、权限、兼容性和验证规则。
- **Platform Core**：负责 registry、Job、Artifact、兼容性、lineage、审计和部署门禁。
- **WebUI**：管理机器人接入、Skill、训练、验证和部署会话；不承担硬实时关节控制。
- **Edge Runtime**：靠近机器人运行策略、Driver、FSM、watchdog、遥测和 stop/safe，不依赖浏览器持续在线。
- **历史 Unitree Reference**：仅在历史文档与 Git 历史中保留来源和迁移事实，不属于 active toolchain。

```text
Robot MJCF/Profile ──> Customized MJLab 1.6 ──> PolicyArtifact
       │                         │                     │
       └── Robot Registry        └── Train/Evaluate    ├── MotionSkill
                                                       └── Validation
WebUI/CLI/Agent ──> Platform Core ──> DeploymentPlan ──> Edge Runtime ──> Robot
```

## 仓库入口与所有权

| 路径 | 职责 | 当前状态 |
|---|---|---|
| `mjlab/` | MJLab 1.6 下游定制基座 | 默认安装和 smoke test 入口 |
| `packages/schemas/` | Robot、Skill、Artifact、Validation 和 Deployment schema | 已有 v1alpha1 基础，后续按 R0–R6 演进 |
| `packages/core/` | Registry、兼容性、Skill lint、Job/Artifact 基础 | B1–B6 骨架已实现 |
| `packages/mjlab_tasks/` | 目标：RoboLab task、MDP 和 robot binding 扩展 | 尚未建立 |
| `robots/` | Robot Profile 与模型 binding | R2 选择首个自研或社区参考机器人 |
| `runtime/` | 通用部署数据面、FSM、推理、安全和遥测 | 仅有接口说明 |
| `integrations/` | 具体厂商或硬件 Driver adapter | 仅有通用边界说明，当前无厂商实现 |
| `skills/` | builtin/installed/dev Skill 工作区 | 安装、注册和调用骨架已实现 |

## 目标工作流

1. 导入或创建机器人 MJCF 和 Robot Profile，完成模型、关节、执行器、传感器和控制参数检查；
2. 在 Customized MJLab 1.6 中绑定任务，生成 observation/action schema；
3. 通过 WebUI、CLI 或 Skill 创建 train/play/evaluate/export Job；
4. 固定 checkpoint、ONNX、resolved config、schema、hash 和代码 revision 为 PolicyArtifact；
5. 运行离线检查、MJLab 回放、指标评测和 sim-to-sim；
6. 创建经过门禁的 DeploymentPlan，由 Edge Runtime 启动 simulation 或未来的 physical session；
7. 保存 Runtime 状态、遥测摘要、故障、stop/safe 和完整 lineage。

## 文档

- [文档索引](docs/README.md)
- [使用指南](docs/guide/README.md)
- [开发主线](docs/development/MAINLINE.md)
- [开发约束](docs/development/CONSTRAINTS.md)
- [当前开发计划](docs/plans/README.md)
- [R1 计划](docs/plans/R1_MJLAB_TOOLCHAIN.md)
- [Robot Profile 规范](docs/specifications/ROBOT_PROFILE.md)
- [Skill 包规范](docs/specifications/SKILL.md)
- [Runtime 与部署规范](docs/specifications/RUNTIME_AND_DEPLOYMENT.md)

## 安全原则

浏览器不能直接输出关节指令，Web 服务崩溃也不能让机器人保持最后一次危险命令。实机运行时必须具备本地急停、
dead-man/watchdog、命令超时、限幅、状态机门禁和安全回退。未经仿真验证的 Skill 默认禁止实机激活；缺少 Driver、
Calibration 或 SafetyProfile 时 physical target 必须保持不可用。

## 上游与许可证

`mjlab/` 来源于开源 MJLab，固定为 `v1.6.0` / `0fb8a681136be94ffc636a3dd423cabb97d91f10`，并按
[`mjlab/UPSTREAM.md`](mjlab/UPSTREAM.md) 与 [`mjlab/ROBOLAB_CHANGES.md`](mjlab/ROBOLAB_CHANGES.md) 维护。
Unitree 来源、commit `1425b15f` 和历史许可证事实保留在 Git 历史与历史文档中，不属于当前 active tree。

RoboLab 不是 Unitree Robotics、MJLab、MuJoCo 或其他上游项目的官方产品。详见
[第三方声明](THIRD_PARTY_NOTICES.md) 和 [上游与致谢](docs/legal/UPSTREAM_AND_ACKNOWLEDGEMENTS.md)。
