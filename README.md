# RoboLab

## 启动方式

在已有 `robolab` Conda 环境中启动本地 Web 平台：

```bash
cd /home/lxy/RoboLab
conda activate robolab
python -m pip install -e packages/schemas -e packages/core -e packages/mjlab_adapter \
  -e services/api -e services/worker -e apps/cli
robolab serve
```

启动后，终端会打印实际访问地址（默认监听 `127.0.0.1`，端口自动选择），在浏览器打开该地址即可。首次创建环境或需要运行 MJLab/Viser 时，先按 [`docs/reference/ENVIRONMENT_SETUP.md`](docs/reference/ENVIRONMENT_SETUP.md) 安装对应依赖；仅修改 WebUI 后，在 `apps/web/` 执行 `npm install && npm run build`，再重新启动服务。

One-Stop Motion Control Platform Based on Heavily Customized MJLab Tools with Deployment Adaptation and Skill Integration.

RoboLab 计划成为一个面向机器人运动控制的本地优先 Web 平台，将机器人接入、强化学习训练、仿真验证、策略导出、Skill 安装和实机部署组织成一条可追踪、可复现、可回滚的工作流。

> 当前仓库已完成 Phase 0–1 的 B1–B6 实现，B7 作为下一阶段 N0 关闭门禁继续验收。
> CPU contract suite 已通过；真实 G1 trained MotionSkill 仍等待当前外部 GPU 训练结束后训练或取得
> checkpoint。当前不启动 RoboLab/G1 GPU 任务。下一阶段集中完成 trained play、ValidationRun、
> sim-to-sim、DeploymentSession 与 Runtime `stop/safe` 纵向闭环。当前状态以
> [`docs/project/DEVELOPMENT_PLAN.md`](docs/project/DEVELOPMENT_PLAN.md) 为准。

## 核心边界

- **Simulation Backend**：当前由 `packages/mjlab_adapter/` 对接 Unitree 兼容路径并建立 G1 黄金基线；
  后续通过统一 Backend Registry 同时支持 `unitree_compat` 与 RoboLab 自有 `mjlab_native`。
- **Robot Profile**：描述“这台机器人是什么、关节如何映射、怎样通信、怎样保证安全”。
- **Skill Package**：统一承载运动能力、可执行平台功能和 Agent 工作流，包含入口、依赖、权限、文档、兼容性和验证规则。
- **Platform Core**：负责注册表、任务编排、产物管理、兼容性检查、审计和部署门禁。
- **WebUI**：提供机器人接入、Skill 管理、训练监控、仿真验证和部署会话界面；不承担硬实时关节控制。
- **Edge Runtime**：靠近机器人运行控制循环、SDK/DDS 通信、看门狗和急停逻辑，仅接受经过校验的高层命令。

建议的数据流：

```text
RoboLab-Skill ──安装/校验──> Skill Registry ──绑定──> Robot Profile
                                             │
WebUI ──API──> Platform Core ──任务──> Backend Registry / Job Worker
                                             │
                                      Simulation Gate
                                             │
                                      Edge Runtime ──> Robot
```

## 仓库关系

| 仓库 | 职责 | 版本策略 |
|---|---|---|
| `RoboLab` | 平台、WebUI、适配层、定制 MJLab、部署运行时 | 平台版本 |
| `RoboLab-Skill` | 可独立下载的 Skill catalog 与能力包 | 每个 Skill 独立语义版本 |

`RoboLab-Skill` 作为全部开源的 catalog monorepo：一个仓库可以包含多个 MotionSkill、PlatformSkill 和 AgentSkill，但每个 Skill 必须有独立 manifest、版本、许可证、入口、权限和兼容性声明。平台安装时固定 Git commit/tag，不直接跟随浮动的 `main`。

## 仓库入口与所有权

当前已存在并可用：

| 路径 | 内容 | 状态 |
|---|---|---|
| `vendor/unitree_rl_mjlab/` | 精选导入的 Unitree 训练/仿真/legacy 部署技术来源 | 已导入并纳入 Git；训练、回放、sim-to-sim 可按上游方式运行 |
| `packages/schemas/` | `RobotProfile`/`JointSet`/`SkillPackage` v1alpha1 JSON Schema | B1 已实现（2026-08-20） |
| `packages/core/` | 关节映射机器校验、Skill×Profile 兼容性判定、Skill lint | B1 已实现（2026-08-20） |
| `apps/cli/` | `robolab` 命令行：check、skill、agent、serve、mjlab | B1–B5 已实现 |
| `tests/contract/` | schema、兼容性、安装、Job、API、MJLab adapter 与样板 Skill 测试 | B7 CPU suite 已通过（最新记录 112 passed） |
| `skills/` | 本机 Skill 工作区（builtin/installed/dev） | B2 已实现扫描、固定安装、注册、不可变卸载保护与 prepare 审查 |
| `docs/` | RoboLab 正式文档 | 可用 |
| `THIRD_PARTY_NOTICES.md` | 第三方声明 | 可用，随依赖演进持续更新 |

以下四个入口目前**只是所有权与接口边界**，目录中只有说明文档，没有平台代码：

| 入口 | 职责 | 实现状态 |
|---|---|---|
| [`packages/mjlab_adapter/`](packages/mjlab_adapter/) | 当前 Unitree/MJLab 兼容适配的过渡入口 | B3 已实现 task 发现与受控 play；N1 后封装为 `unitree_compat`，公共 API 改用 backend/task ID |
| [`integrations/unitree/`](integrations/unitree/) | Unitree Driver/Profile，首个厂商适配器 | 接口边界，未实现；Unitree 专用逻辑暂仍在 vendor `deploy/` |
| [`runtime/`](runtime/) | 与厂商无关的 C++ FSM、推理、安全与遥测 | 接口边界，未实现；共享逻辑暂仍在 vendor `deploy/` |
| [`robots/`](robots/) | 与 Skill 解耦的 Robot Profile | 已实现首个 Unitree G1 29DoF simulation-only Profile；模型资产引用 vendor，不复制 |

## 目标工作流

1. 创建或导入 Robot Profile，完成模型、关节、驱动与安全参数校验。
2. 从本地目录或 Git 仓库安装 Skill，验证 manifest、许可证、入口、权限、依赖、产物哈希和机器人兼容性。
3. 在 WebUI 创建训练、回放、导出或评测任务，并保存完整配置与产物来源。
4. 通过离线检查、MJLab 回放和 sim-to-sim 门禁。
5. 一键执行已通过门禁的仿真 Deployment Plan；未来实机适配完成后，由本地 Edge Runtime 完成连接、安全确认与失败回退。
6. 保存部署会话、遥测摘要、策略版本和操作者审计记录，以便复现与回滚。

迁移期间，`unitree_compat` 提供 G1 黄金基线；双后端等价性通过后，RoboLab 自有
`mjlab_native` 成为默认。Skill 和公共 API 依赖稳定 task/backend ID，不依赖 vendor 脚本路径。

## 文档

- [文档索引](docs/README.md)
- [主线对齐审查](docs/project/MAINLINE_ALIGNMENT_REVIEW.md)
- [下一阶段详细执行计划（当前进度）](docs/project/DEVELOPMENT_PLAN.md)
- [总体架构](docs/reference/ARCHITECTURE.md)
- [仿真、训练与 Play 后端演进策略](docs/reference/SIMULATION_BACKEND_STRATEGY.md)
- [Skill 包规范](docs/reference/SKILL_SPEC.md)

## 安全原则

浏览器不能直接输出关节指令，Web 服务崩溃也不能让机器人保持最后一次危险命令。实机运行时必须具备本地急停、dead-man/watchdog、命令超时、限幅、状态机门禁和安全回退。未经仿真验证的 Skill 默认禁止实机激活。

## 上游声明

`vendor/unitree_rl_mjlab/` 精选导入自 [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)，基线固定为 `1425b15f`，只保留平台实际依赖的技术来源；上游宣传文档、GIF、预编译 runtime 和 Skill 产物未进入 vendor，include/exclude 规则见 `vendor/unitree_rl_mjlab/VENDOR_MANIFEST.yaml`。保留内容继续遵守 Apache-2.0；RoboLab 根目录 MIT 不替代第三方许可证。RoboLab 不是 Unitree Robotics 的官方产品。

详见 [第三方声明](THIRD_PARTY_NOTICES.md) 和 [上游记录](vendor/unitree_rl_mjlab/UPSTREAM.md)。
