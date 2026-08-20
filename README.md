# RoboLab

One-Stop Motion Control Platform Based on Heavily Customized MJLab Tools with Deployment Adaptation and Skill Integration.

RoboLab 计划成为一个面向机器人运动控制的本地优先 Web 平台，将机器人接入、强化学习训练、仿真验证、策略导出、Skill 安装和实机部署组织成一条可追踪、可复现、可回滚的工作流。

> 当前仓库处于架构固化阶段。`mjlab/` 已包含训练、回放、sim-to-sim 和 Unitree 实机部署代码；WebUI、平台后端与 Skill 管理器尚未实现。文档中的目录是目标结构，不代表相应功能已经可用。

## 核心边界

- **MJLab Integration**：由 `packages/mjlab_adapter/` 对接通用 MJLab，并使用 `vendor/unitree_rl_mjlab/` 中精选的 Unitree 训练/仿真技术来源。
- **Robot Profile**：描述“这台机器人是什么、关节如何映射、怎样通信、怎样保证安全”。
- **Skill Package**：统一承载运动能力、可执行平台功能和 Agent 工作流，包含入口、依赖、权限、文档、兼容性和验证规则。
- **Platform Core**：负责注册表、任务编排、产物管理、兼容性检查、审计和部署门禁。
- **WebUI**：提供机器人接入、Skill 管理、训练监控、仿真验证和部署会话界面；不承担硬实时关节控制。
- **Edge Runtime**：靠近机器人运行控制循环、SDK/DDS 通信、看门狗和急停逻辑，仅接受经过校验的高层命令。

建议的数据流：

```text
RoboLab-Skill ──安装/校验──> Skill Registry ──绑定──> Robot Profile
                                             │
WebUI ──API──> Platform Core ──任务──> MJLab / Job Worker
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

## 目标工作流

1. 创建或导入 Robot Profile，完成模型、关节、驱动与安全参数校验。
2. 从本地目录或 Git 仓库安装 Skill，验证 manifest、许可证、入口、权限、依赖、产物哈希和机器人兼容性。
3. 在 WebUI 创建训练、回放、导出或评测任务，并保存完整配置与产物来源。
4. 通过离线检查、MJLab 回放和 sim-to-sim 门禁。
5. 一键执行已通过门禁的仿真 Deployment Plan；未来实机适配完成后，由本地 Edge Runtime 完成连接、安全确认与失败回退。
6. 保存部署会话、遥测摘要、策略版本和操作者审计记录，以便复现与回滚。

## 文档

- [总体架构](docs/ARCHITECTURE.md)
- [产品功能与 WebUI](docs/PRODUCT_DESIGN.md)
- [WebUI 视觉与交互规范](docs/UI_GUIDELINES.md)
- [Skill 包规范](docs/SKILL_SPEC.md)
- [Agent Skill 与外部 Agent 集成](docs/AGENT_INTEGRATION.md)
- [机器人快速适配](docs/ROBOT_ADAPTATION.md)
- [路线图](docs/ROADMAP.md)
- [MVP 验收标准](docs/MVP_ACCEPTANCE.md)
- [上游来源、许可证与致谢](docs/UPSTREAM_AND_ACKNOWLEDGEMENTS.md)
- [MJLab 来源树管理方案](docs/MJLAB_MAINTENANCE.md)
- [待确认决策与多轮 QA](docs/DECISIONS_AND_QA.md)

## 安全原则

浏览器不能直接输出关节指令，Web 服务崩溃也不能让机器人保持最后一次危险命令。实机运行时必须具备本地急停、dead-man/watchdog、命令超时、限幅、状态机门禁和安全回退。未经仿真验证的 Skill 默认禁止实机激活。

## 上游声明

当前工作区的 `mjlab/` 来源于 [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)。正式导入时将迁入 `vendor/unitree_rl_mjlab/`，并只保留实际需要的技术来源；上游宣传文档、GIF、预编译 runtime 和 Skill 产物不进入 active vendor。保留内容继续遵守 Apache-2.0；RoboLab 根目录 MIT 不替代第三方许可证。RoboLab 不是 Unitree Robotics 的官方产品。

详见 [第三方声明](THIRD_PARTY_NOTICES.md) 和 [上游记录](mjlab/UPSTREAM.md)。
