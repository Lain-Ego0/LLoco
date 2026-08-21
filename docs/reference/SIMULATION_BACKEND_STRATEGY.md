# MJLab 1.6 深度定制与仿真工具链策略

状态：2026-08-21 生效。本文定义 RoboLab 如何以 MJLab 1.6 为定制基座，构建面向成品与自研机器人的运动控制工具链。
项目总决策见 [`docs/project/MJLAB_1_6_TECHNICAL_DIRECTION.md`](../project/MJLAB_1_6_TECHNICAL_DIRECTION.md)。

## 1. 决策摘要

RoboLab 不再采用“Unitree 后端迁移为 native 后端”的长期叙事。正式路线是：

```text
固定上游 revision 的 MJLab 1.6
          ↓
RoboLab 定制 MJLab 工具链
          ↓
通用 Robot Profile / Task / MDP / Policy Artifact
          ↓
训练、回放、评测、导出与 Skill 工作流
          ↓
机器人部署适配与安全 Runtime
          ↓
自研机器人 + 成品机器人
```

Unitree RL MJLab 保留为 G1 参考和兼容路径，但不再是默认平台后端。MJLab 1.6 定制主线不等待 Unitree G1
checkpoint，也不以旧 Unitree 行为逐帧等价作为开发前提。

## 2. 代码所有权和边界

### 2.1 Customized MJLab 1.6

`vendor/mjlab/` 是当前 MJLab 1.6 下游定制基座。它可以包含 RoboLab 为运动控制工具链做的核心修改，至少覆盖：

- Robot Registry 和 robot/config 绑定；
- Task、MDP、observation、action、reward 和 termination 扩展点；
- train/play/evaluate/export 的稳定入口；
- 任务配置与 Policy Artifact metadata 生成；
- 自定义 actuator、sensor、控制频率和域随机化支持；
- 针对这些修改的单元、契约、smoke 和回归测试。

MJLab 定制代码不应包含 WebUI、Skill catalog、Artifact Store、API 数据库或厂商 SDK。平台控制面位于
`packages/`、`services/` 和 `apps/`；部署数据面位于 `runtime/`；厂商专用代码位于 `integrations/<vendor>/`。

### 2.2 Unitree reference/compat

`vendor/unitree_rl_mjlab/` 是固定 revision 的精选技术来源，允许用于：

- G1 任务、模型、参数和旧 checkpoint 诊断；
- Unitree sim-to-sim 参考；
- Unitree Driver/Robot Profile 的迁移资料。

它不应扩散到公共 API、Skill manifest、通用 task ID、平台目录或 Runtime 协议。现有
`packages/mjlab_adapter/` 是兼容过渡入口，后续可以整理为 `backends/unitree_compat/`，但这不是
RoboLab 主线的核心实现。

## 3. 稳定平台契约

平台公共层必须表达运动控制意图，而不是 vendor 脚本路径：

```python
class MotionToolchain(Protocol):
    toolchain_id: str  # 例如 robolab_mjlab@1

    def discover_tasks(self) -> list[TaskDefinition]: ...
    def train(self, recipe: TrainingRecipe) -> JobCommand: ...
    def play(self, config: PlayConfig) -> JobCommand: ...
    def evaluate(self, config: EvaluationConfig) -> JobCommand: ...
    def export(self, config: ExportConfig) -> JobCommand: ...
```

Unitree 兼容路径可以实现同一协议，但只能把 vendor task、脚本和路径保留在私有 binding 中。公共对象至少包括：

- `RobotProfile`：模型、关节、执行器、传感器和能力；
- `TaskDefinition`：稳定 task ID、版本、配置 schema 和能力；
- `TrainingRecipe`：机器人、任务、种子、资源、配置快照和代码 revision；
- `PolicyArtifact`：checkpoint/ONNX、输入输出 schema、部署参数、hash 和来源；
- `ValidationRun`：场景、指标、阈值、证据和结果；
- `DeploymentPlan`：固定 Artifact、目标机器人、Runtime 配置和安全门禁。

## 4. “深度定制”的验收标准

不以复制多少上游文件或修改多少行代码作为标准。至少同时满足：

1. 训练、回放、评测和导出可以通过 RoboLab 稳定入口调用；
2. 机器人 Profile 可以生成或校验 MJLab 所需配置；
3. observation/action/deployment schema 可以从任务和策略产物中追踪；
4. 自定义机器人不需要复制 Unitree 目录或 vendor 脚本；
5. 任务指标、验证证据和 Artifact lineage 可被平台记录；
6. 至少一个非 Unitree 机器人完成 simulation-first 接入；
7. 上游 MJLab 更新有明确的兼容性和回滚报告。

## 5. 新的实施阶段

### Stage R0：基座固定和路线切换

- 固定 MJLab 1.6 upstream commit；
- 建立 `vendor/mjlab/UPSTREAM.md`、`ROBOLAB_CHANGES.md` 和补丁分类；
- 让 MJLab 1.6 成为默认开发环境；
- 保留 Unitree 1.2 路径作为 legacy 环境，不再增加其平台功能。

### Stage R1：定制工具链基础

- 定制 Robot Registry、Task Registry 和配置加载；
- 实现统一 train/play/evaluate/export 入口；
- 生成 observation/action/schema 和 Policy Artifact metadata；
- 增加 MJLab 1.6 smoke、contract 和代表性 task 回归。

### Stage R2：机器人适配

- 完善 MJCF/Robot Profile 导入、关节和执行器诊断；
- 支持传感器、frame、控制周期、动作范围和默认姿态；
- 先完成一个非 Unitree 的自研或简化机器人 Profile；
- 再将 G1 作为普通 Unitree Profile 接入同一契约。

### Stage R3：运动策略工作流

- 完成至少一个 velocity 或 tracking task；
- 训练、回放、评测、导出和 Artifact 登记全部走 RoboLab 入口；
- 以独立运动指标验收，不要求复制 Unitree 旧策略；
- 建立 sim-to-sim 目标和迁移报告。

### Stage R4：部署适配和 Runtime

- 实现通用 ONNX runner、FSM、heartbeat、watchdog 和 stop/safe；
- 实现 simulation driver 和至少一个自研机器人 driver；
- 明确厂商专用代码只存在于 adapter/driver；
- 物理部署缺少 Driver、Calibration 或 Safety 时必须 fail closed。

### Stage R5：Skill 和一站式闭环

- MotionSkill 绑定 RobotProfile、Task、PolicyArtifact 和 DeploymentPlan；
- WebUI/CLI 完成 install、train、play、validate、export 和 simulation deploy；
- 保存完整配置、代码、依赖、Artifact、指标和会话 lineage；
- 通过故障注入和干净环境重放验收。

### Stage R6：多机器人和上游维护

- 至少一个自研机器人和一个成品机器人完成同一套工作流；
- 第二机器人不复制整套 task、Runtime 或 Skill 逻辑；
- 定期同步上游 MJLab，并记录冲突、回归和回滚方案；
- 评估是否将定制 MJLab 拆为独立 `RoboLab-MJLab` 仓库。

## 6. Unitree 兼容性门禁

Unitree 兼容性只检查迁移和回归所需的内容：

- 关节顺序、单位、坐标系和动作缩放；
- 控制频率、默认姿态和安全范围；
- 旧 checkpoint 是否可诊断或显式标记为不兼容；
- Unitree sim-to-sim 是否仍可由专用 adapter 运行。

不要求：

- MJLab 1.6 与 MJLab 1.2 逐帧 bitwise 一致；
- 新任务复制旧 reward 的全部数值；
- 新平台功能等待 Unitree checkpoint；
- Unitree backend 长期作为第二个产品主线。

## 7. 完成定义

只有在以下条件都满足时，才能称为“基于深度定制 MJLab 的一站式运动控制平台”：

- MJLab 1.6 定制基座有可追踪的上游 revision 和修改记录；
- RoboLab 拥有稳定的 task、train、play、evaluate、export 和 artifact 契约；
- 至少一个自研或非 Unitree 机器人完成 simulation-first 接入；
- 至少一个真实运动策略完成训练或回放、验证和 sim-to-sim；
- Runtime 具备独立进程、watchdog、stop/safe 和故障注入证据；
- Skill、Profile、PolicyArtifact、ValidationRun 和 DeploymentSession 可追溯；
- Unitree 仅作为一个适配器和兼容来源存在。
