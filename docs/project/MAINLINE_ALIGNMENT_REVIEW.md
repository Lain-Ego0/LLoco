# RoboLab 主线对齐审查

状态：2026-08-21 冻结，作为下一阶段范围与优先级依据。

## 1. 结论

项目在架构和产品定义上没有偏离
“One-Stop Motion Control Platform Based on Heavily Customized MJLab Tools with
Deployment Adaptation and Skill Integration”主线，但实现重心出现了阶段性的
**平台化偏移**。

当前已经形成可靠的本地控制面 MVP，但还不能称为完整的一站式运动控制平台；
`Heavily Customized MJLab` 与 `Deployment Adaptation` 尚未形成与 Skill/控制面同等成熟的
实现证据。

| 主线维度 | 当前状态 | 判断 |
|---|---|---|
| One-Stop Platform | Schema、CLI、API、Worker、WebUI、Artifact 已打通 | 🟡 骨架成立，纵向闭环不完整 |
| Heavily Customized MJLab | curated vendor、task discovery、play 参数适配 | 🟠 仍是薄适配，尚未达到深度定制 |
| Deployment Adaptation | Profile 与兼容性门禁已有，Runtime、Edge、Driver 未实现 | 🔴 当前最大缺口 |
| Skill Integration | Motion/Platform/Agent Skill、安装、权限、Action、Codex export 已实现 | 🟢 当前完成度最高 |
| Motion Control | zero-policy 可启动 Viser，真实 trained policy 尚未通过 | 🟠 尚未形成可信样板 |

## 2. 没有战略跑偏的依据

- Platform Core、Robot Profile、Skill、MJLab Adapter 与 Edge Runtime 的职责边界仍围绕运动控制；
- B1–B6 遵循“契约 -> Worker/Skill -> API/WebUI”的依赖顺序，没有扩展到云平台、多租户或通用低代码；
- Skill Integration 已有真实代码和测试，包括版本固定、权限、兼容性、Job 隔离与 AgentSkill 导出；
- 2026-08-21 CPU contract suite 实测 `113 passed, 1 warning`，基础契约层稳定。

## 3. 三个实现失衡

### 3.1 控制面成熟度高于运动闭环

当前 WebUI 可以管理 Robot、Skill、Job 和 Artifact，但 Train、Validate、Deploy 尚未成为真实工作流。
系统尚不能完成：

```text
Skill/policy -> MJLab play -> validation -> sim-to-sim
             -> DeploymentPlan -> Runtime session -> stop/safe
```

### 3.2 MJLab 仍是薄适配

当前 adapter 主要负责发现 task、构造 vendor `play.py` 命令并交给 Worker；训练配置抽象、
统一指标、导出、验证和部署尚未进入同一平台契约。因此当前实现更接近
“Platform Integration Based on Curated MJLab/Unitree Sources”。

`Heavily Customized` 的达成标准不是大规模修改 vendor，而是平台能够稳定控制并扩展：

- 统一 task/train/play/export/evaluate 入口；
- schema 化训练与运行配置；
- 结构化指标、验证证据和 artifact lineage；
- Robot Profile 驱动的任务与 runtime 适配；
- 独立 Runtime、sim-to-sim 和 Deployment Gate。

### 3.3 Deployment Adaptation 是最大缺口

- `runtime/` 仍只有接口说明；
- `integrations/unitree/` 仍只有接口说明；
- G1 Profile 当前为 simulation-only/L0；
- sim-to-sim、heartbeat、watchdog、DeploymentSession 和 `stop/safe` 尚未实现；
- 真实 G1 trained play 仍等待可用 checkpoint/WandB run。

## 4. 优先级决定

下一阶段立即停止增加横向平台页面和生态能力，转向真实 MotionSkill 的纵向部署闭环：

1. 关闭 B7，使用真实 G1 policy 完成 trained play 与 lineage；
2. 冻结 ValidationRun、DeploymentPlan、DeploymentSession 和 Runtime 协议；
3. 实现最小 ONNX runner、FSM、heartbeat、watchdog 与 safe fallback；
4. 接入 unitree_mujoco sim-to-sim；
5. 完成 Validate/Deploy 控制面和 WebUI；
6. 通过故障注入和可复现验收后，再扩展训练平台、第二机器人和内置 Agent。

仿真后端采用独立的迁移路线：

```text
Unitree backend 黄金基线
  -> SimulationBackend 防腐层
  -> RoboLab-native MJLab task/train/play
  -> 双后端等价性报告
  -> native 成为默认，Unitree 保留兼容
```

这不是立即重写 MJLab。RoboLab 继续使用开源 MJLab、MuJoCo-Warp、RSL-RL、Warp 和 Viser，
但逐步收回 Task Registry、训练/回放协议、Robot-to-task binding、schema、Policy metadata、
Validation 和 Deployment 的所有权。完整方案见
[`../reference/SIMULATION_BACKEND_STRATEGY.md`](../reference/SIMULATION_BACKEND_STRATEGY.md)。

许可证保持独立决策：RoboLab 根项目继续 MIT，Unitree vendor 和 MJLab/MuJoCo 等组件保留
各自的 Apache-2.0 或其他原许可证。

## 5. 防偏移验收原则

后续工作必须至少满足一项，否则不进入当前阶段：

- 缩短从 MotionSkill 到 validated/deployed session 的路径；
- 提高 MJLab 训练、回放、导出或验证的可复现性；
- 提高 Robot Profile/Driver/Runtime 的复用程度；
- 增强部署门禁、watchdog、stop/safe 或故障可解释性；
- 完善 Skill/Profile/Artifact/Validation/Deployment 的 lineage。

“新增页面”“新增 schema”或“新增 adapter”本身不构成完成；必须有真实 G1 policy 的运行证据，
或直接服务于上述闭环的机器验收。

## 6. 当前资源决定

当前 GPU 正在运行其他 IsaacLab 训练。RoboLab 不抢占该训练，也不并行启动 G1 GPU 任务。
待现有训练结束后，先训练或取得可复现的 G1 velocity policy，再执行下一阶段 N0；
详细工作项以 [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md) 为准。
