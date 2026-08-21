# RoboLab 开发主线

状态：规范性文档，2026-08-21 生效。本文回答“RoboLab 是什么、为什么这样开发、最终要证明什么”。阶段状态只以
[plans/README.md](../plans/README.md) 为准。

## 1. 项目定义

> RoboLab: A One-Stop Motion Control Platform Built on a Heavily Customized MJLab Toolchain, with Robot Deployment Adaptation and Skill Integration.

RoboLab 是基于深度定制 MJLab 1.6 工具链、面向商业成品机器人、实验室自研机器人、爱好者机器人和比赛机器人的
一站式运动控制开发与部署平台。

“一站式”覆盖：机器人接入、任务/MDP、训练、回放、评测、导出、PolicyArtifact、Skill、Validation、sim-to-sim、
DeploymentPlan 和安全 Runtime。它不表示云训练、多用户 SaaS 或通用低代码平台。

## 2. 技术主线

```text
MJLab 1.6 upstream revision
  -> RoboLab Customized MJLab toolchain
  -> Robot Profile / Task / MDP
  -> train / play / evaluate / export
  -> Skill / PolicyArtifact / Validation
  -> deployment adaptation / safe Runtime
  -> self-built robot + commercial robot proof
```

MJLab 1.6 位于仓库一级目录 `mjlab/`，与 Platform 同仓库但保持独立源码、测试、依赖和发布边界。RoboLab 可以修改 MJLab
核心中与运动控制工具链相关的代码，但必须记录 upstream revision、修改类型、测试、兼容影响和回滚方式。

## 3. 深度定制的含义

“Heavily Customized”不以改动行数衡量，也不要求重写 MuJoCo、MuJoCo-Warp 或 RSL-RL。它要求 RoboLab 实际拥有：

- Robot Registry 与 Profile -> MJLab config 绑定；
- Task Registry、MDP、observation、action、reward 和 termination；
- schema 化 train/play/evaluate/export；
- PolicyArtifact 与 observation/action/deployment metadata；
- 自研 actuator、sensor、控制频率和域随机化适配点；
- 结构化 metric、Validation 和回归证据；
- Skill、Profile、Artifact、DeploymentPlan 与 Runtime 的稳定关联。

WebUI、API、Worker、数据库、Skill Registry 和厂商 Driver 不应全部塞进 MJLab 核心。

## 4. 机器人范围

自研、爱好者和比赛机器人是正式主线对象，不是商业机器人支持完成后的附加功能。Robot Profile 必须允许仅凭 MJCF、
执行器、传感器和控制配置形成 simulation-first 接入；真实硬件能力后续通过 Driver、Calibration 和 SafetyProfile 增补。

Unitree RL MJLab 早期路线已经退役。Unitree 源码、adapter、G1 Profile、Integration 和旧命令均不在 active tree。未来如支持
G1，必须作为新的商业机器人适配器，通过当时稳定的公共契约重新接入。

## 5. 平台对象

- `RobotProfile`：机器人模型、关节、执行器、传感器、控制、能力与安全；
- `TaskDefinition`：稳定 task ID、版本、配置 schema、能力与 robot binding；
- `TrainingRecipe`：任务、机器人、seed、runner、资源和 resolved config；
- `PolicyArtifact`：checkpoint/ONNX、输入输出 schema、部署参数、hash 和 lineage；
- `SkillPackage`：可安装能力、入口、依赖、权限、兼容和验证规则；
- `ValidationRun`：固定输入、场景、指标、阈值、证据和结果；
- `DeploymentPlan`：Artifact、target、Runtime 配置、required gates 和 fallback；
- `DeploymentSession`：实际运行状态、heartbeat、事件、遥测和安全终态。

## 6. 独立验收原则

RoboLab 使用自己的运动、复现和安全指标验收，不以旧 Unitree/MJLab 1.2 行为逐帧等价作为正确性标准。

最终必须分别证明：

1. MJLab 基座和下游修改可追踪、可回归、可回滚；
2. RoboLab 拥有完整 task/train/play/evaluate/export 工具链；
3. 至少一个真实自研、爱好者或社区可复现机器人完成 simulation-first 接入；
4. 至少一个真实策略完成训练或可信回放、评测、导出和 Artifact 固定；
5. Runtime 通过 watchdog、stop/safe 和故障注入；
6. 自研机器人和成品机器人复用同一公共契约；
7. 任一部署会话可以追溯到代码、环境、Profile、Task、Skill 和 Artifact。

## 7. 当前边界

R0 已完成 MJLab 根目录迁移、环境固定、旧 Unitree 整栈退役、smoke 和维护规则。当前只允许推进 R1 定制工具链；首个
自研/社区参考机器人选择属于 R2，GPU 训练属于 R3，平台纵向整合属于 R4，Runtime 属于 R5，多机器人证明属于 R6。

具体执行顺序见 [当前开发计划](../plans/README.md)，跨阶段禁止事项见 [开发约束](CONSTRAINTS.md)。
