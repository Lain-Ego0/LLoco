# RoboLab 主线对齐审查

状态：2026-08-21 修订。本文件评估当前实现与项目原始愿景的差距，并约束后续优先级。

## 1. 原始愿景

> One-Stop Motion Control Platform Based on Heavily Customized MJLab Tools with Deployment Adaptation and Skill Integration.

正式解释见 [`MJLAB_1_6_TECHNICAL_DIRECTION.md`](MJLAB_1_6_TECHNICAL_DIRECTION.md)：RoboLab 以 MJLab 1.6
为深度定制基座，服务成品与自研机器人，统一运动任务开发、训练、验证、Skill 和部署适配。

“面向不同机器人”是项目目标，不是把“厂商中立”作为新的产品主叙事。爱好者、实验室和比赛参赛者的自研机器人与
商业成品机器人具有同等优先级。

## 2. 当前客观状态

| 主线维度 | 当前状态 | 判断 |
|---|---|---|
| One-Stop Platform | Schema、CLI、API、Worker、WebUI、Artifact 骨架已打通 | 🟡 控制面骨架成立，纵向闭环未完成 |
| Heavily Customized MJLab | MJLab 1.6 源码已在仓库，但当前平台仍主要调用 Unitree/MJLab 1.2 脚本 | 🔴 尚未开始正式 1.6 定制主线 |
| Robot Adaptation | G1 simulation-only Profile 已存在，通用自研机器人接入工具链未实现 | 🔴 与原始目标有明显差距 |
| Deployment Adaptation | Runtime、Driver、ValidationRun、DeploymentSession 未实现 | 🔴 当前核心缺口 |
| Skill Integration | Motion/Platform/Agent Skill、安装、权限、Action 和导出已实现 | 🟢 当前完成度最高 |
| Motion Control Evidence | zero-policy 可启动 Viewer，真实策略纵向闭环尚未完成 | 🟠 尚无完整可信样板 |

CPU contract suite 和已有 B1–B6 实现仍然有效，但不能替代 MJLab 1.6 定制、真实运动策略和部署 Runtime 的证据。

## 3. 已确认的方向偏移

项目早期为了快速建立 G1 样板，选择 `unitree_rl_mjlab` 作为直接实现入口。该选择提供了模型、任务、部署和 sim-to-sim
参考，但随后形成了以下不合理依赖：

- 默认环境仍固定 MJLab 1.2；
- task discovery 和 play 依赖 vendor 脚本；
- G1 checkpoint 被设置成全部后续工作的前置门禁；
- `unitree_compat` 与 `mjlab_native` 被设计为长期对等后端；
- 双后端等价被错误地提升为新主线发布前提；
- 自研机器人被排到第二机器人或后期阶段。

这些安排偏离了“深度定制 MJLab、服务自研和成品机器人”的原始愿景。偏移来自早期实现选型，不代表项目愿景需要修改。

## 4. 修正后的主线

```text
MJLab 1.6 upstream revision
  -> RoboLab customized MJLab toolchain
  -> custom/non-Unitree robot onboarding
  -> task/train/play/evaluate/export
  -> Skill + Policy Artifact + Validation
  -> deployment adaptation + safe Runtime
  -> custom robot and commercial robot proof
```

Unitree 路线在 R0.6 前仅作为退役输入，R0.6 后不再存在于 active tree：

```text
Unitree RL MJLab
  -> historical G1 assets/config/deployment evidence
  -> Git history and upstream commit for audit/recovery
  -> future G1 support only through a new MJLab 1.6 adapter
```

## 5. 优先级修订

下一阶段按以下顺序推进：

1. 冻结技术决策，固定 MJLab 1.6 upstream revision 和依赖；
2. 将 MJLab 移到根目录并建立 1.6 默认环境，在 R0.6 删除 1.2 legacy 环境；
3. 建立 RoboLab 对 MJLab 的修改记录、回归测试和统一运动工具链入口；
4. 接入一个非 Unitree 的自研或简化参考机器人；
5. 完成真实 task 的 train/play/evaluate/export 和 Policy Artifact；
6. 将已有 Skill、Job、API 和 WebUI 接到新工具链；
7. 实现部署 Runtime、sim-to-sim、watchdog 和 stop/safe；
8. R0.6 后如仍需要 G1，再将其作为普通商业机器人适配器重新实现；
9. 用至少两个不同来源机器人验证抽象。

GPU 训练资源不再阻塞第 1–4、6 和 Runtime 骨架工作。真实策略质量验收仍必须等待可用 GPU，不能用 zero-policy 替代。

## 6. 防偏移验收原则

后续工作必须直接满足至少一项：

- 增强 RoboLab 对 MJLab 1.6 任务、机器人、训练、回放、评测或导出的控制能力；
- 降低自研机器人从 MJCF/Profile 到可训练环境的接入成本；
- 缩短 Skill/Policy 到验证和部署会话的路径；
- 提高 Runtime、watchdog、stop/safe 或故障解释能力；
- 完善 Profile、Task、Artifact、Validation 和 Deployment 的 lineage；
- 提高 MJLab upstream sync 的可维护性和可回滚性。

仅新增页面、schema、adapter 或文档不构成阶段完成。每阶段必须有机器测试、可运行样板或可复查证据。

## 7. 对外描述规则

项目愿景可以使用“一站式”和“深度定制 MJLab”，但当前状态必须同时说明：

- 平台控制面和 Skill 骨架已经实现；
- MJLab 1.6 定制、自研机器人纵向样板和安全部署仍在开发；
- Unitree G1 是历史首个样板和后续适配对象，不是 RoboLab 的产品身份；
- physical target 在 Driver、Calibration 和 Safety 验收前保持不可激活。

## 8. 当前资源决定

当前 GPU 正在运行其他 IsaacLab 训练，RoboLab 不抢占、不终止该任务。该限制只影响需要 GPU 的训练与评测；
MJLab 1.6 环境、源码维护、CPU smoke test、Robot Profile、MJCF 诊断、平台契约和 Runtime 骨架可以继续推进。

详细工作项以 [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md) 为准。
