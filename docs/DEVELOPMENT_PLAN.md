# RoboLab 开发主线与批次计划

状态：v0.1（2026-08-20 建立）。本文是执行层主线：批次顺序、依赖、状态与追溯。

分工：[ROADMAP](ROADMAP.md) 管“阶段与退出条件”，[MVP_ACCEPTANCE](MVP_ACCEPTANCE.md) 管“怎么算验收”，本文管“现在该干什么、按什么顺序、做到哪了”。三者冲突时以决策记录 [DECISIONS_AND_QA](DECISIONS_AND_QA.md) 为准，修改决策必须回到决策文档更新。

## 当前位置

- ✅ 设计基线冻结（Q1–Q26、D-001–D-032）
- ✅ 精选 vendor 导入与外部 runtime 发现（`vendor/unitree_rl_mjlab/`）
- ✅ 文档与所有权边界批次（四个入口 stub：adapter / integration / runtime / robots）
- ✅ **B1 契约冻结（2026-08-20）**；`robolab` conda 主环境已建立（Phase 0 完成标准见 B1 退出小节）
- ✅ **B6 最小 WebUI 完成（2026-08-20）**；下一批为 B7 MVP 验收演练

## 批次总览

```text
B1 契约冻结 ──> B2 Profile 样板与 Skill 安装 ──> B3 Job 协议与 Worker
      │                 │                            │
      │                 └────────────┬───────────────┘
      │                              ▼
      │                       B4 三个样板 Skill 与 CLI 端到端
      │                              │
      ▼                              ▼
B1 同时解锁 ─────────────────> B5 平台 API 与 robolab serve
                                     │
                                     ▼
                              B6 最小 WebUI ──> B7 MVP 验收演练
```

依赖原则：契约（B1）是一切的前置；样板 Skill（B4）同时依赖安装链路（B2）和 Job 运行（B3）；WebUI（B6）只消费已稳定的 API（B5），不提前开工（D-026）。

## B1 契约冻结（Phase 0 收尾）

目标：冻结 `RobotProfile v1alpha1` 与 `SkillPackage v1alpha1` 最小 schema，并提供可机器执行的兼容性判定，达成 Phase 0 退出条件——“任何人能解释一份 Skill 为什么与某 Robot Profile 兼容或不兼容”。

| # | 任务 | 依据 | 验收映射 | 状态 |
|---|---|---|---|---|
| B1.1 | `packages/schemas`：RobotProfile v1alpha1 JSON Schema，字段按 ROBOT_ADAPTATION §2 固化 | D-019、Q14 | §6 schema 错误可解释 | ✅ 2026-08-20 |
| B1.2 | `packages/schemas`：SkillPackage v1alpha1 JSON Schema，覆盖 MotionSkill/PlatformSkill/AgentSkill 三种 kind | D-003、D-012、Q8/Q9 | §3.1、§4、§5 | ✅ 2026-08-20 |
| B1.3 | 关节映射契约的机器校验：名称唯一、映射双射、索引连续性、数组长度、限位方向 | D-005、ROBOT_ADAPTATION §4 | §3.3 | ✅ 2026-08-20 |
| B1.4 | `packages/core`：compatibility 判定（capability + Profile 版本），输出可解释原因 | D-005 | Phase 0 退出条件 | ✅ 2026-08-20 |
| B1.5 | `robolab check` CLI：对 Profile/Skill 运行 schema + 兼容性检查并给出原因 | D-026、Q22 | §6 | ✅ 2026-08-20 |
| B1.6 | 基础 lint：许可证声明、artifact SHA-256 校验规则 | D-009、D-005 | §3.1 | ✅ 2026-08-20 |

实现备注：关节映射固化为独立 `JointSet v1alpha1` 文档（`packages/schemas`），被 RobotProfile `description.jointSet` 引用；ONNX 维度实测属于 B4 contract test。CLI 落位 `apps/cli`（与 `apps/web` 并列的用户入口），B5 的 `robolab serve` 将挂在同一入口下。observation/action schema 的机器比对等 B2 任务绑定导出 schema id 后启用，当前输出“记录待比对”提示。

退出：ROADMAP Phase 0 的六项中，schema 冻结（B1.1/B1.2）与 lint/校验（B1.6）由本批完成；剩余四项分别由 B2.1（首个 simulation-only 样板）、B4.1（velocity MotionSkill）、B4.2（可执行 PlatformSkill）完成。**Phase 0 完成 = B1 + B2.1 + B4.1 + B4.2（含其依赖）全部完成。**

## B2 Profile 样板与 Skill 安装链路

| # | 任务 | 依据 | 验收映射 |
|---|---|---|---|
| B2.1 | 选定 vendor 中 G1 29DoF 的公开 MJCF 作为首个 simulation-only 样板，并落成 `robots/unitree.g1.29dof/` Profile，通过 B1 校验器自证 L0（本任务的交付物同时是 ROADMAP Phase 0 的“样板选定”） | D-006、D-019、Q14 | §3.2 | ✅ 2026-08-20 |
| B2.2 | Skill 扫描：发现 `builtin/`、`installed/`、`dev/`，标出来源与可变性 | D-014 | §6 | ✅ 2026-08-20 |
| B2.3 | 从本地 RoboLab-Skill checkout 安装单个 Skill：解析、固定 revision、校验、复制到 `installed/`、注册 | D-002、D-029、Q25 | §6 | ✅ 2026-08-20 |
| B2.4 | 不可变安装约束：同一 `id@version` 内容不同被拒绝；卸载保留被历史 Job 引用的哈希 | D-014 | §6 | ✅ 2026-08-20 |
| B2.5 | 权限审查与 Conda prepare 的显式步骤（不自动执行 `setup.sh`） | D-022、Q17 | §4、§6 | ✅ 2026-08-20 |

依赖：B1（schema 与校验器）。Conda 主环境假定已可用。

## B3 Job 协议与 Worker

| # | 任务 | 依据 | 验收映射 |
|---|---|---|---|
| B3.1 | `robolab-job-v1` 子进程协议：Job 输入、`events.jsonl`、`result.json`、取消与清理 | D-004、Q22 | §3.5/3.6、§6 | ✅ 2026-08-20 |
| B3.2 | 本地 Worker：独立进程/进程组运行 Job，不以 root 运行第三方代码 | D-004、架构约束 | §2、§6 | ✅ 2026-08-20 |
| B3.3 | 统一 action registry 初版：CLI 与后续 WebUI/Agent 共用 | Q7、ARCHITECTURE §4.2 | §6 | ✅ 2026-08-20 |
| B3.4 | `packages/mjlab_adapter` 最小版：发现 vendor registry task，受控子进程调用 `scripts/play.py`，采集日志与退出码；MVP 只发现/展示训练 task/config/等价 CLI，`train.py` 启动属下一里程碑 | D-024、Q19 | §3.5、ARCHITECTURE §4.1 | ✅ 2026-08-20 |

依赖：B1。注意 adapter 只调用 vendor 现有入口，不重构 vendor 内部。

## B4 三个样板 Skill 与 CLI 端到端（RoboLab-Skill 仓库）

| # | 任务 | 依据 | 验收映射 |
|---|---|---|---|
| B4.1 | G1 Velocity MotionSkill：只携带自身 ONNX/deploy 参数，引用 G1 Profile 与 MJLab task，不复制 XML/mesh/runtime | D-017、D-027、Q9/Q23 | §3 全部 | ✅ 2026-08-20 |
| B4.2 | MJCF Inspector PlatformSkill：报告 schema（`report.json`/`report.md`/`robot_profile.draft.yaml`）与只读权限默认 | D-015、Q12 | §4 全部 | ✅ 2026-08-20 |
| B4.3 | Robot Onboarding AgentSkill：`SKILL.md` + `skill.yaml` + `references/`，Codex `.agents/skills/` 导出原型 | D-011、D-012、Q8/Q9 | §5 全部 | ✅ 2026-08-20 |
| B4.4 | CLI 端到端：安装三个样板 → 校验 → 调用 → 查看结果，全程无需手工复制文件 | D-026、Q22 | §8 | ✅ 2026-08-20 |

依赖：B2（安装链路、G1 Profile）、B3（Job 运行）。

## B5 平台 API 与 `robolab serve`

| # | 任务 | 依据 | 验收映射 |
|---|---|---|---|
| B5.1 | FastAPI API（仅 loopback）、SQLite 元数据、content-addressed artifact store | D-001、D-018 | §2 | ✅ 2026-08-20 |
| B5.2 | Job 日志、取消、状态、配置 snapshot；产物 lineage 记录 | 架构 §2.2 | §3.6/3.8 | ✅ 2026-08-20 |
| B5.3 | `robolab serve` 统一启动 API、静态 WebUI 占位与 Worker；端口选择与 URL 输出 | D-028、Q10、Q24 | §2 | ✅ 2026-08-20 |
| B5.4 | 启动健康检查：Python、Conda、MuJoCo、MJLab、GPU、磁盘 | 验收 §2 直接要求 | §2 | ✅ 2026-08-20 |

依赖：B3（Worker/Job 协议）、B4（有可安装的内容可演示）。

## B6 最小 WebUI

| # | 任务 | 依据 | 验收映射 |
|---|---|---|---|
| B6.1 | React + TypeScript 骨架，一级导航 Dashboard/Robots/Skills/Jobs/Artifacts/Settings | D-018、D-023、Q18 | §2 | ✅ 2026-08-20 |
| B6.2 | Skills 页：安装、版本固定、权限展示、actions 调用、运行状态 | D-014、D-022 | §3.4、§6 | ✅ 2026-08-20 |
| B6.3 | Jobs 页：实时日志、阶段、运行时间、停止按钮 | — | §3.6 | ✅ 2026-08-20 |
| B6.4 | Robots 页：G1 Profile 展示与兼容性矩阵（Profile × Skill × hash） | ROBOT_ADAPTATION §8 | §3.2 | ✅ 2026-08-20 |
| B6.5 | 视觉遵守 UI_GUIDELINES：中文默认、克制低饱和、无渐变/发光/玻璃拟态 | D-020、Q15 | §2 | ✅ 2026-08-20 |

依赖：B5。Train/Validate/Deploy/Agent 只作详情入口或禁用，不制造空页面（Q18）。

## B7 MVP 验收演练

| # | 任务 | 依据 | 验收映射 |
|---|---|---|---|
| B7.1 | 按 MVP_ACCEPTANCE §8 在干净机器完整演练三条路径 | — | §8 |
| B7.2 | CPU CI：schema、安装、Job 协议、MJCF Inspector contract test | D-030、Q26 | §6 |
| B7.3 | 本地环境测试记录：MJLab play / Viser 手动验证清单 | D-030 | §3 |

依赖：B4、B6。

## 维护规则

1. 每完成一个任务就地更新状态（⬜/🔶/✅ + 日期），本文是“当前进度”的唯一权威来源；
2. 新增或重排批次必须说明依据的决策 ID；与既有决策冲突时先更新 DECISIONS_AND_QA；
3. 超出 MVP 范围的内容（MVP_ACCEPTANCE §7）不进入本文批次，只在 ROADMAP Phase 2+ 跟踪；
4. 批次内部可并行，跨批次严格按依赖顺序；不允许为赶进度绕过 B1 契约直接写功能代码。
