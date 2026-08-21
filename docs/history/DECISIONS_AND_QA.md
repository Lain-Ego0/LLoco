# 决策记录与多轮 QA

本文把聊天中的产品选择固化为仓库决策。已确认项直接约束后续实现；待确认项在下一轮回答后更新。

## 已知代码事实

- `vendor/unitree_rl_mjlab/` 精选导入自 `unitreerobotics/unitree_rl_mjlab`，基线固定为 `1425b15f`，已纳入 RoboLab Git（导入前本地完整工作副本通过完整 blob 比对验证）。
- 现有代码支持 velocity、tracking、MJLab play、ONNX 导出、unitree_mujoco 和多种 Unitree 机器人部署目录。
- `RoboLab-Skill` 是独立公开仓库，目前从最小 README/LICENSE 开始建设。
- 仓库已有多种 Unitree XML/MJCF，可用于没有实机条件时的 simulation-first 开发。
- B1 已完成（2026-08-20）：`packages/schemas`（RobotProfile/JointSet/SkillPackage v1alpha1 JSON Schema）、`packages/core`（关节映射机器校验、兼容性判定、lint）、`apps/cli`（`robolab check`）已落地，`tests/contract` 91 项通过；`robolab` conda 主环境已建立并 editable 安装三包。
- B1 对 D-016 的解释：`source.revision` 为 `main`/分支名时 lint 报错；为 `refs/tags/*` 时接受但警告建议固定 commit SHA（catalog 现有 g1_velocity 即使用 tag）；完整 40 位 SHA 无警告。
- MJLab `BuiltinPositionActuator` 约定 actuator 与目标关节同名（`spec.add_actuator(name=joint_name)`），JointSet 的 `mjcf.actuator` 沿用该约定。

## 第一轮已确认决策

| ID | 决策 | 状态 |
|---|---|---|
| D-001 | 个人本地使用，只监听本机，不建设服务器、多用户或云端模式 | 已确认 |
| D-002 | `RoboLab-Skill` 是全部开源的统一 catalog；安装到 RoboLab 本地 Skill 目录 | 已确认 |
| D-003 | Skill 可包含可直接运行的平台功能、运动能力或写给 Agent 的 Skill | 已确认 |
| D-004 | 可执行 Skill 必须有 manifest、文档和入口，由 Worker 独立进程运行 | 架构约束 |
| D-005 | Robot Profile 与 Skill 分离，通过 capability/schema/hash 显式匹配 | 架构约束 |
| D-006 | 首版使用公开 XML/MJCF 和仓库已有 Unitree 模型，仿真优先 | 已确认 |
| D-007 | WebUI 保留一键部署；实机驱动、标定、电机通信和传感器接口先定义后实现 | 已确认 |
| D-008 | 使用 Conda，首版不考虑 Docker | 已确认 |
| D-009 | 平台、Skill、模型和动作以全部开源为目标，每项仍保留自身许可证 | 已确认 |
| D-010 | 根项目 MIT 与 `vendor/unitree_rl_mjlab/` Apache-2.0、其他第三方许可证并存 | 已确认 |
| D-011 | 外部开发 Agent 优先；先做 AgentSkill registry/adapter，平台内 Agent 后续接入 | 已确认 |
| D-012 | AgentSkill 使用根目录 `SKILL.md` + RoboLab `skill.yaml` | 已确认 |
| D-013 | Web 技术栈必须本地直接运行，不依赖云服务 | 已确认 |
| D-014 | 正式 Skill 不可变安装，开发 Skill 使用 `skills/dev` | 已确认 |
| D-015 | 首版 MJCF 做预览、诊断、手动语义映射和已有 task 绑定 | 已确认 |
| D-016 | MVP 不使用 LFS/Release；Skill 不复制 MJLab/环境/公共模型，仅带独有小型产物 | 已确认 |
| D-017 | 使用 G1 Velocity、MJCF Inspector、Robot Onboarding 验收三类 Skill | 已确认 |
| D-018 | FastAPI/Pydantic/SQLite + React/TypeScript 以本地单命令方式运行 | 已确认 |
| D-019 | 首个完整 simulation-only Profile 使用 Unitree G1 29DoF | 已确认 |
| D-020 | UI 中文默认、桌面优先、克制低饱和；禁止渐变和明显 AI 风格 | 已确认 |
| D-021 | 首版复用 MJLab/Viser/MuJoCo viewer，不重写 3D 引擎 | 已确认 |
| D-022 | 只读/普通仿真可直接运行；依赖、Profile 和高风险动作分级确认 | 已确认 |
| D-023 | 首版一级导航限定为 Dashboard/Robots/Skills/Jobs/Artifacts/Settings | 已确认 |
| D-024 | MVP 只发现/展示训练 task/config/CLI，完整训练管理后置 | 已确认 |
| D-025 | workspace 运行数据放 `var/`，全局偏好放用户配置目录 | 已确认 |
| D-026 | 先实现 schema、Skill/Job/CLI 和样板，再开发 WebUI | 已确认 |
| D-027 | G1 Velocity Skill 只携带自身 ONNX/deploy 参数，不复制公共环境 | 已确认 |
| D-028 | `robolab serve` 统一启动本地 API、静态 WebUI 和 Worker | 已确认 |
| D-029 | 开发时优先发现同级 RoboLab-Skill，发布时使用可配置 Git URL | 已确认 |
| D-030 | CPU CI 覆盖协议/Inspector；MJLab viewer 本地测试；实机 CI 后置 | 已确认 |
| D-031 | Unitree 上游采用精选 vendor 导入，排除宣传文档、预编译 runtime 和 Skill 产物 | 已确认 |
| D-032 | Unitree 是首个 integration，不作为 RoboLab 品牌或通用 API 命名来源 | 已确认 |

## 第一轮回答带来的设计变化

- Skill 从纯模型包升级为统一的 `MotionSkill`、`PlatformSkill` 和 `AgentSkill`。
- “下载即可调用”通过 `skills/installed`、统一 action registry 和 Conda prepare 实现。
- Python/CLI/C++ Skill 不直接 import 到 API，避免依赖冲突和插件崩溃拖垮平台。
- AgentSkill 可以指导或调用平台工具，但权限来自 manifest/tool registry，而不是自然语言内容。
- Robot Profile 允许只有 simulation target；physical capability 不完整时部署按钮明确禁用。
- “一键部署”执行完整 Deployment Plan，首版优先实现一键仿真部署。

## 第二轮结果

### Q7. Agent 是否内置在 WebUI

结果：外部开发 Agent 优先。RoboLab 先让 Codex、Claude、DeepSeek 等读取/导出 AgentSkill 并调用稳定工具；平台内置 Agent 是后续优先增强。

### Q8. AgentSkill 文件兼容性

结果：接受根目录 `SKILL.md` + `skill.yaml`。Codex 可导出到 `.agents/skills/`；其他 Agent 使用薄适配器，不假定所有厂商发现规则相同。

### Q9. 第一组示例 Skill

结果：接受。三个示例是验证三种插件运行路径的最小样板，同时作为官方开发示例：

1. `G1 Velocity`：验证 MotionSkill 安装后可直接 MJLab play；
2. `MJCF Inspector`：验证可执行 PlatformSkill 能读取模型并生成报告；
3. `Robot Onboarding`：验证 AgentSkill 能指导开发者生成 simulation-only Profile。

### Q10. Web 技术栈与本地运行

结果：必须支持本地直接运行。目标使用方式是激活 Conda 后执行一个命令，例如 `robolab serve`，本地 FastAPI 同时提供 API 和编译后的 React 静态页面，浏览器访问 `127.0.0.1`。运行期不要求云服务，也不要求用户安装 Node；Node 只用于前端开发/构建。

### Q11. 安装与开发模式

结果：接受不可变正式安装和 `skills/dev` 开发链接。

### Q12. 首版模型导入深度

结果：接受模型预览、诊断、手动 canonical joint 映射和已有 task 绑定，不自动推断全部训练参数。

### Q13. 大文件发布

结果：MVP 暂不需要 LFS/Release。Skill 不能复制 MJLab、环境、Robot Profile 公共资产或第三方 runtime；当前独有 ONNX/动作样例约为 `0.84-11.24 MiB`，普通 Git 足够。未来出现真正大型独有产物再启用外置下载。

## 第三轮结果：首个可运行版本

MVP 验收标准单独维护在 `MVP_ACCEPTANCE.md`。

### Q14. 首个 Robot Profile

结果：接受现有 `Unitree G1 29DoF` 作为第一个完整 simulation-only Profile，在接口稳定后验证 Go2/其他公开模型。

### Q15. WebUI 语言与风格

结果：中文默认，英文保留在 task/field/API 标识。桌面优先、简洁清晰、配色克制，不使用渐变、发光、玻璃拟态或明显“AI 风格”颜色。

### Q16. 模型与仿真查看器

结果：复用 MJLab 已有 Viser/MuJoCo viewer。平台只负责启动、嵌入/跳转、状态和结果，避免重复开发。

### Q17. Skill 操作确认策略

结果：接受分级确认规则。只读检查和普通仿真直接运行；创建 Conda 环境、安装依赖、修改 Profile 需要确认；physical deployment 和标定写入单独确认。

### Q18. 首版页面

结果：接受首版导航。`Train / Validate / Deploy / Agent` 暂作为详情入口或禁用功能，不制造空页面。

### Q19. 训练功能进入 MVP 的程度

结果：接受。MVP 优先 play/inspect，只发现并展示现有 task、配置与等价 CLI；训练启动、指标和 resume 放下一里程碑。

### Q20. 本地数据目录

结果：接受。`var/` 全部 gitignore；全局用户偏好才进入用户配置目录。

## 第四轮结果：仓库管理与开工顺序

架构 QA 已完成，以下选择形成实施基线。本文是历史决策记录；当前批次和完成状态以
[`docs/project/DEVELOPMENT_PLAN.md`](../project/DEVELOPMENT_PLAN.md) 为准。

### Q21. Unitree 上游来源的 Git 管理方式

决策时本地存在一份约 340 MiB、完整匹配上游 `1425b15f` 但未被 RoboLab Git 跟踪的工作副本。最初讨论了完整 squash vendor；进一步评估品牌独立性、仓库体积和目录所有权后，最终修订为精选 vendor。

结果：选择精选 vendor 导入。目标路径为 `vendor/unitree_rl_mjlab/`，保留实际依赖的训练、机器人资产、脚本、仿真和 legacy deploy 技术源码；排除上游 README/doc/GIF、预编译 ONNX Runtime/MuJoCo 库以及应迁入 RoboLab-Skill 的策略/动作产物。完整 include/exclude 清单写入 `VENDOR_MANIFEST.yaml`。

实施状态：已完成（2026-08-20）。导入为独立 commit，外部 ONNX Runtime/MuJoCo 发现以单独的 RoboLab patch commit 跟进。

### Q22. 第一段代码的实现顺序

结果：接受先实现 schemas、Skill 扫描/安装、`robolab-job-v1`、CLI 和三个样板，再开始 WebUI。

### Q23. G1 Velocity Skill 的文件边界

结果：接受把 ONNX 和 deploy 参数放入 G1 Velocity Skill，只引用 G1 Profile/MJLab task，不复制 XML、mesh 或 runtime。

### Q24. 本地进程模型

结果：接受 `robolab serve` 统一管理 API、静态 WebUI 和本地 Worker，调试时允许拆分。

### Q25. 默认 Skill catalog 来源

结果：接受开发环境优先自动发现同级 RoboLab-Skill，不存在时使用配置的 GitHub URL，不把绝对路径写入发布配置。

### Q26. MVP 测试范围

结果：接受普通 CPU CI 覆盖 schema、安装、Job 协议和 MJCF Inspector；MJLab play/Viser 作为本地环境测试；实机测试暂不进入 CI。

## 架构 QA 完成状态

Q1-Q26 已全部完成并固化。下一阶段不再继续扩大设计范围；实现进度与剩余工作按
[`docs/project/DEVELOPMENT_PLAN.md`](../project/DEVELOPMENT_PLAN.md) 跟踪，验收标准见
[`docs/project/MVP_ACCEPTANCE.md`](../project/MVP_ACCEPTANCE.md)。

## 2026-08-21 主线复核与下一阶段决策

| ID | 决策 | 状态 |
|---|---|---|
| D-033 | 当前架构没有战略跑偏，但实现存在阶段性“平台化偏移”；下一阶段停止增加横向平台功能，以真实 G1 MotionSkill 的 play、validation、sim-to-sim、deployment 与 `stop/safe` 纵向闭环为唯一主线 | 已确认 |
| D-034 | `Heavily Customized MJLab` 不以大规模修改 vendor 为目标，而以统一 task/train/play/export/evaluate 契约、结构化指标、Robot Profile 适配、Artifact lineage 与独立 Runtime/Deployment Gate 作为达成标准 | 已确认 |
| D-035 | B1–B6 执行计划归档；B7 作为下一阶段 N0 关闭门禁继续执行，N0 未用真实 trained G1 policy 通过前，不进入 N1–N5 实现 | 已确认 |
| D-036 | 当前 GPU 被其他 IsaacLab 训练占用；RoboLab 不抢占、不终止、不并行启动 G1 GPU 任务。现有训练结束后先训练或取得可复现 G1 checkpoint | 已确认 |

详细判断见
[`docs/project/MAINLINE_ALIGNMENT_REVIEW.md`](../project/MAINLINE_ALIGNMENT_REVIEW.md)，
执行批次见 [`docs/project/DEVELOPMENT_PLAN.md`](../project/DEVELOPMENT_PLAN.md)。
