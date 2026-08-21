# RoboLab MVP 验收标准

状态：v0.1；三类 Skill 样板和本地运行方式已确认。

## 1. MVP 目标

MVP 要证明的不是完整实机平台，而是下面这条本地闭环成立：

```text
Conda 启动 RoboLab -> 发现 catalog -> 安装 Skill -> 校验/准备
                    -> WebUI/CLI 调用 -> Job 运行 -> 查看结果
                    -> MJLab 仿真验证 -> 保存版本与产物
```

运行时不依赖远程服务器、账号系统、Docker 或云端服务。网络断开后，除首次下载依赖/源码外，已安装功能仍可使用。

## 2. 本地启动验收

目标用户流程：

```bash
conda activate robolab
robolab serve
```

验收条件：

- FastAPI 仅监听 loopback，默认不暴露到局域网；
- 自动选择或提示可用端口，并输出 WebUI URL；
- FastAPI 提供编译后的 React 静态资源，普通运行不要求 Node；
- WebUI 遵守 `UI_GUIDELINES.md`，不使用渐变、发光或玻璃拟态；
- SQLite、Skill、Job 和 Artifact 默认写入仓库可配置的本地数据目录；
- 启动页显示 Python、Conda、MuJoCo、MJLab、GPU 和磁盘健康状态；
- 关闭服务不终止由独立 Worker/Edge 安全接管的任务。

## 3. 样板 A：G1 Velocity MotionSkill

用途：验证模型型 Skill 从 catalog 到 MJLab play 的完整链路。

来源：复用现有 G1 velocity ONNX 和 deploy 参数，不复制 MJLab、G1 XML/mesh 或 ONNX Runtime。

必须通过：

1. `skill.yaml` schema、许可证和 artifact SHA-256 校验；
2. 识别 `unitree.g1.29dof` Robot Profile 和 `Unitree-G1-Flat` task；
3. 校验 policy 输入输出、joint set、control frequency 和 deploy 参数；
4. 在 WebUI 显示 `play` action 及其速度命令表单；
5. 创建独立 Job 调用 MJLab play；
6. 实时显示日志、阶段、运行时间和停止按钮；
7. 生成运行配置、结果摘要，条件允许时保存视频；
8. 同一个 Skill 版本重复运行不发生隐式文件覆盖。

首版不要求真实 G1，也不要求 physical deployment。

## 4. 样板 B：MJCF Inspector PlatformSkill

用途：验证可执行 Python Skill、输入 schema、权限和结构化产物。

输入：一个 workspace 内的 XML/MJCF 主文件及可选资产根目录。

至少检查：

- XML include 和 mesh/textures 是否可解析；
- body、joint、actuator、sensor、site、geom、keyframe 数量；
- joint name 唯一性、range、limited、axis 和默认位置；
- actuator 与 joint 绑定、控制范围和缺失执行器；
- free joint/root body、IMU 候选 frame、足端/末端候选；
- 质量、惯量、碰撞体和明显的空资产/路径问题；
- MuJoCo 实际加载错误与 warning；
- 生成 canonical joint mapping 草稿所需的候选信息。

必须产出：

- `report.json`：稳定 schema，供平台/Agent 使用；
- `report.md`：开发者可读报告；
- `robot_profile.draft.yaml`：明确标记为草稿，不能直接实机部署；
- 可选模型截图或 viewer session 信息。

模型交互首版复用 MJLab/Viser/MuJoCo viewer，Inspector 不实现自有 Three.js 渲染器。

权限默认是 workspace 只读、run directory 可写、无网络、无机器人命令。

## 5. 样板 C：Robot Onboarding AgentSkill

用途：验证外部开发 Agent 能发现并正确使用 RoboLab 的领域流程。

包内至少包含：

- 根目录 `SKILL.md`，带 `name` 和 `description`；
- `skill.yaml`，声明只读/仿真工具白名单；
- `references/`：Robot Profile、MJCF 诊断和成熟度等级说明；
- 一个 Codex export 测试，将内容发布到 `.agents/skills/robot-onboarding/`。

Agent 应能指导开发者：

1. 选择或导入 MJCF；
2. 调用 MJCF Inspector；
3. 阅读报告并提出需要人工确认的 mapping；
4. 生成 simulation-only Profile 草稿；
5. 运行 L0/L1 检查；
6. 明确说明缺少 physical capability，不能声称已支持实机。

AgentSkill 不要求首版 WebUI 内置聊天，也不能直接调用 physical deployment。

## 6. Skill 系统通用验收

- 能扫描 `builtin/`、`installed/`、`dev/`，并标出来源和可变性；
- 能从本地 RoboLab-Skill checkout 安装单个 Skill；
- 同一 `id@version` 内容不同会被拒绝；
- 可显示 README、LICENSE、权限、Conda 模式、actions 和测试状态；
- PlatformSkill 在独立进程组中运行，可以取消并清理；
- stdout/stderr、`events.jsonl`、`result.json` 和 artifacts 均归档到单次 run；
- schema 错误、依赖缺失、进程失败和用户取消有不同的可解释状态；
- WebUI、CLI、外部 Agent 调用同一个 action registry；
- 卸载不会删除历史 Job 正在引用的内容哈希。

## 7. 非 MVP 范围

- 实机电机控制和自制机器人完整 Driver；
- 自动推断完整奖励函数、域随机化和训练超参数；
- 平台内置通用聊天 Agent；
- Claude/DeepSeek 的正式 adapter；
- Docker、服务器、多用户和远程 Worker；
- CompositeSkill、在线商店、签名和自动发布流水线。

## 8. MVP 完成定义

在一台本地 Conda 工作站上，用户无需手工复制策略到 MJLab 部署目录，就能通过 WebUI 安装并运行 G1 Velocity；能用 MJCF Inspector 分析任意合规的公开模型；能把 Robot Onboarding AgentSkill 导出给 Codex。三个路径都产生可复现 Job、结构化结果、明确错误和版本记录，即视为 MVP 纵向闭环完成。
