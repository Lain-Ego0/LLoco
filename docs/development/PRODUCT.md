# 产品功能与 WebUI 设计

## 1. 产品定位

RoboLab 是面向成品与自研机器人的运动控制工程工作台，不只是训练脚本的按钮封装。WebUI 的价值在于统一机器人、任务、
策略、Skill、状态、验证证据和安全门禁，同时仍允许高级用户查看并复制等价 CLI 命令。

首版仅服务当前工作站上的个人用户，默认绑定 `127.0.0.1`。不设计账号、团队空间、远程服务器和多租户权限；仍保留本地运行记录，便于复现与排错。

## 2. 信息架构

### Workspace

- 环境健康检查：GPU、MuJoCo、依赖、磁盘、DDS/SDK；
- 最近 Job、验证失败、活跃部署和告警；
- 当前代码 revision 与上游偏离状态。

### Robots

- Robot Profile 和 Robot Instance 列表；
- 模型预览、关节映射、控制参数、Driver 状态；
- 适配向导、成熟度等级和兼容 Skill 矩阵；默认接入流程必须支持 MJCF-only、自研和比赛机器人，不要求先选择厂商。

### Skills

- catalog 源管理、本地安装、版本与许可证；
- 区分 Motion、Platform、Agent 三类 Skill，展示入口、Conda 环境和权限；
- 通过 schema 自动生成 action 表单，并可从 WebUI、CLI 或 Agent 调用同一 action；
- compatibility reason：明确显示为什么兼容/不兼容；
- artifact 哈希、训练来源、验证报告、安装/回滚；
- 只对满足门禁的 Robot Instance 显示“准备部署”。

### Agent（外部开发 Agent 优先）

- 管理 AgentSkill 的 `SKILL.md` 与允许工具，并导出给外部 Codex/Claude/DeepSeek 等开发 Agent；
- 让 Agent 创建 Job、分析报告和指导机器人接入；
- Agent 的每个工具调用都显示真实 action、参数、权限和结果；
- Agent 默认不能直接发送电机命令或激活实机部署。

首版先建设 registry、CLI/工具协议与 Agent adapter，不强制内置聊天。平台内 Agent 页面是后续优先增强，并复用同一 action registry。

### Train

- 选择 Robot Profile + Customized MJLab 1.6 TaskDefinition + TrainingRecipe；
- schema 驱动表单与 raw config 双视图；
- 资源选择、seed、resume、视频和实验标签；
- 提交前显示最终命令、配置 diff 和预计输出目录。

### Jobs & Artifacts

- 队列、日志、指标曲线、GPU/CPU 状态、取消和重试；
- checkpoint、ONNX、deploy config、视频、验证报告；
- 产物 lineage：从哪个代码、配置、数据和 Job 产生。

### Validate

- 离线 schema/shape 检查；
- MJLab play；
- sim-to-sim；
- 场景指标、视频、失败原因和批准记录。

### Deploy

- 分离 simulation 和 physical session；
- 选择固定 Skill/Robot/Artifact hash；
- 安全 checklist、本地 Edge 二次确认、启动/停止；
- 低频实时遥测、状态机、告警和事件时间线；
- 一键进入安全状态应调用 Edge，而不是依赖浏览器持续连接。

### Audit & Settings

- 操作者、时间、配置修改、批准和部署历史；
- catalog、artifact storage、worker、Edge 和 secrets；
- 第三方许可证与版本信息。

## 3. 当前主线产品边界

当前 R0–R6 必须形成一个完整纵向闭环：

1. 发现或创建 Customized MJLab 1.6 tasks；
2. 从公开或自研 MJCF 创建一个自研、爱好者或社区可复现的 simulation-only Robot Profile；
3. 在 Customized MJLab 1.6 中完成任务训练或可信策略回放、评测和导出；
4. 从 `RoboLab-Skill` 安装一个 MotionSkill 和一个可执行 PlatformSkill；
5. 准备/复用 Conda 环境，审查权限并通过 contract test；
6. 从 UI 启动 Skill action、train/play/evaluate/export 和验证，读取结构化事件与日志；
7. 保存 Artifact、配置快照与验证记录；
8. 启动该 Robot Profile 对应的 simulator/sim-to-sim target，并提供“一键仿真部署”；
9. Runtime 在 stop、heartbeat loss 和 Driver 断开时进入 SAFE；
10. 所有步骤可从本地记录复现。

实际硬件适配完成前，physical target 保持不可激活。R6 之后再考虑平台内置 Agent 页面、CompositeSkill、签名、
更强隔离、Docker、远程 Worker 和多人模式。详细批次只以
[`../plans/README.md`](../plans/README.md) 为准。

## 4. 关键 UX 原则

- **显示真实状态**：区分 installed、compatible、sim-validated、hardware-approved、active。
- **失败可解释**：兼容性失败必须定位到版本、关节、schema、频率或安全参数。
- **保留 CLI 逃生舱**：每个 Job 可复制等价命令；平台元数据不能绑架现有开发流程。
- **下载即可发现，不等于自动执行**：安装后 action 立即可见，第一次运行前明确展示依赖与权限。
- **危险操作有上下文**：实机按钮显示机器人、策略哈希、网卡、状态和门禁，不使用含糊的“Run”。
- **配置不可静默变化**：提交后保存 snapshot；再次运行产生新 revision。
- **日志不是遥测**：日志用于诊断，结构化事件和指标用于 UI 与门禁。
- **视觉服务于信息**：禁止渐变、霓虹发光、玻璃拟态、装饰性网格和大面积 AI 紫蓝色；优先表格、清晰边框、留白和稳定状态色。
- **避免重复造 viewer**：模型与仿真首版复用 MJLab/Viser/MuJoCo viewer，WebUI 管理 session 与结果。

## 5. API 轮廓

初期可围绕资源设计：

```text
GET/POST  /api/robots
GET/POST  /api/robot-instances
GET/POST  /api/catalogs
GET/POST  /api/skills
POST      /api/skills/{id}/install
POST      /api/skills/{id}/prepare
POST      /api/skills/{id}/actions/{action}:invoke
POST      /api/compatibility/check
GET/POST  /api/jobs
POST      /api/jobs/{id}/cancel
GET       /api/artifacts/{id}
POST      /api/validations
POST      /api/deployments
POST      /api/deployments/{id}/arm
POST      /api/deployments/{id}/stop
WS        /api/events
```

`arm` 与 `start` 应是不同动作：WebUI 只能发起 arm 请求，实机 Edge 需要本地条件/确认后才进入 active。

## 6. 安全与权限

- 即使 MVP 是单用户本地版，也记录操作者和时间。
- real deployment、修改 Safety Profile、接受未知许可证是高风险动作。
- 可执行 Skill 按 manifest 获得最小权限；默认不能获得网络、任意 workspace 写入、设备或机器人命令。
- API 对路径做 workspace 限制，不能接受任意文件系统路径作为 Skill 输入。
- 训练、转换与可执行 Skill 均与 API 分进程，并设置资源、超时、进程组取消和输出目录。
- Edge 必须在命令超时后自动撤销 active session。
