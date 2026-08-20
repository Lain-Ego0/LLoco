# Agent Skill 与外部 Agent 集成

## 1. 定位

RoboLab 不在首版重新训练或实现一个基础 Agent。Codex、Claude、DeepSeek 等成熟开发 Agent 负责理解任务与规划；RoboLab 提供机器人领域 Skill、稳定工具、结构化结果和安全边界。

```text
External Developer Agent
        │ reads exported SKILL.md
        │ invokes approved CLI/tool actions
        ▼
RoboLab Agent Adapter -> Action Registry -> Job/Robot/Artifact services
```

首版以外部开发 Agent 为主。平台内置聊天 Agent 很有价值，但应在 action schema、Job 协议和权限模型稳定后复用这些接口，而不是单独实现另一条执行路径。

## 2. 公共 AgentSkill 包

```text
robot_onboarding/
├── skill.yaml                # RoboLab 版本、权限、工具与兼容性
├── SKILL.md                  # Agent 可读指令，根目录
├── README.md
├── LICENSE
├── scripts/                  # 可选辅助脚本
├── references/               # 可选详细资料
├── assets/                   # 可选模板/资源
├── schemas/
└── tests/
```

`SKILL.md` 是跨 Agent 的人类可读公共核心，`skill.yaml` 是 RoboLab 的机器可读控制面。不同 Agent 的发现目录、附加元数据和工具连接方式由 adapter 处理。

## 3. Codex 适配

[OpenAI 官方 Codex Skills 文档](https://developers.openai.com/codex/skills)说明：

- Skill 是包含根目录 `SKILL.md` 的目录；
- `SKILL.md` 至少包含 `name` 和 `description`；
- Skill 可以包含 `scripts/`、`references/`、`assets/`；
- Codex 会扫描仓库路径上的 `.agents/skills/`；
- 可选的 `agents/openai.yaml` 可以提供界面信息、调用策略和工具依赖。

因此 RoboLab 不要求 Codex 原生扫描 `RoboLab/skills/installed/`，而是提供类似以下的导出动作：

```text
robolab agent export robot-onboarding --target codex
```

目标行为：

1. 解析并校验 AgentSkill；
2. 将兼容内容复制/链接到 `.agents/skills/robot-onboarding/`；
3. 保留 `SKILL.md`、scripts、references 和 assets；
4. 从 `skill.yaml` 生成可选的 `agents/openai.yaml`；
5. 将 RoboLab action 暴露为稳定 CLI/JSON 工具；
6. 记录导出来源、Skill 版本和内容哈希。

Codex 能否调用某个工具仍取决于实际运行环境和权限。`SKILL.md` 中写出命令不会自动授予机器人控制权限。

## 4. 其他 Agent 适配

Claude、DeepSeek 或其他 Agent 通过各自 adapter 使用同一公共包：

- 如果支持目录 Skill，映射 `SKILL.md` 和资源目录；
- 如果只接受项目说明，则生成对应的说明文件；
- 如果支持工具协议，则连接 RoboLab Action Registry；
- 如果只支持 shell，则提供稳定的 `robolab ... --json` CLI；
- vendor-specific 文件放在 `agents/<vendor>/` 或安装时生成，不污染公共指令。

在实现并核对各厂商当前文档前，不承诺同一个目录可以被所有 Agent 自动发现。兼容的目标是“同一语义来源可导出”，不是“所有 Agent 使用完全相同的内部格式”。

## 5. 工具接入阶段

### Stage A：读取与指导

Agent 读取 `SKILL.md`、references 和平台生成的报告，给开发者操作建议。它可以调用只读 CLI，例如模型检查、任务列表和 Artifact 查询。

### Stage B：受控工具调用

RoboLab 暴露 schema 化 action：创建仿真 Job、生成 Profile 草稿、运行检查和读取结果。可以增加 MCP 等 adapter，但 Action Registry 是唯一业务入口。

### Stage C：平台内置 Agent

WebUI 增加对话、Skill 选择、工具调用确认和结果展示。模型 provider 可替换，本地/远程模型策略以后再决定；内置 Agent 仍不能绕过 action 权限或 Deployment Gate。

## 6. Agent 安全边界

- 默认允许读取文档、模型报告和 Job 状态；
- 修改 Profile、启动耗时训练和安装依赖需要显示 action 参数；
- physical deployment、标定写入和电机命令始终需要专门 capability；
- AgentSkill 本身不能获得 manifest 未声明的网络、文件或设备权限；
- 每次工具调用保存 Skill、Agent adapter、参数、结果和时间；
- 外部 Agent 生成的 shell 文本不等于平台已批准的命令。

## 7. 首版交付边界

首版实现：AgentSkill 安装与校验、根目录 `SKILL.md`、Codex export 原型、稳定只读/仿真 CLI action。Claude/DeepSeek adapter 和 WebUI 内置 Agent 在核心 action 协议稳定后逐步增加。
