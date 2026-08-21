# RoboLab Skill Package 规范

状态：`v1alpha1` 已冻结（2026-08-20，B1）；已确认支持可执行 Skill 与 Agent Skill。机器可执行 JSON Schema 在 `packages/schemas/src/robolab_schemas/data/skill_package.v1alpha1.schema.json`，`robolab check` 对 manifest 做结构 + lint 校验。schema 与本文冲突时以 schema 为准，修改必须两边同步。

## 1. 统一定义

RoboLab Skill 是一个可下载、可安装、可版本化、可调用的功能包。Skill 可以提供机器人运动能力、平台工具或 Agent 工作流，但都使用同一个 `skill.yaml` 描述身份、入口、依赖、权限、兼容性和文档。

Robot Profile 负责“机器人是什么以及怎样接入”，Skill 负责“平台或机器人能做什么”。Skill 不应复制整套机器人驱动；需要硬件能力时声明对 Robot Profile capability 的依赖。

## 2. Skill 类型

MVP 支持三类 Skill：

| kind | 典型内容 | 执行方式 |
|---|---|---|
| `MotionSkill` | ONNX、参考动作、部署参数、训练 recipe | MJLab/Edge motion runtime |
| `PlatformSkill` | 模型检查、数据转换、标定辅助、评测或其他 Python/CLI/C++ 功能 | Worker 独立进程 |
| `AgentSkill` | `SKILL.md`、工作流、提示词、平台工具声明 | Agent runtime；工具仍经平台 API/Job 调用 |

后续可以增加 `CompositeSkill` 来编排多个 Skill。一个包原则上只声明一个主 `kind`；AgentSkill 如果需要自定义可执行工具，应将工具作为同仓库的 PlatformSkill 依赖，而不是让 Agent 绕过平台直接运行任意 shell。

## 3. 源仓库与本地安装目录

公开的 [`RoboLab-Skill`](https://github.com/Lain-Ego0/RoboLab-Skill) 作为官方 catalog monorepo：

```text
RoboLab-Skill/
├── catalog.yaml
├── skills/
│   ├── motion/g1_velocity/
│   ├── platform/mjcf_inspector/
│   └── agent/robot_onboarding/
└── tools/                    # catalog 校验和打包工具
```

RoboLab 主仓库中的运行布局：

```text
RoboLab/skills/
├── builtin/                  # 随平台发布并受平台版本控制
├── installed/                # 从 catalog 安装的不可变版本；默认 gitignore
└── dev/                      # 本地开发链接/checkout，允许热更新

RoboLab/var/
├── skill-envs/               # 可选的 Skill Conda 环境
├── skill-cache/              # 下载与内容寻址缓存
└── runs/                     # Job 输入、事件、日志和产物
```

正式安装执行“解析 catalog -> 下载固定 revision -> 校验 -> 复制到 `skills/installed/<id>/<version>/<hash>` -> 注册”。开发模式可以链接 `skills/dev/<id>`，但 UI 必须清楚标记为可变、未固定版本。

## 4. 通用包结构

```text
my_skill/
├── skill.yaml                # 必需：机器可读 manifest
├── README.md                 # 必需：用户文档、示例和限制
├── LICENSE                   # 必需：代码和默认资源许可
├── environment.yml           # 可选：独立 Conda 依赖
├── SKILL.md                  # AgentSkill 必需；根目录 Agent 指令
├── src/                      # 可选：Python/C++/脚本源码
├── scripts/                  # Agent 可复用脚本或 PlatformSkill 辅助入口
├── references/               # Agent 渐进读取的参考文档
├── assets/                   # Agent 模板/资源；不要复制机器人公共资产
├── artifacts/                # ONNX 等不可变产物
├── motions/                  # 动作数据
├── params/                   # 部署/训练配置
├── schemas/                  # 输入、输出和 UI schema
└── tests/                    # smoke/contract/simulation 测试
```

所有 Skill 全部开源。每个包仍需带独立许可证，因为代码、模型、动作数据和 mesh 可能来自不同来源；附加资源的不同许可证应在 README 或 `NOTICE` 中逐项说明。

## 5. 通用 Manifest

```yaml
apiVersion: robolab.dev/v1alpha1
kind: PlatformSkill

metadata:
  id: tools.mjcf-inspector
  name: MJCF Inspector
  version: 0.1.0
  description: Inspect a public MJCF model and generate a diagnostic report.
  license: MIT
  source:
    repository: https://github.com/Lain-Ego0/RoboLab-Skill
    revision: 0123456789abcdef0123456789abcdef01234567
  # 可选：内容上游出处（复用 vendor/第三方产物时必填），revision 为完整 commit SHA
  provenance:
    repository: https://github.com/unitreerobotics/unitree_rl_mjlab
    revision: 1425b15f73bd4095f0df53709d7c389c3eb9e790
    paths:
      - deploy/robots/g1/config/policy/velocity/v0/...

spec:
  compatibility:
    platform: ">=0.1.0 <0.2.0"
    skillApi: v1alpha1

  runtime:
    type: python
    protocol: robolab-job-v1
    environment:
      mode: inherit            # inherit 或 conda
      file: environment.yml    # mode=conda 时使用
    entrypoint:
      module: robolab_mjcf_inspector.main

  actions:
    inspect:
      title: Inspect MJCF
      inputSchema: schemas/inspect.input.json
      outputSchema: schemas/inspect.output.json
      ui: form

  permissions:
    filesystem:
      read: [workspace, skill]
      write: [run]
    network: false
    subprocess: false
    robotState: false
    robotCommand: false

  validation:
    smoke: tests/smoke.yaml
```

示例 revision 是占位符，发布版本不允许引用浮动 `main`、`latest` 或无哈希下载。

## 6. 可执行 Skill 协议

可执行 Skill 不导入 FastAPI/API 主进程，而由 Worker 以子进程启动。即使使用同一个 `robolab` Conda 环境，也保持进程隔离。

`robolab-job-v1` 的最小约定：

1. 平台创建只属于本次运行的 `run_dir`；
2. 将 action、已验证参数、允许的路径和平台版本写入 `input.json`；
3. 通过环境变量传入 `ROBOLAB_RUN_DIR` 和 `ROBOLAB_INPUT_FILE`；
4. Skill 将结构化事件逐行写入 `events.jsonl`，普通 stdout/stderr 作为日志；
5. 产物只能写入 `run_dir/artifacts/`，完成时写 `result.json`；
6. 取消时先发送终止信号，超时后强制结束整个进程组；
7. 退出码、result schema 和 artifact hash 决定 Job 成败。

首版支持 Python module 和普通 command。长期运行的 service Skill、浏览器插件和进程内 Python hook 暂不进入 MVP。

## 7. Conda 依赖策略

- RoboLab 本身使用一个主 Conda 环境，例如 `robolab`。
- 简单或官方 Skill 可声明 `environment.mode: inherit`，直接使用主环境依赖。
- 依赖冲突明显的 Skill 可附 `environment.yml`，由平台创建内容哈希命名的独立环境。
- 安装 Skill 与创建环境分开显示；下载完成不等于环境可运行。
- 环境创建、更新和删除都要可取消、可重试，并保存 `conda env export` 快照。
- MVP 不使用 Docker；manifest 保留未来增加 container runtime 的扩展点。

## 8. MotionSkill 扩展字段

### 8.1 资源复用与大文件策略

Skill 不复制 RoboLab/MJLab 本体、Conda 环境、ONNX Runtime、MuJoCo、机器人公共 XML/MJCF/mesh 或厂商 SDK。它通过 platform/task/Robot Profile 版本声明依赖，只携带该 Skill 独有的内容：

- policy ONNX；
- 参考动作；
- 训练/部署参数；
- 自身源码、schema、文档和测试；
- 少量演示资源。

当前仓库样例中，G1 velocity ONNX 约 `0.84 MiB`，mimic ONNX 外部数据约 `0.94 MiB`，dance motion NPZ 约 `11.24 MiB`，暂时都可以直接使用普通 Git。因此 MVP 不引入 Git LFS/GitHub Release 下载器。

后续只有 Skill 独有的二进制频繁更新、单文件明显变大或 catalog clone 体积成为问题时，才按包选择 Git LFS/Release，并在 manifest 固定 URL、size、SHA-256 和许可证。训练 checkpoint、完整数据集和公共机器人资产默认不进入运行时 Skill 包。

```yaml
kind: MotionSkill
spec:
  runtime:
    type: onnx
    runner: rl_policy
    protocol: robolab-motion-v1   # 与可执行 Skill 的 robolab-job-v1 区分
  compatibility:
    robots:
      - profile: unitree.g1.29dof
        version: ">=1.0.0 <2.0.0"
    controlMode: joint_position
    controlHz: 50
    jointSet: g1.29dof.canonical.v1
    # 观测/动作 schema 使用稳定 id（B2 任务绑定导出后可机器比对），也接受 sha256:<hex>
    observationSchema: unitree.g1.velocity.v0.observation.v1
    actionSchema: unitree.g1.29dof.joint-position.v1
  artifacts:
    - name: policy
      path: artifacts/policy.onnx
      mediaType: application/onnx
      size: 878421                # 可选；声明后 lint 实测核对
      sha256: POLICY_FILE_HASH
    - name: deploy-params
      path: params/deploy.yaml
      mediaType: application/yaml
      sha256: PARAM_FILE_HASH
  actions:
    play:
      title: Play in MJLab
      inputSchema: schemas/command.json
      task: Unitree-G1-Flat
    deploy:
      title: Prepare real-robot deployment
      inputSchema: schemas/command.json
  safety:
    maturity: experimental        # experimental | validated | deprecated
    defaultTarget: simulation
    requiredGates: [offline, mjlab_play, sim_to_sim]
    fallbackState: damping
    realRobotRequiresExplicitConfirmation: true
```

MotionSkill 的兼容不能只写 `robots: [g1]`。必须同时校验 profile/version、joint set、观测/动作 schema、控制模式、频率、状态估计和安全限制。`robolab check --skill S --profile P` 输出逐条可解释原因（B1.4）。

## 9. AgentSkill 扩展字段

```yaml
kind: AgentSkill
spec:
  runtime:
    type: agent
    instructions: SKILL.md
  actions:
    assist:
      title: Robot Onboarding Assistant
      inputSchema: schemas/assist.input.json
  tools:
    allow:
      - robolab.robots.inspect
      - robolab.jobs.create
      - robolab.artifacts.read
    deny:
      - robolab.deployments.activate_real
  permissions:
    filesystem:
      read: [workspace, skill]
      write: [run]
    network: false
    subprocess: false
    robotState: false
    robotCommand: false
```

AgentSkill 是给开发者 Agent 使用的操作说明与工具白名单。它不能因为自然语言中写了某条命令就获得权限；真正操作仍通过平台工具层进行 schema 校验、审计和用户确认。

根目录 `SKILL.md` 使用易读的 YAML frontmatter，最少提供 `name` 和 `description`。RoboLab 自身的版本、权限、action 和依赖仍以 `skill.yaml` 为准：

```markdown
---
name: robot-onboarding
description: Inspect a robot model and guide simulation-only onboarding.
---

# Robot onboarding

Follow the validated RoboLab onboarding workflow...
```

这种布局与 [Codex Skills 官方文档](https://developers.openai.com/codex/skills) 描述的核心目录结构兼容：Codex 使用根目录 `SKILL.md`，并可按需读取 `scripts/`、`references/` 和 `assets/`。Codex 仓库级发现使用 `.agents/skills/`，因此 RoboLab 后续提供 export/sync 命令，而不是要求 Codex 直接理解 `skills/installed`。

Claude、DeepSeek 等 Agent 的发现路径、附加元数据和工具协议可能不同。RoboLab 只定义公共核心包，再由薄适配器导出到各 Agent 的当前格式；不宣称存在跨厂商统一标准。

## 10. 安装与调用生命周期

```text
discover -> resolve revision -> fetch -> verify license/hash
         -> review permissions -> install -> prepare conda env
         -> contract test -> register actions -> invoke
```

- 同一 `id@version` 如果内容哈希不同必须拒绝，发布者应提升版本。
- 可执行 Skill 首次运行前显示权限与入口命令；安装不会自动执行 `setup.sh`。
- Skill 更新产生新版本并重新进行 contract test，不静默覆盖正在使用的版本。
- 卸载不删除仍被历史 Job、Artifact 或 DeploymentSession 引用的缓存。
- WebUI、CLI 和 Agent 最终都调用同一个 action registry，避免三套执行路径。

## 11. 与现有 MJLab 的映射

- `deploy/robots/*/config/policy/velocity/...` -> `MotionSkill`；
- `deploy/robots/g1*/config/policy/mimic/...` -> `MotionSkill` 的 policy + motion + params；
- `src/tasks/*/config/*` -> task binding 或 Training 类型 PlatformSkill；
- `scripts/csv_to_npz.py` -> 可迁移为动作转换 PlatformSkill；
- 模型诊断、标定向导、日志分析 -> PlatformSkill；
- 机器人接入指导、训练参数建议、故障排查流程 -> AgentSkill；
- C++ `State_RLBase` / `State_Mimic` -> 平台共享 motion runner，不属于单个 Skill。

## 12. 发布检查清单

- 稳定 `metadata.id`、SemVer、README 和 LICENSE；
- manifest、action input/output schema 与文档一致；
- 源码、模型、动作、mesh 和示例数据均可公开分发；
- artifact 有 SHA-256，不引用浮动 revision（`main`/分支名被 lint 拒绝；tag 可用但会收到“建议固定 commit SHA”警告）；
- 可执行入口在独立进程中正常启动、取消和清理；
- 权限为最小集合，未声明的 filesystem/network/device 操作应失败；
- MotionSkill 明确机器人、关节、observation/action 和安全门禁；
- AgentSkill 明确允许工具，默认不能直接激活实机；
- 至少提供 smoke test，并记录已验证的 RoboLab/Conda 版本。
