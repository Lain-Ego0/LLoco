# Lain's LocoLab 项目进度与验收表

> 更新时间：2026-09-03  
> 进度依据：当前源码、`GO2_MIGRATION_PLAN.md`、本地 validation 日志和实际验收命令

本文档同时追踪三个不同维度：

1. **迁移完整性**：源任务、算法和接口是否已经接入。
2. **策略成熟度**：训练结果是否已经达到可展示、sim-to-sim 或实机质量。
3. **新架构实施度**：LainLoco 的机器人优先架构是否已经落地。

三者不得互相替代。

## 1. 状态定义

| 状态 | 含义 |
|---|---|
| ✅ 已验收 | 当前实现已满足本表列出的验收条件，并有命令或产物证据 |
| 🟡 进行中 | 主要调用链存在，但仍有明确的行为、结构或部署验收未完成 |
| ⬜ 未开始 | 尚未形成可验收实现 |
| ⏸ 暂缓 | 有意延后，不阻塞当前阶段 |
| ➖ 范围外 | 不属于当前迁移完成条件 |

## 2. 总览

| 维度 | 状态 | 当前结论 | 下一验收点 |
|---|---|---|---|
| Go2 资产迁移 | ✅ | MJCF、12 actuator 和关键实体可解析 | 新包迁移后重复资产检查 |
| 14 个源任务入口 | ✅ | 14/14 可构建、reset、step | 新 Catalog 与旧 ID 等价 |
| 两个公共 Velocity 基线 | ✅ | Flat/Rough 已注册 | 迁移后保持兼容 |
| 标准 PPO 技能调用链 | ✅ | 六个技能可训练并保存 checkpoint | 最新实现的行为质量验收 |
| 自定义算法调用链 | ✅ | CTS、DreamWaQ、AMP、TS 和 student 可更新 | 架构拆分后回归 |
| ONNX/策略接口 | ✅ | v1 契约和多类 wrapper 已完成前向验证 | Policy Bundle 化 |
| 策略效果成熟度 | 🟡 | 历史 1024×1000 结果均有 reward 改善 | 按最新语义重新选择性长训 |
| sim-to-sim 完整闭环 | 🟡 | Python 部署 adapter 已有 | 独立 runtime 连续控制验收 |
| 真实 Go2 硬件闭环 | ➖ | 不属于当前迁移范围 | 独立安全和实机计划 |
| 架构设计文档 | ✅ | 目标边界、依赖和阶段已固化 | 用 ADR 管理后续变更 |
| LainLoco 独立包 | ⬜ | 当前代码仍在 `mjlab` 包内 | `import lainloco` 与 entry point |
| 机器人优先目录 | ⬜ | 当前仍集中于 `tasks/velocity` | Go2 tasks 分域完成 |
| Task/Profile 分离 | ⬜ | 当前算法仍编码在 Task ID | 新 Catalog 与兼容 alias |
| Policy Bundle | ⬜ | 当前以 ONNX 和元数据组件为主 | bundle 导出与拒绝错误契约 |
| 品牌与仓库改名 | 🟡 | README 已使用 Lain's LocoLab | 仓库、包、CLI 正式改名 |
| 项目级许可证与声明 | ⬜ | 仅有 mjlab 组件许可证 | 项目 LICENSE 与第三方资产清单 |

## 3. 当前迁移验收

### 3.1 基线和资产

| 验收项 | 状态 | 验收标准 | 当前证据 |
|---|---|---|---|
| 源任务清单 | ✅ | 14 个源入口、观测和时序基线已记录 | `GO2_MIGRATION_PLAN.md` 阶段 0 |
| Go2 XML 编译 | ✅ | `nq=19`、`nv=18`、`nbody=14`、`ngeom=56` | `go2-asset-check` |
| 环境 actuator | ✅ | `nu=12` | `go2-asset-check` |
| 关键名称 | ✅ | base、12 joints、4 foot sites 可解析 | `go2-asset-check` |
| Flat/Rough 场景 | ✅ | 均可构建、reset 和 step | smoke 与公共注册入口 |

复验命令：

```bash
cd /home/lxy/RoboLab/mjlab
uv run go2-asset-check
```

### 3.2 任务契约

| 任务组 | 数量 | 构建/reset/step | 观测维度 | 历史 1024×1000 | 当前结论 |
|---|---:|---|---|---|---|
| Trot | 1 | ✅ | ✅ | ✅ | 调用链完成，效果仍可继续调参 |
| Jump / Spring-Jump / Backflip | 3 | ✅ | ✅ | ✅ | 调用链完成；跨引擎行为需校准 |
| Handstand / Leggedstand | 2 | ✅ | ✅ | ✅ | 调用链完成；1000 iter 表现仍弱 |
| DreamWaQ / AMP-DreamWaQ | 2 | ✅ | ✅ | ✅ | VAE、AMP 和导出链路已验证 |
| CTS / AMP-CTS | 2 | ✅ | ✅ | ✅ | teacher/student latent 链路已验证 |
| AMP-TS / TS | 2 | ✅ | ✅ | ✅ | teacher 训练链路已验证 |
| AMP-TS-Student / TS-Student | 2 | ✅ | ✅ | ✅ | 当前纯递归蒸馏 runner 已做容量短测 |
| **合计** | **14** | **14/14** | **14/14** | **14/14 历史记录** | 不等同于全部动作收敛 |

最近一次契约复验：

```bash
cd /home/lxy/RoboLab/mjlab
uv run go2-contract-check
```

结果：2026-09-03，`14/14` 通过。

### 3.3 观测基线

| 任务 | Actor | Critic / Privileged | 状态 |
|---|---:|---:|---|
| Trot | 470 | 204 | ✅ |
| Jump | 470 | 210 | ✅ |
| Spring-Jump | 470 | 195 | ✅ |
| Backflip | 470 | 150 | ✅ |
| Handstand | 45 | 86 | ✅ |
| Leggedstand | 45 | 86 | ✅ |
| DreamWaQ | 45 | 783 | ✅ |
| AMP-DreamWaQ | 45 | 783 | ✅ |
| CTS | 45 | 278 | ✅ |
| AMP-CTS | 45 | 281 | ✅ |
| AMP-TS | 45 | 309 | ✅ |
| AMP-TS-Student | 45 | 309 | ✅ |
| TS | 45 | 309 | ✅ |
| TS-Student | 45 | 309 | ✅ |

维度一致只证明接口形状，没有单独证明字段数值、训练效果或物理等价。

### 3.4 算法与导出

| 能力 | 状态 | 已验收 | 尚需验收 |
|---|---|---|---|
| PPO | ✅ | 标准训练、checkpoint、镜像约束 | 最新配置的最终行为质量 |
| CTS | ✅ | 3:1 teacher/student、latent distillation、ONNX | 重构后的数值回归 |
| DreamWaQ | ✅ | VAE、速度监督、重建、KL、ONNX | 长训效果和 terrain 泛化 |
| AMP | ✅ | motion loader、transition discriminator、replay | 动捕覆盖与最终动作质量 |
| TS Teacher | ✅ | terrain/privileged encoder、checkpoint | 最终 teacher 质量 |
| TS Student | ✅ | 三层 LSTM、纯蒸馏 runner、recurrent ONNX | 长时闭环稳定性 |
| Policy Contract v1 | ✅ | 普通、多输入、递归接口前向 | Bundle 和跨版本迁移策略 |
| sim-to-sim runtime | 🟡 | Python adapter 和历史状态工具 | 独立进程连续控制与频率核对 |

## 4. 训练结果边界

历史 `1024 environments × 1000 iterations` 批量验证中，14 个任务均以退出码 0 完成并生成 `model_999.pt`。这些结果证明当时版本的端到端训练稳定性。

需要保留以下限制：

- 该轮完整训练早于最后一轮源代码语义修正；
- 最新修正只执行了对应契约、容量或短迭代测试；
- reward 改善不等于动作达到源视频质量；
- Jump、Handstand 和 Leggedstand 的历史 episode length 较短，仍需专项训练和 MuJoCo 参数校准；
- 当前没有真实 Go2 硬件闭环结论。

完整历史指标见 [GO2_MIGRATION_PLAN.md 的验收结果](GO2_MIGRATION_PLAN.md#14-验收结果与边界)。

## 5. 新架构实施验收表

### A0：冻结行为契约

| 验收项 | 状态 | 完成条件 |
|---|---|---|
| 16 个现有 Go2 ID 清单 | ✅ | Registry 可枚举 14 个源入口和 2 个公共入口 |
| 14 个观测维度 | ✅ | `go2-contract-check` 14/14 通过 |
| Go2 机器人契约 | ✅ | 12 joints、顺序、scale、dt 有权威记录 |
| ONNX v1 契约 | ✅ | 普通、多输入和 recurrent 前向通过 |

### A1：独立包边界

| 验收项 | 状态 | 完成条件 |
|---|---|---|
| `packages/lainloco` | ⬜ | 独立 `pyproject.toml` 和 `src/lainloco` 存在 |
| mjlab workspace/path dependency | ⬜ | LainLoco 依赖 mjlab，mjlab 不依赖 LainLoco |
| task entry point | ⬜ | 安装 LainLoco 后 mjlab 自动发现任务 |
| 独立 CLI | ⬜ | `lainloco --help` 可运行 |
| Go2 专用脚本迁出 | ⬜ | mjlab scripts 中不再拥有 Go2 专用入口 |

### A2：RobotSpec 与 Catalog

| 验收项 | 状态 | 完成条件 |
|---|---|---|
| RobotSpec | ⬜ | Go2 资产、实体、关节和控制参数单一来源 |
| TaskSpec | ⬜ | 技能和地形不包含算法参数 |
| TrainingSpec | ⬜ | 算法、模型、storage、runner 显式绑定 |
| ExperimentSpec | ⬜ | 可组合 Robot、Task、Training 和 Contract |
| Catalog | ⬜ | 无裸元组注册，ID 唯一且可查询 |
| 旧 ID alias | ⬜ | 现有命令继续得到等价配置 |

### A3：机器人优先任务目录

| 验收项 | 状态 | 完成条件 |
|---|---|---|
| Locomotion | ⬜ | Velocity/Trot 从旧 monolith 迁出 |
| Aerial | ⬜ | Jump/Spring-Jump/Backflip 独立可读 |
| Balance | ⬜ | Handstand/Leggedstand 独立可读 |
| Go2 MDP | ⬜ | Go2 专用 action/command/event/obs/reward 迁出通用 velocity |
| 旧 `env_cfgs.py` 清理 | ⬜ | 仅保留兼容导出，最终删除实现 |
| 回归 | ⬜ | 14/14 contract 和必要 smoke 继续通过 |

### A4：Task 与 TrainingProfile 分离

| 验收项 | 状态 | 完成条件 |
|---|---|---|
| 算法移出 Go2 namespace | ⬜ | 通用算法不导入机器人模块 |
| Go2 training profiles | ⬜ | 只保存 Go2 参数和 observation binding |
| AMP 组合能力 | ⬜ | AMP 作为辅助训练能力表达 |
| Student workflow | ⬜ | 使用显式 teacher checkpoint 和 distill 命令 |
| 新 Task ID | ⬜ | ID 不再永久编码算法名 |
| 兼容训练 | ⬜ | 旧 Task ID checkpoint 可继续加载 |

### A5：部署闭环

| 验收项 | 状态 | 完成条件 |
|---|---|---|
| Policy Bundle | ⬜ | ONNX、contract、normalization、robot、task、manifest 齐全 |
| 契约拒绝机制 | ⬜ | 错误 joint/obs/dt/version 会明确失败 |
| sim-to-sim | ⬜ | 独立 runtime 连续控制并正确维护 history/state |
| Go2 FSM | ⬜ | Passive、Stand、Policy、安全回退可运行 |
| 实机部署 | ➖ | 另立安全验收计划，不阻塞当前架构阶段 |

### A6：公开发布准备

| 验收项 | 状态 | 完成条件 |
|---|---|---|
| 仓库改名 | ⏸ | 用户调整为最终品牌名 |
| Python distribution | ⬜ | `lainloco` 构建和安装通过 |
| 项目 LICENSE | ⬜ | 明确项目代码许可证 |
| 第三方声明 | ⬜ | mjlab、Unitree 资产和源迁移代码来源清晰 |
| CI | ⬜ | lint、contract 和 CPU smoke 自动化 |
| 贡献指南 | ⬜ | 新机器人、新任务、新算法流程明确 |

## 6. 架构阶段门

进入下一阶段前必须满足：

| 阶段门 | 必须通过 |
|---|---|
| A1 → A2 | `import lainloco`、任务 entry point、现有 16 ID 可见 |
| A2 → A3 | Spec/Catalog 单元测试、旧 ID 映射测试 |
| A3 → A4 | 14/14 contract、资产检查、每个技能族至少一个 smoke |
| A4 → A5 | PPO、CTS、DreamWaQ、AMP、Teacher、Student 各一次 runner update |
| A5 → A6 | Policy Bundle 重载、sim-to-sim 连续控制、错误契约拒绝测试 |

任何阶段未通过对应门禁时，不删除旧实现和兼容路径。

## 7. 进度更新规则

1. 每次更新修改文档顶部日期。
2. 状态改为 ✅ 时必须填写验收命令、日志或产物路径。
3. 代码存在但没有执行验证时，只能标记为 🟡。
4. 历史长训与当前代码版本不一致时，必须标记为“历史结果”。
5. Contract、短训、收敛、sim-to-sim 和实机分别记录，禁止合并成“训练成功”。
6. 失败记录不删除；修复后补充新的成功记录并说明旧失败是否仍有效。
7. 重构阶段优先记录接口等价，不在同一验收项中混入奖励调参。

## 8. 常用验收命令

```bash
cd /home/lxy/RoboLab/mjlab

# 资产和机器人接口
uv run go2-asset-check

# 14 个源任务的动作及观测契约
uv run go2-contract-check

# 单任务有限步环境检查
uv run go2-smoke Mjlab-Trot-Flat-Unitree-Go2 --agent random --steps 4

# 查看全部注册入口
uv run list-envs

# 静态检查；具体范围随迁移阶段调整
uv run ruff check src tests
```

完整训练验证不应在每次结构性修改后自动重复。只有训练数学、观测数值、奖励或物理配置发生变化时，才选择对应任务进行定向训练验收。
