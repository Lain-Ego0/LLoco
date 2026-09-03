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
| ONNX/策略接口 | ✅ | v1 契约、多类 wrapper 与 Bundle runtime 已完成前向验证 | 长时数值回归 |
| 策略效果成熟度 | 🟡 | 历史 1024×1000 结果均有 reward 改善 | 按最新语义重新选择性长训 |
| sim-to-sim 完整闭环 | ✅ | Bundle runtime 在独立 mjlab 环境按 50 Hz 连续控制 | 扩展长时数值回归 |
| 真实 Go2 硬件闭环 | ➖ | 不属于当前迁移范围 | 独立安全和实机计划 |
| 架构设计文档 | ✅ | 目标边界、依赖和阶段已固化 | 用 ADR 管理后续变更 |
| LainLoco 独立包 | ✅ | uv workspace、独立包、task entry point、train/play/distill/export/Bundle CLI 已验收 | A6 发布元数据 |
| 机器人优先目录 | ✅ | Go2 Spec、Catalog、MDP、技能环境与部署 FSM 已归位 | 保持单一事实来源 |
| Task/Profile 分离 | ✅ | 技能 ID、8 个 TrainingSpec、learning 实现与显式 distill workflow 已分离 | 保持契约一致 |
| Policy Bundle | ✅ | 六件套、SHA-256、语义/ONNX 校验与 runtime 已完成 | 真实训练产物长期回归 |
| 品牌与仓库改名 | 🟡 | README 已使用 Lain's LocoLab | 仓库、包、CLI 正式改名 |
| 项目级许可证与声明 | 🟡 | 已建来源清单；Unitree 参考为 Apache-2.0，算法源快照无许可证 | 解决算法来源权限并由所有者选许可证 |

## 3. 当前迁移验收

### 3.1 基线和资产

| 验收项 | 状态 | 验收标准 | 当前证据 |
|---|---|---|---|
| 源任务清单 | ✅ | 14 个源入口、观测和时序基线已记录 | `GO2_MIGRATION_PLAN.md` 阶段 0 |
| Go2 XML 编译 | ✅ | `nq=19`、`nv=18`、`nbody=14`、`ngeom=56` | `lainloco validate asset` |
| 环境 actuator | ✅ | `nu=12` | `lainloco validate asset` |
| 关键名称 | ✅ | base、12 joints、4 foot sites 可解析 | `lainloco validate asset` |
| Flat/Rough 场景 | ✅ | 均可构建、reset 和 step | smoke 与公共注册入口 |

复验命令：

```bash
cd /home/lxy/RoboLab
uv run --package lainloco --extra cpu lainloco validate asset
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
cd /home/lxy/RoboLab
uv run --package lainloco --extra cpu lainloco validate contracts
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
| sim-to-sim runtime | ✅ | Bundle runtime 维护 history/state，独立 mjlab 50 Hz 闭环通过 | 长时稳定性与指标 |

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
| 14 个观测维度 | ✅ | `lainloco validate contracts` 14/14 通过 |
| Go2 机器人契约 | ✅ | 12 joints、顺序、scale、dt 有权威记录 |
| ONNX v1 契约 | ✅ | 普通、多输入和 recurrent 前向通过 |

### A1：独立包边界

| 验收项 | 状态 | 完成条件 |
|---|---|---|
| `packages/lainloco` | ✅ | 独立 `pyproject.toml`、`src/lainloco` 和 workspace lock 已生成 |
| mjlab workspace/path dependency | ✅ | 元数据测试确认 LainLoco → mjlab 单向依赖 |
| task entry point | ✅ | `mjlab.tasks` entry point 自动注册 16 个旧 ID 和 8 个新 ID |
| 独立 CLI | ✅ | `uv run --package lainloco --extra cpu lainloco --help` 通过 |
| Go2 专用脚本迁出 | ✅ | asset/contract/smoke 已迁至 `lainloco validate`，mjlab 入口已移除 |

### A2：RobotSpec 与 Catalog

| 验收项 | 状态 | 完成条件 |
|---|---|---|
| RobotSpec | ✅ | 12 关节顺序、动作尺度、physics/control dt 与实体名已固化 |
| TaskSpec | ✅ | 8 个技能/地形规格不含优化器或算法参数 |
| TrainingSpec | ✅ | 8 个 profile 显式绑定当前可导入的算法、模型、storage 与 runner |
| ExperimentSpec | ✅ | 16 个迁移组合连接 Robot、Task、Training 与 PolicyContract |
| Catalog | ✅ | 不可变 Catalog 拒绝重复 ID，单元测试通过 |
| 旧 ID alias | ✅ | 16 个旧 ID 可见；8 个新技能 ID 与旧配置逐项等价 |
| Core 模块边界 | ✅ | Robot/Task/Training/PolicyContract/Experiment 分文件，Catalog 使用显式 compose 入口 |

### A3：机器人优先任务目录

| 验收项 | 状态 | 完成条件 |
|---|---|---|
| Locomotion | ✅ | Velocity/Trot 位于 `tasks/locomotion`，训练观测变体独立存放于 `variants.py` |
| Aerial | ✅ | Jump/Spring-Jump/Backflip 工厂由 `tasks/aerial` 拥有 |
| Balance | ✅ | Handstand/Leggedstand 工厂由 `tasks/balance` 拥有 |
| Go2 MDP | ✅ | action/command/event/observation/reward 均已迁出通用 velocity |
| 旧 `env_cfgs.py` 清理 | ✅ | mjlab 实现已删除，entry point 提供无反向依赖的旧模块 alias |
| 回归 | ✅ | 迁移后 14/14 contract、52 项结构/MDP 回归和静态检查通过 |

### A4：Task 与 TrainingProfile 分离

| 验收项 | 状态 | 完成条件 |
|---|---|---|
| 算法移出 Go2 namespace | ✅ | 算法、模型、storage、motion、symmetry 已迁至 `lainloco.learning`，不导入机器人模块 |
| Go2 training profiles | ✅ | 参数、observation binding 与 Go2 runner 位于 `go2/training`，Catalog 引用通用 learning 实现 |
| AMP 组合能力 | ✅ | TrainingSpec 将 AMP 表达为 auxiliary capability，AMP-CTS 真实更新通过 |
| Student workflow | ✅ | `lainloco distill` 显式接收 task/profile/teacher checkpoint 并使用独立 recurrent runner |
| 统一训练/回放 | ✅ | `lainloco train` 与 `play` 均显式解析 task/profile；计划解析与 dry-run 通过 |
| 新 Task ID | ✅ | 8 个 `LainLoco-*` ID 只描述技能；训练组合由 ExperimentSpec/profile 解析 |
| 兼容训练 | ✅ | 旧模块/Task ID alias 通过，迁移前 TS `model_999.pt` 已由新 distill workflow 加载更新 |

### A4 实施记录（2026-09-03）

- 自定义算法、模型、rollout storage、motion loader 和 symmetry 已从
  `mjlab.tasks.velocity.rl.go2_algorithms` 迁至 `lainloco.learning`；Go2 专用 ONNX/history/
  recurrent 适配器迁至 `robots/unitree/go2/deploy/policy.py`，runner 与配置由
  `robots/unitree/go2/training/` 拥有。通用 `learning`/`runtime` 不再反向导入机器人领域。
- mjlab 的 Velocity runner 已去除 Go2 导入，仅保留内置速度任务所需的通用 ONNX
  导出；LainLoco runner 负责策略契约和 recurrent student 逻辑。
- entry point 为旧算法与 `rl_cfg` 模块提供临时 alias；新旧路径和 TrainingSpec 引用
  均可导入。依赖方向现由 AST contract test 固化；迁入代码的动态 PyTorch 属性已全部收窄，根级 ty/pyright
  不再排除任何实现文件。
- `lainloco distill` 已用迁移前的 `mjlab/logs/go2_validation/go2_ts/model_999.pt`
  完成 2 环境、50 steps/env 的真实 student 更新；workflow 默认本地 TensorBoard 且作用域化
  teacher checkpoint 环境变量。
- `lainloco train`、`play` 已封装 mjlab 维护的 launcher/viewer，并在启动前通过 Catalog
  校验 task/profile；trained playback 强制提供本地 checkpoint，普通训练拒绝误用蒸馏 profile。
- A4 阶段门已在迁移后实现上逐项执行：PPO、CTS、DreamWaQ、AMP-CTS、TS teacher、
  TS student 均完成一次真实 CPU runner update。最终 ruff、ty、pyright、`80 passed`
  与 14/14 contract 通过；日志位于 `mjlab/logs/go2_a4_smoke/`。

### A5：部署闭环

| 验收项 | 状态 | 完成条件 |
|---|---|---|
| Policy Bundle | ✅ | ONNX、contract、normalization、robot、task、manifest 齐全 |
| Checkpoint 导出 | ✅ | `lainloco export` 严格加载 task/profile 对应 actor 并直接生成完整 Bundle |
| 契约拒绝机制 | ✅ | 错误 joint/obs/dt/version、摘要和 ONNX shape 会明确失败 |
| sim-to-sim | ✅ | 独立 runtime 连续控制并正确维护 history/state |
| Go2 FSM | ✅ | Passive、Stand、Policy、锁存安全回退可运行 |
| 实机部署 | ➖ | 另立安全验收计划，不阻塞当前架构阶段 |

### A5 实施记录（2026-09-03）

- `runtime/policy_bundle.py` 生成并重载 `policy.onnx`、`contract.json`、
  `normalization.npz`、`robot.yaml`、`task.yaml`、`manifest.json`；manifest 固化
  format/contract version、robot/task/profile 和每个产物的 SHA-256/大小，且不覆盖目标目录。
- 加载器交叉核对 RobotSpec、TaskSpec、PolicyContract、normalization 与 ONNX 图，拒绝
  错误 robot/joint/action/observation/history/recurrent/dt/version、路径、摘要和 shape。
- PolicyContract 现显式记录 primary 与 conditional 字段；history 以及 teacher 的
  `terrain || privileged` 维度/顺序不再依赖 profile 名称或运行时猜测。
- Go2 contract version、actor/history/teacher/recurrent 固定维度现在集中于
  `go2/contract.py`；训练 Experiment 与部署适配器引用同一份事实。
- `BundlePolicyRuntime` 统一执行单输入、conditional history 与 recurrent ONNX，在 episode
  边界按 env id 清零 history/hidden/cell；`SimToSimRuntime` 强制每个 control tick 一次推理，
  并核对 `physics_dt=0.005`、`control_dt=0.02`。
- `lainloco bundle create|validate|rollout` 已实际完成 2 环境、8 tick 的 headless mjlab
  闭环：8 次 policy call、0.16 s 仿真时间、退出码 0。
- `lainloco export` 已用迁移前 `go2_ts/model_999.pt`、A4 TS teacher 和 A4 TS student
  checkpoint 完成严格导出与 Bundle 重载；A4 两份 Bundle 均在 2 环境执行 8 tick/0.16 s
  闭环。旧 teacher/student 的 running normalizer 会写入 ONNX，旧 teacher 缺少的
  profile 常量 `min_std` 会从当前显式配置补齐；其余权重仍保持 `strict=True` 加载。
- TS student Bundle 以 `obs,h,c → actions,he,ce` 递归接口运行，并按环境选择性清零状态。
- Go2 FSM 已实现 Passive、Stand 平滑插值、Policy 动作转换，以及急停、陈旧观测、
  非有限/越界状态与动作、错误 dt 触发的锁存安全回退；SDK、硬件 mapping 和物理安全仍
  明确不在此验收内。
- 当前最终门禁：ruff、format、ty、pyright 通过，`115 passed`，14/14 task contract 通过，
  `uv build --package lainloco` 成功生成 wheel 与 sdist。

### A6：公开发布准备

| 验收项 | 状态 | 完成条件 |
|---|---|---|
| 仓库改名 | ⏸ | 用户调整为最终品牌名 |
| Python distribution | ✅ | `uv build --package lainloco` 与 workspace 安装通过 |
| 项目 LICENSE | ⏸ | 等待所有者选择，且需先解决算法源快照再分发权限 |
| 第三方声明 | 🟡 | `THIRD_PARTY_NOTICES.md` 已建；算法源快照无许可证仍是发布阻塞 |
| CI | ✅ | `.github/workflows/ci.yml` 覆盖 lint、type、contract、CPU smoke 和 build |
| 贡献指南 | ✅ | `CONTRIBUTING.md` 固化新机器人、任务、算法、契约和验收流程 |
| 架构决策记录 | ✅ | 0001–0005 ADR 已记录背景、决定、替代方案、后果和迁移影响 |

### A6 当前记录（2026-09-03）

- GitHub Actions 已使用 locked CPU workspace 执行 ruff、ty、pyright、contract/regression、
  asset、14 task contract、smoke 和 distribution build；本地等价命令已通过，远端 workflow
  首次运行状态仍需在 push 后确认。
- 第三方清单确认当前 16 个 Go2 OBJ 与两个 MJCF 文件逐字节来自本地
  `unitree_rl_mjlab-main` 快照；其 canonical upstream 与本地 `LICENCE` 均确认 Apache-2.0。
  `My_unitree_go2_gym-main` 与公开上游 README 逐字节一致但仓库没有许可证，因而其迁移
  实现公开发布前仍须取得授权或独立替换。
- 未擅自选择项目许可证，也未执行会改变远端/本地路径的仓库改名；二者保留为所有者决策。
- `docs/architecture/decisions/` 已落地机器人优先、Task/Profile 分离、mjlab 扩展边界、
  PolicyContract 版本化和旧 Task ID 兼容五项 ADR。
- 测试树已分为 `tests/contracts`、`tests/integration` 与 `tests/training`；CPU CI 执行前两层，
  training 层要求记录 revision、task/profile、容量、预算、checkpoint 与行为指标。

### A1/A2 实施记录（2026-09-03）

本轮只迁移包、注册和领域对象边界，没有修改环境数值、奖励或训练数学：

```bash
cd /home/lxy/RoboLab

uv run --package lainloco --extra cpu lainloco --help
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --package mjlab --extra cpu \
  pytest -q tests/contracts/test_lainloco_catalog.py
uv run --package mjlab --extra cpu ruff check packages/lainloco/src tests/contracts
uv run --package mjlab --extra cpu ty check
uv run --package mjlab --extra cpu pyright
uv run --package lainloco --extra cpu lainloco validate asset
uv run --package lainloco --extra cpu lainloco validate contracts
uv run --package lainloco --extra cpu lainloco validate smoke \
  Mjlab-Velocity-Rough-Unitree-Go2 --agent zero --steps 1 --num-envs 1
```

结果：Catalog/entry point/依赖方向/alias 等价测试通过，ruff、ty、pyright
通过，资产检查通过，14/14 源任务 contract 通过，Rough 基线一步 smoke 通过。

### A3 实施记录（2026-09-03）

- `env_cfgs.py` 的环境工厂实现已物理迁至 `tasks/locomotion`、`tasks/aerial`、
  `tasks/balance` 与共享 `tasks/special.py`；mjlab 不再保存该实现。
- `tasks/legacy.py` 已缩减为纯兼容重导出，不再包含环境构造逻辑；Catalog 直接引用
  各技能模块，所有权测试防止回退到兼容层。
- Go2 action、command、event、observation 和 reward 已迁至
  `lainloco.robots.unitree.go2.mdp`；mjlab 原 observation/reward 文件仅保留通用项。
- Flat 与 Rough 不再互相调用；两个公开 factory 并列组合同一个显式 terrain-profile
  基础构造器，并由结构契约测试阻止恢复 Rough→Flat 删除式继承。
- Aerial/Balance 的公开 factory 已由各领域目录拥有；五个源任务仍共享
  `tasks/special.py` 的初始化与分支构造，以避免在没有数值快照保护时复制 14 套环境。
  后续若继续拆成逐技能文件，先补配置语义快照，再按单一变化原因提取，不把行数当完成指标。
- bootstrap 在运行时提供旧模块与 package-level symbol alias，旧导入路径仍可加载，
  同时保持发布元数据中的依赖方向为 LainLoco → mjlab。
- ruff、ty、pyright 通过；Catalog、alias 与 mjlab 通用 MDP 回归 `52 passed`；
  `lainloco validate contracts` 再次 `14/14` 通过。

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
cd /home/lxy/RoboLab

# 资产和机器人接口
uv run --package lainloco --extra cpu lainloco validate asset

# 14 个源任务的动作及观测契约
uv run --package lainloco --extra cpu lainloco validate contracts

# 单任务有限步环境检查
uv run --package lainloco --extra cpu lainloco validate smoke \
  Mjlab-Trot-Flat-Unitree-Go2 --agent random --steps 4

# 查看全部注册入口
uv run --package lainloco --extra cpu lainloco envs

# 显式组合训练/回放，以及 checkpoint 直接导出 Bundle
uv run --package lainloco --extra cpu lainloco train go2/trot \
  --profile ppo --num-envs 2 --iterations 1 --gpu-ids cpu --dry-run
uv run --package lainloco --extra cpu lainloco play go2/trot \
  --profile ppo --agent random --dry-run
uv run --package lainloco --extra cpu lainloco export /path/to/model.pt \
  --task go2/trot --profile ppo --dry-run

# 静态检查；具体范围随迁移阶段调整
uv run --package mjlab --extra cpu ruff check packages/lainloco/src tests/contracts
```

完整训练验证不应在每次结构性修改后自动重复。只有训练数学、观测数值、奖励或物理配置发生变化时，才选择对应任务进行定向训练验收。
