# 仿真、训练与 Play 后端演进策略

状态：设计基线 v1，2026-08-21 冻结。本文定义 RoboLab 如何从当前 Unitree RL MJLab
兼容路径演进为 RoboLab 自有的 MJLab 仿真、训练、回放与验证层。

## 1. 决策摘要

RoboLab 采用 **先兼容、再抽象、后替换** 的路线：

```text
RoboLab + Unitree backend + MJLab
  -> G1 黄金基线
  -> SimulationBackend 防腐层
  -> RoboLab-native MJLab task/train/play
  -> 双后端等价性验证
  -> mjlab_native 成为默认
```

RoboLab 不 fork 或重写整个 MJLab，也不把 Unitree RL MJLab 继续作为平台核心。长期边界是：

- MJLab、MuJoCo、MuJoCo-Warp、Warp、RSL-RL 和 Viser作为开源基础继续复用；
- RoboLab 拥有任务标识、Robot-to-task binding、训练/回放协议、observation/action schema、
  checkpoint metadata、验证与部署抽象；
- Unitree RL MJLab 降级为 `unitree_compat` 兼容和回归后端；
- RoboLab 自有的 `mjlab_native` 后端在等价性验收后成为默认；
- Unitree SDK 与 DDS 只进入 `integrations/unitree/`，不进入通用仿真接口或 Runtime。

## 2. 许可证边界

RoboLab 根项目继续使用 **MIT License**，本次决策不切换根许可证。

依赖或导入代码继续保留各自许可证：

| 内容 | 当前许可证 | 处理原则 |
|---|---|---|
| RoboLab 自有代码 | MIT | 根 `LICENSE` 保持不变 |
| `vendor/unitree_rl_mjlab/` | Apache-2.0 | 保留上游许可证、归属和修改记录 |
| MJLab、MuJoCo、MuJoCo-Warp、Warp | Apache-2.0 | 作为依赖使用，不重新标记为 MIT |
| RSL-RL | BSD-3-Clause | 保留原许可证 |
| Viser | MIT | 保留原许可证 |
| Robot model、mesh、motion、policy | 逐项确认 | 不自动继承代码许可证 |

MIT 与 Apache-2.0/BSD-3-Clause 依赖可以并存。架构去 Unitree 化和根许可证选择是两个独立问题。

## 3. 当前耦合与风险

当前平台运行路径仍直接依赖 `vendor/unitree_rl_mjlab`：

- API/CLI 使用 `vendor_root` 定位 vendor；
- task discovery 运行 vendor `scripts/list_envs.py`；
- play 直接构造 vendor `scripts/play.py` 命令；
- G1 task ID、reward、observation、actuator 与模型路径由 vendor 定义；
- Robot Profile 的 simulation/task binding 暴露 vendor 路径和 task ID。

这条路径可以完成首个 G1 样板，但如果继续作为平台默认，会导致：

- 第二个非 Unitree 机器人被迫遵循 Unitree 工程布局；
- 平台稳定接口随 vendor CLI 和 registry 变化；
- MotionSkill 隐式依赖脚本路径和 Unitree task ID；
- 难以证明 RoboLab 拥有独立的训练、验证和部署语义。

## 4. 目标架构

```text
WebUI / CLI / Agent
        |
        v
RoboLab Action + Job API
        |
        v
SimulationBackend Registry
  |                    |
  v                    v
unitree_compat       mjlab_native
  |                    |
vendor scripts      RoboLab tasks/train/play/export
  |                    |
  +------ MJLab / MuJoCo-Warp / RSL-RL / Viser ------+
```

建议的最终目录边界：

```text
packages/
  simulation/                 # 后端无关契约、注册表、稳定 ID
  mjlab_tasks/                # RoboLab 自有 task、mdp、robot binding
backends/
  unitree_compat/             # vendor discovery/train/play/export 适配
  mjlab_native/               # RoboLab 自有 MJLab 执行入口
vendor/
  unitree_rl_mjlab/           # 固定上游来源与迁移参考，不承载平台 API
integrations/
  unitree/                    # SDK/DDS/硬件 Driver
runtime/                      # 后端和厂商无关的数据面
```

现有 `packages/mjlab_adapter/` 是过渡实现。防腐层完成后，其 vendor 专用实现迁入
`backends/unitree_compat/`；通用接口不得继续以 vendor 路径为参数。

## 5. 平台后端契约

接口表达平台意图，返回受控 Job 描述，不在 API 进程内直接执行训练代码：

```python
class SimulationBackend(Protocol):
    backend_id: str

    def discover_tasks(self) -> list[TaskDefinition]: ...
    def train(self, recipe: TrainingRecipe) -> JobCommand: ...
    def play(self, config: PlayConfig) -> JobCommand: ...
    def evaluate(self, config: EvaluationConfig) -> JobCommand: ...
    def export(self, config: ExportConfig) -> JobCommand: ...
```

关键领域对象由 RoboLab 定义：

| 对象 | 必须固定的内容 |
|---|---|
| `TaskDefinition` | 稳定 task id、版本、能力、配置 schema、后端 binding |
| `TrainingRecipe` | Robot Profile、task、seed、资源、resume、最终配置 snapshot |
| `PlayConfig` | policy artifact、viewer、环境数、命令、录制和确定性选项 |
| `EvaluationConfig` | 场景、episode、metrics、阈值和证据要求 |
| `JobCommand` | argv、cwd、environment、allowed paths、outputs、backend id/revision |
| `PolicyArtifact` | checkpoint/ONNX、输入输出 schema、训练来源、hash |

平台 API 只接受 `backendId` 和稳定对象 ID。`vendor_root`、`play.py` 路径和 vendor task ID
属于 `unitree_compat` 私有 binding，不进入公共 Skill、API 或 WebUI 契约。

## 6. RoboLab 自有 MJLab 任务层

`packages/mjlab_tasks/` 复用上游 MJLab API，但由 RoboLab 拥有运动控制语义：

```text
packages/mjlab_tasks/
  src/robolab_mjlab_tasks/
    robots/                   # Profile -> MJLab entity/config binding
    tasks/                    # velocity、tracking 等稳定任务
    mdp/                      # observations、rewards、terminations、commands
    train/                    # schema 化训练入口
    play/                     # checkpoint/ONNX 回放入口
    export/                   # policy export 与 metadata
```

第一条 native task 应从 G1 flat velocity 开始，但稳定 task id 不使用厂商品牌，例如：

```text
motion.velocity.flat@1
```

Robot Profile binding 决定该任务如何实例化到 `unitree.g1.29dof`。这样新增机器人主要增加
Profile/robot binding，而不是复制整套 task 和 runner。

## 7. 五阶段迁移

### Stage 1：G1 黄金基线

在现有 GPU 训练完成后，使用当前 Unitree backend 得到并固定：

- G1 velocity checkpoint、训练配置、seed、代码和依赖 revision；
- trained play、Viser 视频/截图、结构化日志和 Artifact hash；
- observation/action metadata；
- unitree_mujoco sim-to-sim baseline；
- 关键运动指标与已知限制。

没有黄金基线，不开始 native 后端替换。

### Stage 2：防腐层

- 冻结 `SimulationBackend` 与领域对象；
- 建立 backend registry；
- 将现有 vendor discovery/play 封装为 `unitree_compat`；
- CLI/API 从 `vendor_root` 转为 `backendId`；
- 保存 backend id、版本、revision 和最终 `JobCommand` snapshot。

此阶段只改变依赖方向，不改变 G1 行为。

### Stage 3：RoboLab-native MJLab

- 建立 `packages/mjlab_tasks/`；
- 迁移 G1 flat velocity 的通用 task、MDP 和 robot-specific binding；
- 实现 RoboLab 自有 train/play/export 入口；
- 从任务配置机器导出 observation/action schema；
- 保持 vendor 源码只读，不在迁移时反向修改 vendor 来适配 native 后端。

### Stage 4：双后端验证

同一个固定 G1 policy、Profile、seed 和场景分别运行：

```text
unitree_compat --+
                 +--> Equivalence Report
mjlab_native ----+
```

比较分三层：

1. **结构必须完全一致**：joint ordering、observation/action shape、control frequency、action scale、
   task version 和 termination/reward term 集合；
2. **数值按阈值比较**：固定状态输入下的 observation、action transform、reward term 和接触事件；
3. **行为按统计比较**：固定场景 rollout 的速度误差、base height/姿态、foot contact、终止率、视频与
   sim-to-sim 行为。GPU 非确定性场景不要求逐帧 bitwise 相同。

每项阈值写入版本化 `EquivalenceSpec`，报告关联两个 backend revision 与所有 Artifact hash。

### Stage 5：切换默认并消除硬编码

只有 Stage 4 通过后：

- `mjlab_native` 成为默认；
- `unitree_compat` 保留兼容、回归和旧 checkpoint 诊断用途；
- `packages/core` 不导入 vendor；
- `services/api` 不默认依赖 vendor task 或路径；
- `apps/cli` 使用 RoboLab 稳定 task/backend ID；
- Robot Profile 不暴露 vendor 文件路径；
- MotionSkill 不依赖 `play.py` 或 vendor task ID；
- vendor 更新不再决定公共 API 版本。

## 8. 删除与回滚规则

- 双后端等价性通过前不删除 vendor task、脚本或 G1 模型；
- native 后端出现回归时可显式选择 `unitree_compat`，但不能静默回退；
- vendor 仍按固定 upstream revision 管理，不把 RoboLab 新功能写入 vendor；
- 每次默认后端切换必须有 migration note、兼容矩阵和回滚命令；
- legacy checkpoint 无法在 native 后端加载时，保留可解释的兼容性结果，而不是自动转换。

## 9. 完成定义

只有同时满足以下条件，才可称为完成“基于 MJLab 的深度定制”：

- RoboLab 拥有稳定的任务、训练、回放、评测和导出契约；
- 至少一个真实 G1 policy 在 native 后端训练或回放成功；
- unitree/native 双后端等价性报告通过；
- 平台公共 API 与 Skill 不暴露 vendor 路径或脚本；
- native 后端完成验证和 sim-to-sim 门禁；
- Unitree 只作为兼容 backend、Robot Profile/Driver 和来源归属出现。
