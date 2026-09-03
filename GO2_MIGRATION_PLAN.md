# Unitree Go2 全量迁移计划

本文档说明如何将 `/home/lxy/下载/My_unitree_go2_gym-main` 迁移到
`/home/lxy/RoboLab/mjlab`，目标是在当前 mjlab 环境中完整复现源项目的 14 个
任务、四套自定义算法、回放、ONNX 导出和 sim-to-sim 接口。

## 1. 迁移目标

迁移后的实现应保持以下行为契约：

- 12 个执行器的关节顺序、关节名和动作方向；
- `dt=0.005`、控制 decimation、默认关节姿态和动作缩放；
- actor、privileged、critic observation 的维度、顺序、缩放、噪声和裁剪；
- 历史观测长度、命令采样、奖励、终止条件和 domain randomization 语义；
- 训练、回放、ONNX 推理和 Unitree 部署所需的输入输出格式。

迁移采用 mjlab 的实体、传感器、manager、任务配置和 RSL-RL runner 体系，
不把 Isaac Gym 的 `LeggedRobot` 类原样复制过去。

目录组织采用“任务配置分层、实现集中复用”的方式，而不是复制 14 份平行环境：

```text
mjlab/src/mjlab/tasks/velocity/
├── config/go2/          # 14 个入口的配置工厂与注册，按任务名调用
├── mdp/                 # 共享观测、奖励、命令、动作延迟和传感器逻辑
└── rl/go2_algorithms/   # CTS/AMP/DreamWaQ/TS 的模型、算法、storage、runner
```

这样既能在注册表中按任务独立启动，又避免资产、joint order、观测拼接和奖励函数
在平行目录中发生漂移。若后续某个任务出现大量独有逻辑，再将该任务拆成薄配置模块；
仍保持底层 MDP 和算法模块单一来源。

### 关于“每个任务复制一套目录”

平行目录在查看单个任务时确实直观，但在本次迁移范围内会带来几个实际问题：

- 同一 Go2 资产、关节顺序和传感器配置要复制 14 次，后续修正容易只改到部分任务；
- 观测维度和字段顺序会在复制后悄悄分叉，训练 checkpoint、ONNX 和部署输入难以互换；
- CTS/AMP/DreamWaQ/TS 的模型、storage 和 runner 会出现多份近似实现，调用关系反而更难追踪；
- 注册表、导入路径和测试矩阵扩大，新增一个公共修复需要逐目录同步。

因此本计划采用“一个任务名对应一个配置工厂”的调用边界：

```text
CLI task id
  └─ config/go2/__init__.py 注册入口
       └─ config/go2/env_cfgs.py 任务差异（奖励/观测/时序）
            ├─ mdp/                 共享实现
            └─ rl/go2_algorithms/   算法实现
```

调用仍按任务名隔离，例如 `uv run train Mjlab-Backflip-Flat-Unitree-Go2`；
目录只承载实现复用，不改变任务的独立入口。只有当某任务的资产或仿真生命周期
真正不同，才值得增加独立目录，并保留上述注册边界。

## 阶段0基线记录（源项目）

以下数值来自源项目各任务的 `*_Config.py`，作为迁移时的维度和时序基线；其中
critic 列是源项目直接送入 value/encoder 的配置维度，CTS/TS 的历史输入另列。

| 源任务 | actor obs | critic / privileged | history | episode (s) | command resample (s) |
|---|---:|---:|---:|---:|---:|
| `go2_trot` | 470 (47×10) | 204 (68×3) | 10 | 24 | 5 |
| `go2_jump` | 470 (47×10) | 210 (70×3) | 10 | 24 | 5 |
| `go2_spring_jump` | 470 (47×10) | 195 (65×3) | 10 | 5 | 5 |
| `go2_backflip` | 470 (47×10) | 150 (50×3) | 10 | 4 | 5 |
| `go2_handstand` | 45 | 86 | 1 | 20 | 10 |
| `go2_leggedstand` | 45 | 86 | 1 | 20 | 5 |
| `go2_dreamwaq` | 45 | 783 (261×3) | 5×45 | 20 | 10 |
| `go2_amp_dreamwaq` | 45 | 783 (261×3) | 5×45 | 20 | 10 |
| `go2_cts` | 45 | 278 | 5×45 | 20 | 8 |
| `go2_amp_cts` | 45 | 281 | 5×45 | 20 | 10 |
| `go2_amp_ts` | 45 | 309 | student history | 20 | 10 |
| `go2_amp_ts_student` | 45 | 309 | student history | 20 | 10 |
| `go2_ts` | 45 | 309 | student history | 20 | 10 |
| `go2_ts_student` | 45 | 309 | student history | 20 | 10 |

公共基线：12 个动作、仿真步长 `dt=0.005`、控制 decimation=4、动作缩放
`0.25`。特殊动作任务的默认姿态、周期和奖励权重已接入，仍需以行为短训检查
MuJoCo 与 PhysX 的数值差异；该表用于防止观测维度和时序发生无意漂移。

源项目基线启动形式：`python legged_gym/scripts/train.py --task <task>`；回放形式：
`python legged_gym/scripts/play.py --task <task>`。当前 mjlab 对应入口为
`uv run train <task>`、`uv run play <task> --agent random`，阶段1的有限步无界面验证
使用 `uv run --package lainloco --extra cpu lainloco validate smoke <task>
--agent {zero,random} --steps 4`。

## 2. 任务清单

源项目任务入口位于：

`/home/lxy/下载/My_unitree_go2_gym-main/legged_gym/envs/__init__.py`

### 标准 PPO

- `go2_trot`
- `go2_jump`
- `go2_spring_jump`
- `go2_backflip`
- `go2_handstand`
- `go2_leggedstand`

### DreamWaQ

- `go2_dreamwaq`
- `go2_amp_dreamwaq`

### CTS

- `go2_cts`
- `go2_amp_cts`

### Teacher-Student

- `go2_amp_ts`
- `go2_amp_ts_student`
- `go2_ts`
- `go2_ts_student`

四套算法能力为：标准 PPO、AMP、CTS、DreamWaQ；AMP-CTS、AMP-DreamWaQ、
AMP-TS 是组合变体，TS student 使用 LSTM 蒸馏流程。

## 3. 参考实现

以下项目可作为部分实现参考：

`/home/lxy/下载/unitree_rl_mjlab-main`

重点参考：

- `src/assets/robots/unitree_go2/`：Go2 MJCF、网格、执行器和初始状态；
- `src/tasks/velocity/config/go2/`：Go2 flat/rough velocity task；
- 接触传感器、地形扫描、碰撞配置和 mjlab 任务注册方式。

参考项目面向较旧的 mjlab 版本，最终代码以 RoboLab 当前 mjlab API 为准。

## 4. 版本与依赖策略

源项目使用 Isaac Gym、Python 3.8、旧版 `rsl_rl` 和旧版 MuJoCo；当前 mjlab
使用 Python 3.10–3.13、MuJoCo-Warp 和 `rsl-rl-lib`。

迁移原则：

- 使用 RoboLab/mjlab 当前虚拟环境；
- 不安装或覆盖源项目的 Isaac Gym 和旧版 `rsl_rl`；
- 旧算法代码只作为网络结构和数学逻辑参考；
- 自定义算法放在独立的 model、storage、algorithm、runner 模块中；
- 通过 mjlab 的任务注册机制接入，不修改通用训练脚本的核心流程。

## 5. Go2 机器人资产迁移

建议目录：

```text
mjlab/src/mjlab/asset_zoo/robots/unitree_go2/
├── __init__.py
├── go2_constants.py
└── xmls/
    ├── go2.xml
    ├── scene_go2.xml
    └── assets/*
```

步骤：

1. 以 MuJoCo MJCF 为主迁移机器人结构，URDF 只用于参数和名称参考；
2. 用 `MjSpec` 封装 XML，并用 `EntityCfg` 描述实体和初始状态；
3. 用 `BuiltinPositionActuatorCfg` 配置 hip、thigh、calf 执行器；
4. 明确 `base_link`、四个足端 site、足端碰撞 geom 和非足端碰撞 geom 名称；
5. 增加 IMU、地形 raycast、足端高度和接触传感器；
6. 保留粗糙地形、斜坡、楼梯和特殊动作场景；
7. 先确认模型能够 reset、step 和渲染，再接入训练。

资产验收只需确认模型可编译、执行器数量为 12、关键实体和传感器名称可解析。
**无需检查文件哈希值**，不以哈希一致作为迁移条件。

## 6. Isaac Gym 到 mjlab 的映射

| 源项目 | mjlab 实现 |
|---|---|
| `compute_observations` | observation groups / observation terms |
| `_reward_*` | `RewardTermCfg` 和自定义 reward function |
| `check_termination` | `TerminationTermCfg` |
| `_reset_*` | reset events 和实体初始状态 |
| `_post_physics_step_callback` | interval events、commands、metrics 或 task state |
| Isaac Gym contact tensor | `ContactSensorCfg` |
| `_get_heights` / heightfield | `RayCastSensorCfg`、`TerrainHeightSensorCfg` |
| `Terrain` curriculum | terrain generator 和 curriculum terms |
| `task_registry.register` | `register_mjlab_task` |

公共 MDP 先覆盖速度命令、姿态/速度/动作观测、地形高度、足接触、命令跟踪、
姿态稳定、关节限制、碰撞、足端 clearance、动作平滑和课程。特殊任务再覆盖
phase、目标姿态、跳跃/翻转奖励和终止逻辑。

## 7. 算法迁移方案

### 7.1 标准 PPO

优先使用当前 mjlab 已支持的 RSL-RL PPO。源项目的 MLP 结构和 PPO 超参数可
转换为 `RslRlOnPolicyRunnerCfg`。先对齐 observation/action/reward，再调参数。

### 7.2 CTS

实现 Go2 专用 CTS model 和 runner：

- teacher 使用 privileged observation；student 使用 history observation；
- 保留 teacher/student 环境比例和索引策略；
- rollout storage 同时保存两类输入、动作、log-prob、value 和 history；
- PPO 更新与 student encoder/latent 蒸馏更新分开；
- inference 和 ONNX 导出只使用 student 路径。

当前调用仍保持“每个任务一个 task id、底层实现单一来源”：CTS/AMP-CTS 的
actor 解析为 `CtsActorModel`，value 解析为 `CtsCriticModel`（输入为
`latent(32) || critic_obs`）；其他变体解析对应的 DreamWaQ/TS actor 与共享
value。这样无需在 14 个目录中复制同一套模型代码，也不会隐藏调用关系。

### 7.3 AMP

迁移动作参考数据加载、AMP observation、replay buffer、判别器和生成器更新。
动捕数据需要按 MuJoCo 的 joint/body 顺序重新整理，不能直接假定 Isaac Gym
的刚体索引仍然适用。

当前 loader 接受源项目 `datasets/mocap_motions_go2/*.txt`（JSON 内容）目录；训练
或最小 AMP update 前设置 `GO2_MOTION_DIR=/path/to/mocap_motions_go2`，会采样真实
expert transition。未设置时仍保留无数据环境下的有限 smoke fallback。

### 7.4 DreamWaQ

迁移 VAE/CENet、历史观测编码、显式速度估计、latent 重建和 KL loss。辅助
优化器、样本 mask、live 样本处理和 checkpoint 保存封装在自定义 algorithm 中。

### 7.5 AMP-TS、TS 和组合算法

先分别完成基础 AMP、CTS、DreamWaQ，再组合成 AMP-CTS、AMP-DreamWaQ 和
AMP-TS。TS student 需要单独处理 LSTM hidden state、蒸馏 runner 和 ONNX 导出；
纯 TS 不启用 AMP 判别器和动捕数据。当前导出 wrapper 将策略条件输入显式化；
TS teacher 使用 terrain 与 privileged 的拼接输入，TS student 的源兼容递归接口使用
当前帧 `obs(45)` 以及三层 LSTM 的 hidden/cell，并由部署适配器跨步保存和按环境清零状态。

## 8. 观测、动作和 checkpoint 契约

建立固定契约表，至少包含：

- joint name 到 action index 的映射；
- 每个 observation term 的维度、顺序、scale、noise 和 clip；
- privileged/critic 额外字段及拼接顺序；
- action scale、default joint position、PD 参数和 torque limit；
- history frame 数、拼接方向和 reset 填充规则；
- student ONNX 输入维度及 hidden-state 处理方式。

调用约定：

```text
uv run train Mjlab-<Task>-<Flat|Rough>-Unitree-Go2
uv run play  Mjlab-<Task>-<Flat|Rough>-Unitree-Go2 --agent random
```

任务 id 在 `config/go2/__init__.py` 集中注册；配置工厂位于
`config/go2/env_cfgs.py`，公共 MDP 位于 `velocity/mdp/`，自定义算法位于
`velocity/rl/go2_algorithms/`。新增任务只需增加薄配置入口，不复制资产、观测或
算法实现。

Isaac Gym checkpoint 不保证可以直接加载到当前 mjlab/RSL-RL。默认迁移策略是
重写算法和网络后重新训练；只有在契约完全一致并完成一次回放确认后，才考虑转换
旧 checkpoint。

## 9. 实施阶段

### 阶段 0：基线记录

记录每个任务的配置、观测维度、动作维度、关键奖励项和启动命令，保存少量日志
或视频作为行为参考，不进行长时间重跑。

### 阶段 1：Go2 资产和最小环境

交付 Go2 entity、actuator、contact/raycast sensor、flat terrain，以及零动作和
随机动作回放入口。

### 阶段 2：公共 MDP 和标准 PPO

完成公共 Go2 velocity/rough-terrain MDP，依次注册六个标准 PPO 任务：
`go2_trot`、`go2_jump`、`go2_spring_jump`、`go2_backflip`、`go2_handstand`、
`go2_leggedstand`。

### 阶段 3：CTS 和 AMP

先完成 `go2_cts`，再完成 `go2_amp_cts`；独立完成 AMP motion loader 和判别器
后，再接入 AMP-TS。

### 阶段 4：DreamWaQ 和 TS

完成 `go2_dreamwaq`、`go2_amp_dreamwaq`、`go2_ts` 和两个 student 任务，包括
辅助 loss、LSTM、蒸馏和策略导出。

### 阶段 5：部署接口

统一 `train`、`play`、ONNX 导出、MuJoCo 回放和 Unitree C++ 部署所需的输入输出
格式。硬件部署必须在安全和吊装条件下进行。

## 10. 测试与验收策略

遵循正常迁移流程，但控制测试规模：

- **不要过度测试**：文档和迁移阶段不要求每个任务长时间训练、大规模 benchmark
  或逐文件对比；
- 每个阶段只做必要的 smoke test：导入、模型编译、reset、少量 step、shape 检查
  和短时零动作/随机动作回放；
- 标准 PPO 做短跑训练，确认 reward、done、time-out 和 checkpoint 保存；
- 自定义算法完成一次最小 rollout/update，确认 loss、storage、runner 和导出路径；
- 只有出现 NaN、shape 错误、接触/动作方向错误或明显行为回归时，才增加针对性测试；
- **无需检查哈希值**，资产验收以可加载、名称匹配和仿真行为正常为准。

## 11. 主要风险

- MuJoCo-Warp 与 Isaac Gym PhysX 的接触和数值细节不同，奖励曲线不保证逐点一致；
- 地形、摩擦、接触 condim、PD 和动作延迟会显著影响 sim-to-real；
- 旧 runner/storage 与当前 RSL-RL API 的差异是自定义算法的主要迁移风险；
- 任意 observation 顺序或 joint order 改动都必须同步更新训练、回放、ONNX 和部署。

## 12. 完成标准

满足以下条件后视为完成：

1. 14 个任务均可通过 mjlab 任务注册和 CLI 启动；
2. 四套算法及组合变体均有对应的 model、algorithm、storage 和 runner；
3. Go2 flat/rough/special-action 场景能够 reset、step、回放和保存 checkpoint；
4. AMP、DreamWaQ、CTS、TS 的专用输入、loss、storage 和导出逻辑可运行；
5. 至少一个 teacher policy 和一个 student policy 能够按约定格式导出和推理；
6. 算法策略、训练 checkpoint 和 ONNX 导出接口的输入输出语义保持不变；真实 Unitree
   硬件闭环不属于本次迁移范围。

## 13. 当前执行状态

阶段 0（源项目基线记录）和阶段 1（Go2 资产、最小环境、有限回放入口）已完成；
阶段 2–4 的任务、MDP 与算法调用链已接入，当前处于源代码语义复核和训练验收阶段。
阶段 5 仅保留算法推理与 ONNX/sim-to-sim 接口，真实 Go2 硬件闭环不在本次范围内。

已完成的增量实现：

- Go2 MJCF、网格、12 个 position actuator、默认姿态和动作缩放已接入当前 mjlab API；
- MJCF 编译验收结果：`nq=19`、`nv=18`、`nu=12`、`nbody=14`、`ngeom=56`，关键 `base_link`、四足端和 12 个关节名称可解析；
- Go2 flat/rough 场景、足端接触/高度传感器和碰撞匹配已接入；
- `Mjlab-Velocity-Flat-Unitree-Go2`、`Mjlab-Velocity-Rough-Unitree-Go2` 已注册；
- Trot 任务已注册，actor 为 47 维单帧 × 10，critic 为 68 维特权单帧 × 3；
- Trot 已接入源项目的平面速度/偏航跟踪、垂向速度、姿态、基座高度、关节加速度、动作平滑、髋部默认姿态、trot 相位、静止接触等奖励；Jump 已接入源项目的正向垂向/角速度/姿态核、默认姿态/髋部姿态、动作延迟和静止接触奖励；两者均保留 470/204 或 470/210 的观测契约。
- 六个标准 PPO 入口均已注册并可 reset/step；Trot/Jump/Spring-Jump/Backflip 已对齐 470 维 actor 历史，分别对齐 204/210/195/150 维 critic 历史；Handstand/Leggedstand 已对齐 45 维 actor 和 86 维 critic；
- 五个特殊动作已接入独立观测、源奖励表、终止、reset、动作/观测延迟和触发状态；
  当前剩余工作是行为级校准，而不是缺失任务调用链；
- 已用少量 CPU smoke test 验证 Go2 编译、flat/rough/trot reset-step、任务列表和现有 velocity 测试。
- 初始迁移阶段新增有限步、无 viewer 的 `go2-smoke` 入口；A1 重构后实现已迁至
  `lainloco validate smoke`，继续支持 `zero`/`random` 动作回放。
- 初始迁移阶段的 `go2-asset-check` 已迁至 `lainloco validate asset`，分别检查原始 Go2 XML 的 `nq=19,nv=18,nbody=14,ngeom=56`、环境注入后的 `nu=12` 及关键 body/joint/site 名称；该检查不比较哈希值。
- 初始迁移阶段的 `go2-contract-check` 已迁至 `lainloco validate contracts`，对源项目 14 个任务逐一执行一次无 viewer 的
  reset/zero-action step，并检查 12 动作接口及 actor/critic 观测维度；该检查不做长时训练、性能基准或哈希比较。
- `lainloco validate contracts` 已通过：14/14 个源任务均完成构建、reset、zero-action step，
  actor/critic 维度与阶段 0 基线一致。
- `lainloco validate asset --task Mjlab-Velocity-Flat-Unitree-Go2` 已实际通过，输出 XML `nq=19,nv=18,nbody=14,ngeom=56`，环境 `nu=12`，关键名称全部解析成功。
- 新增 `tasks/velocity/rl/go2_algorithms/` 兼容层：CTS teacher/student 编码器、DreamWaQ VAE、AMP 判别器、TS 编码器、Go2 rollout storage、GAE 和有限 rollout runner；各模块已完成小 batch 前向、loss 和 storage smoke 验证。
- 源项目 14 个任务现在均有对应的 mjlab 注册入口；CTS/AMP/DreamWaQ/TS 入口使用 rough-terrain 的 45 维 actor 与源项目对应的 233/278/281/309/783 维 critic 输入契约，均已完成一次有限步 reset/step 验证。
- 当前 registry 可枚举 16 个 Go2 入口：上述源项目 14 个任务，加上 `Mjlab-Velocity-Flat/Rough-Unitree-Go2` 两个公共基线；注册数量检查仅用于入口完整性，不替代逐项行为验收。
- 六个 teacher/custom 任务已切换到当前 RSL-RL 的 `Go2AuxiliaryPPO` 子类：CTS/AMP-CTS
  执行 latent distillation，DreamWaQ/AMP-DreamWaQ 执行 VAE+KL，TS/AMP-TS 执行
  teacher policy PPO，AMP teacher/custom 变体额外执行 discriminator update；两个 TS-Student
  入口使用独立的纯行为蒸馏 runner，不执行 PPO 或 AMP discriminator update。
- 自定义 actor conditioning 已接入 RSL-RL actor：CTS 使用 privileged teacher latent，DreamWaQ 使用 history VAE latent，TS/TS-Student 使用 terrain/privileged encoder 或 history LSTM；CTS、DreamWaQ、TS-Student 均已完成一次最小 runner 学习迭代验证。
- CTS、DreamWaQ、AMP-TS 的 actor 拼接顺序已统一为源项目的 `latent || current_observation`；普通 PPO 仍保持单一 actor observation。
- AMP 动捕管线已新增 `Go2MotionLoader`：读取源项目 `Frames/FrameDuration/MotionWeight` JSON 轨迹，归一化根部四元数并采样成对的 31 维 AMP 状态；AMP 任务新增独立 `amp` observation group，设置 `GO2_MOTION_DIR` 后判别器使用真实动捕 expert transition。
- `Go2MotionLoader` 已提供按关节名称生成 permutation 的工具，并可对 12 关节、四足局部位置和关节速度同步重排；默认源项目与 MJCF 均使用 `FL, FR, RL, RR` 的三关节分组顺序。
- AMP 判别器已按源项目改为 transition discriminator：输入为当前/下一帧 AMP 状态拼接（`31×2=62`），动捕 loader 提供 expert pair，rollout 的 AMP 状态提供 policy pair。
- AMP 判别器隐藏层容量已恢复源配置 `1024→512`，适用于 AMP、AMP-CTS、AMP-DreamWaQ 和 AMP-TS 组合任务。
- 新增 `Go2AmpReplayBuffer` 环形 replay buffer，AMP update 会跨 PPO iteration 保存并抽样 policy transition，再与 motion loader 的 expert transition 配对训练。
- AMP 更新已补齐源项目的运行均值/方差归一化、`[-10, 10]` 裁剪、LeakyReLU + LSGAN（expert=`+1`、policy=`-1`）、梯度惩罚和 `0.02×amp_reward_coef` 的策略奖励整形；判别器、归一化统计和 replay buffer 均随 checkpoint 保存。
- AMP rollout transition 现在按 storage 的 `[time, env]` 轴配对并跳过 done 边界，不再用展平后的 `roll()` 把不同环境或不同 episode 拼成伪 transition；没有动捕目录时仍仅保留明确的 smoke fallback。已用源目录 13 条动捕轨迹做一次真实 expert-pair 更新验证。
- AMP 观测的基座高度已优先使用 MuJoCo 地形扫描恢复源项目 `_get_base_heights()` 的地形相对均值，平面或缺少扫描传感器时回退到根部高度。
- Trot/Jump/Spring-Jump 的特权 contact mask 已使用源项目的足端垂向力阈值 `5`，CTS/TS/站立任务使用阈值 `1`，并保留无力场传感器时的 found-bit 回退。
- DreamWaQ VAE、CTS teacher encoder 及 TS teacher/student encoder 已与实际 RSL-RL actor 共享对应模块；辅助蒸馏/重建更新会作用于最终 PPO/ONNX 使用的策略参数。
- DreamWaQ/AMP-DreamWaQ 已增加 3 维 body-frame 线速度标签组，VAE 辅助更新同时计算显式速度估计、当前观测重建和 KL 损失，与源项目的 `vel_buf` 监督保持一致。
- DreamWaQ VAE 的输入已对齐源项目：5×45 历史的最新 45 维作为 actor 当前帧，VAE 仅编码前 4 帧（180 维）；训练、推理和 ONNX wrapper 使用同一切分。
- 自定义 actor 已提供多输入 ONNX wrapper：统一输入为 `actor` 与 `conditional`，CTS/DreamWaQ/TS-Student 均传 history，TS teacher 传 `terrain || privileged`；各接口已完成 CPU ONNX 导出和前向验证。
- CTS 的训练 actor 仍按 3:1 混合 privileged teacher/history student，但回放和 ONNX 已修正为
  源项目一致的纯 student 路径；导出条件输入从错误的 233 维 teacher privileged 改为 225 维
  五帧历史，2 batch ONNX Runtime 前向已通过。
- 新增 `Go2HistoryBuffer`、`Go2OnnxPolicy` 和 `Go2DeploymentAdapter`，明确五帧历史的 oldest→newest 拼接、reset 回填和 CTS/DreamWaQ/TS teacher/student 的条件输入；已用导出的 AMP-CTS ONNX 文件完成一次适配器前向。
- 架构收口后上述 Go2 专用适配器由 `robots/unitree/go2/deploy/policy.py` 拥有，
  Go2 checkpoint/导出/student runner 位于 `go2/training/runner.py`；通用 `learning` 与
  `runtime` 通过 AST 契约禁止反向导入机器人领域。
- Handstand/Leggedstand 已加入源任务目标重力、目标基座高度、指定足端离地和单支撑接触奖励；Spring-Jump/Backflip 已加入源项目的基座高度下限终止条件。
- 五个特殊动作任务已移除通用平地配置中 70° 姿态的 `fell_over` 终止，改用源项目的躯干接触终止（`trunk_ground_touch`）；这样倒立、跳跃和翻转达到目标姿态时不会被通用姿态阈值提前截断。
- Jump 的静止基座高度核改为源项目的 `0.30 m` 正奖励；Spring-Jump 已加入起跳上升速度、全足离地、飞行高度和落地站姿高度项；Backflip 已加入上升速度、俯仰角速度、飞行/落地高度、姿态和左右关节对称项。
- Jump/Spring-Jump/Backflip 与八个自定义任务的偏航角速度采样范围已对齐源项目的 `[-1, 1]` rad/s；站立任务保留各自的 `[-0.4, 0.4]` 范围。
- Jump 已切换到源项目一致的向上楼梯地形生成器（`8×8 m`、`10×20` 网格、步宽 `0.31 m`、平台 `3 m`、步高 `0–0.1 m`），Spring-Jump/Backflip 继续使用源项目的平面场景。
- 八个 custom 粗糙地形任务已切换为源项目的五类地形比例（平滑坡、随机粗糙、上/下楼梯、离散障碍 `0.15/0.15/0.30/0.30/0.10`），并按源配置设置 CTS/AMP-CTS 的 `70 m` 边界、其他 custom 的 `25 m` 边界；公共 rough 基线仍保留 mjlab 默认 preset。
- Spring-Jump/Backflip 已加入 `Go2TriggeredCommand`：两者均在源项目一致的 `50–60` 控制步触发第三 command 分量，并维护 `triggered`、`was_in_flight`、`has_jumped`、起始/着陆位置、最大高度和最大俯仰角速度；相关 critic flag、起跳/腾空/落地高度、着陆位置和姿态奖励已接入。
- Spring-Jump/Backflip 的触发命令已接入源项目的一次性起跳推力：按 0.8 初始概率写入线速度（Spring `1.5–2.2`、Backflip `2.0–3.5` m/s），Backflip 另写入 `2.0–2.5` rad/s 的俯仰角速度。
- Spring-Jump/Backflip 已补充源配置中的 pre-trigger 姿态、起跳后平面速度、关节/髋关节、角速度、关节速度/力矩、足端接触力和 source flight linear-velocity 奖励项；通用速度任务中无源对应的 tracking/pose 项在这两个任务中禁用。
- Spring-Jump 的 `flight` 已改为读取 `Go2TriggeredCommand.was_in_flight` 的持久腾空状态（权重 2.0），不再误用周期接触相位奖励；CTS/DreamWaQ/TS 自定义任务已补充源项目的 stumble 接触力比惩罚和对应 collision 聚合项。
- Spring-Jump/Backflip 的触发状态现在使用足端垂向力 `>1 N` 的 contact filter，与源项目 `check_jump()` 一致；观测中的特权 contact mask 仍按各任务源配置分别使用 `>5 N` 或 `>1 N`。
- Trot 使用 1–3 子步动作延迟缓冲，Jump 使用 1–3 子步 reset 采样延迟；Spring-Jump、Backflip、Handstand、Leggedstand 和八个自定义任务使用源项目的 decimation 内随机切换动作语义。
- Trot、Jump、Spring-Jump、Backflip、Leggedstand 已增加源项目的 1–3 子步 motor/IMU 观测延迟缓冲；缓冲在每个 MuJoCo 子步积分后更新，policy observation 读取各环境 reset 时采样的延迟索引，Handstand/自定义任务保持源配置未启用的观测延迟。
- Trot 的 reset 姿态已覆盖为源项目的对称站姿（四髋 `0`、四大腿 `0.8`）；Jump/翻转/站立任务继续使用各自源配置的默认姿态。
- Trot/Jump/站立类接触奖励现在优先读取 ContactSensor 的实际力：运动步态使用足端垂向力阈值 `5`，站立支撑使用 `1`，无命令四脚接触使用合力阈值 `0.1`；只有传感器没有力字段时才回退 found-bit。
- Jump 的 70 维特权帧末尾已从占位零值改为读取当前 MuJoCo 的摩擦系数和 `base_mass / 10`，与源项目的 `env_frictions`、`body_mass / 10` 字段对应。
- 源特权观测中的四足接触字段已从 mjlab 传感器顺序 `(FR, FL, RR, RL)` 显式重排为源项目顺序 `(FL, FR, RL, RR)`；Trot 相位奖励同步修正为对应的对角支撑模式。
- CTS/AMP-CTS 已增加 3:1 teacher/student 环境掩码；AMP-CTS 开启源项目的正奖励裁剪，八个自定义任务已接入源项目的 startup 级 encoder/COM/PD/质量/摩擦与间隔推力随机化；仅站立任务额外保留其关节阻尼、摩擦和 armature 随机化。
- DreamWaQ/CTS 特权观测中的源项目随机化字段已从固定占位改为读取当前 MuJoCo 批量模型（摩擦、质量/COM、PD 增益等），在未提供对应模型字段时才回退为零。
- Handstand/Leggedstand 的 34 维随机化特权块也改为读取当前模型的质量、COM、PD、阻尼、摩擦损失和 armature 字段；源任务未启用的 restitution 字段保留零值。
- TS/AMP-TS 已将源项目的 `privileged_buf`（70 维域随机字段 + 4 维足接触 = 74 维）与 309 维 critic 输入分离；教师 encoder 使用 74 维，critic 仍保留 actor、域随机、接触、地形和线速度的完整拼接。
- TS critic 的拼接顺序已固定为源项目的 `base_lin_vel(3) || actor(45) || domain(70) || contact(4) || terrain(187)`，避免仅维度一致而字段位置错位。
- 部署适配器已对条件输入做维度校验：CTS teacher 使用 233 维特权输入，TS teacher 使用 `terrain(187) || privileged(74)` 的 261 维输入，student/history 仍使用 5×45 历史。
- 新增 `Go2RecurrentOnnxPolicy`，封装源 TS-Student 的 `obs(45), h, c → actions, h_next, c_next` ONNX 契约；hidden/cell 由部署循环显式保存并可在 episode reset 时清零。
- `StudentActorModel` 的 student LSTM 已采用源项目的三层 `45→256` 结构和 `256→256→128→32` 编码头，并提供 `as_recurrent_onnx()`；velocity runner 在保留标准双输入 ONNX 的同时，会为 TS-Student 额外导出 `*_recurrent.onnx`（`obs,h,c → actions,he,ce`）伴随文件，已用 CPU ONNX Runtime 验证输出形状。
- TS-Student runner 支持通过 `GO2_TS_TEACHER_CHECKPOINT=/path/to/go2_ts/model.pt` 加载并
  冻结 teacher 的 terrain/privileged encoder 和 actor，并用 teacher actor 初始化 student actor；
  未设置时仅保留有限 smoke 用的随机 teacher 回退，并明确打印警告，不将其误认为有效蒸馏训练。
- `Go2HistoryBuffer` 的 reset/首帧行为已对齐源项目 deque：策略调用先读取更新前历史，随后才把当前帧写入 newest 槽位；首帧之前保持零，后续按 oldest→newest 滚动，避免部署初始动作因重复首帧而偏移。
- `Go2TriggeredCommand` 与 `go2_commands.py` 通过 `config/go2` → `tasks/velocity/mdp` 的显式依赖接入；没有新增独立任务目录或跨目录隐式注册。
- 已用 4 步/环境的短跑验证标准 PPO 不产生 NaN，并完成 TS-Student checkpoint 保存、ONNX 导出和重新加载；单步/环境训练的优势归一化 NaN 属于零方差测试规模，不作为有效训练配置。
- CTS teacher 与 TS-Student 的 ONNX 文件已用 `onnxruntime` 实际加载执行，和 PyTorch wrapper 输出最大误差约 `4.5e-8`（CTS）/`1.9e-8`（TS-Student）。
- 已用当前 `VelocityOnPolicyRunner` 完成一次真实 CLI 级 CTS PPO+蒸馏迭代（2 环境、4 步），并完成一次 AMP-CTS PPO+蒸馏+判别器迭代；AMP-CTS 使用源动捕目录成功采样 expert pair，未出现 shape/NaN 错误。
- 已用当前 `VelocityOnPolicyRunner` 完成一次真实 CLI 级 DreamWaQ PPO+VAE/KL 辅助迭代
  （2 环境、4 步）；TS-Student 随后改由 `VelocityDistillationRunner` 加载并冻结 TS teacher，
  按源项目同时最小化 latent/action 的 L2 范数，不再混入 PPO 梯度。
- TS-Student CLI 短跑同时生成标准双输入 ONNX 与源部署契约的 `*_recurrent.onnx`（`obs,h,c → actions,he,ce`）伴随文件；两者均在 CPU 环境完成导出，递归文件已用 ONNX Runtime 检查 batch=1 的输出形状。
- `Go2RecurrentDeploymentAdapter` 已接入部署包，负责 TS-Student 三层 LSTM hidden/cell 的跨步保存、全量/按环境 reset 和 `obs(45) → actions(12)` 调用；已用导出的递归 ONNX 做两次连续推理及 reset 检查。
- `Go2OnnxPolicy`/`Go2RecurrentOnnxPolicy` 已增加输入、输出名称及静态维度校验：普通策略固定 `actor(45)`、`actions(12)`，条件输入宽度按导出文件检查；运行时同时校验 batch 和动作输出形状，避免部署时静默使用错版本文件。
- 自定义 ONNX 已加入 `go2_policy_contract_version=1` 和明确的 mode/维度/history/reset metadata；
  conditional 与递归导出均使用动态 batch。CTS student、DreamWaQ、TS teacher、TS student 和
  TS student recurrent 五种接口均从新导出文件以 batch=2 重新加载前向通过，部署 adapter 会拒绝
  mode 不匹配的文件。
- Handstand/Leggedstand 的站立奖励已进一步按源索引校准：`feet_clearance`/`feet_air_time` 使用 `contact_foot` 支撑腿，手部高度核只读取 `feet_name_reward` 的腾空腿，Leggedstand 对称关节只比较后侧支撑对；两个任务均完成一次 1-step reset/step smoke。
- 站立任务的足端 clearance 已恢复源实现的 `-0.02 m` 足半径偏移；Leggedstand 的 `feet_air_time` 已改为仅在首次接触帧计算 `last_air_time - 0.4`，并通过 Handstand/Leggedstand 各一次 1-step smoke。
- Handstand/Leggedstand 的 readiness gate 已按源实现改为批量平均 base-height reward 的统一开关；目标高度分别使用源系数 `5`/`10`，并通过 2 环境各一次 1-step smoke。
- Trot 的速度/偏航跟踪已恢复源项目的 gait-readiness 门控：移动命令使用批量对角步态比例 `>0.7`，静止命令使用四足接触条件；并通过 2 环境 2-step smoke。
- Trot 相位接触核已修正 `phase == 0.5` 的边界，严格使用源项目的 `phase < 0.5` 与 `phase > 0.5` 两个互补条件；1-step smoke 通过。
- AMP-DreamWaQ、AMP-TS 已在源动捕目录下完成一次真实 CLI 短迭代（2 环境、4 步），
  分别确认 VAE+AMP、teacher PPO+AMP 更新链路可运行；AMP-TS-Student 与 TS-Student
  共用源项目的纯递归蒸馏流程，均不读取动捕或更新 AMP discriminator。
- Jump 的源奖励核已进一步对齐：恢复源项目的移动/静止线速度与偏航跟踪分支，修正足端 clearance 的 `-0.02 m` 偏移和 `[0, 0.05]` 裁剪，增加基于 `compute_first_contact` 的 `last_air_time−0.5` 着地奖励，并关闭源配置未启用的关节位置限制项、将 action-rate 权重设为 `-0.01`；Jump 仍通过 470/210 观测契约 smoke。
- DreamWaQ、AMP-TS、AMP-TS-Student 的 CLI 导出文件已由 ONNX Runtime 实际加载；分别通过 `dreamwaq` 历史输入、`ts_teacher` 的 `terrain||privileged` 输入和 `ts_student` 历史输入完成适配器前向，输出均为 12 维动作。
- Trot 的源奖励核已进一步对齐：关闭未在源配置中启用的 joint-limit/二阶 smoothness 项，将 action-rate 改为 `-0.01`，并按源项目的对角腿组、`-0.02 m` 足端偏移、正弦目标高度和互补相位门控重写 clearance；Trot 470/204 观测契约 smoke 通过。
- Spring-Jump 的着陆位置奖励已增加源项目的成功门槛（`has_jumped`、最大高度 `>0.42 m`、姿态误差 `<0.6`），并将源配置中的 joint-limit `-10` 与 action-rate `-0.01` 权重恢复；触发状态、飞行/落地奖励仍通过 `Go2TriggeredCommand` 管理。
- Spring-Jump 腾空足端 clearance 的批量四元数变换已修正为按足端展开后扁平化计算；此前 2 环境 smoke 暴露的 `[B]`/`[B×足数]` shape 错误已消除，2-step smoke 和 velocity 测试均通过。
- Handstand/Leggedstand 的站立奖励已进一步恢复源项目语义：roll-rate yaw 跟踪按两类姿态分别使用相反符号，低横滚/偏航速度采用源项目的正向指数奖励，摆腿 clearance 使用两条腿互补相位；两个任务均重新完成 1-step smoke。
- Handstand/Leggedstand 的站立 reward scale 已补齐源项目的关节加速度正则 `dof_acc=-2.5e-7`；`dof_pos_limits=-2` 仅在 Handstand 启用，Leggedstand 保持关闭，并分别完成 1-step smoke。
- Backflip 已按源 reward scale 关闭继承的 `inverted_orientation` 项，并恢复 `dof_pos_limits=-10`、`action_rate=-0.01`；projected-gravity IMU/特权线速度缩放修正后保持 2-step smoke 通过。
- 各任务的 interval push 已按源项目重新设置：Trot/Jump/Spring-Jump/Backflip 为 4 s，Handstand/Leggedstand 为 8 s，自定义 TS 系列为 8 s；水平线速度与角速度范围已对齐，z 方向推力关闭。
- 自定义 reward scale 已区分 TS 的嵌套配置继承语义：AMP-TS/TS teacher 替换了
  `scales` 类，因此关闭基础 `smoothness`、`foot_clearance` 与 `feet_air_time`；两个
  student 显式继承基础 `scales`，保留 `smoothness=-0.005`、`foot_clearance=-0.01`
  和 `feet_air_time=1`，但不额外启用 `stumble`。`dof_pos_limits=-2` 仅保留在源配置
  实际声明的 CTS、DreamWaQ、AMP-CTS、AMP-DreamWaQ 和 AMP-TS teacher 中；配置检查通过。
- 自定义命令范围已按 8 个源配置逐项拆分：CTS/DreamWaQ 的横向范围为 ±1，AMP-CTS 为 ±0.65，AMP-DreamWaQ/AMP-TS/TS 为 ±0.6，两个 student 为 ±0.5；TS 系列 yaw/heading 范围为 ±π，其余为 ±1，重采样周期保持 8/10 s。
- Trot/Jump 已补回源配置中遗漏的 torque penalty：Trot 使用平方力 `-0.0001`，Jump 使用源实现的绝对力 `-0.0002`，并通过配置导入检查确认。
- Trot 的偏航命令范围已修正为源项目的 ±1 rad/s（此前 flat 公共配置训练态仍继承 ±0.5），并与 Jump 一起完成 1-step actor/critic shape smoke。
- 六个标准动作任务的启动级随机化已接入源范围：摩擦、encoder bias、base/link 质量、COM 和 PD 增益；Handstand/Leggedstand 另启用各自的关节 friction/damping/armature 随机化，Trot/Jump/Spring-Jump/Handstand smoke 已验证事件可执行。
- Trot/Jump 使用源项目的 `default_dof_pos + U(-0.1, 0.1)` reset；Spring-Jump 使用默认关节姿态；Backflip 使用 `default_dof_pos×U(0.5,1.25)`；Handstand/Leggedstand 使用 `default_dof_pos×U(0.5,1.5)`；CTS 使用 `default_dof_pos×U(0.9,1.1)`，其余 custom 使用 `default_dof_pos×U(0.5,1.5)`，并分别通过对应 reset 事件接入。
- 标准 flat 任务的 reset root pose 已固定为源项目初始 x/y/z/yaw；rough custom 任务在 terrain origin 周围采用源项目的 x/y ±1 m、z/yaw 固定分布；Trot 与 CTS smoke 已复核。
- 已按源 `commands.curriculum` 开关修正 curriculum：Spring-Jump、Backflip、Handstand、Leggedstand 及 TS/AMP-TS student 移除公共 `command_vel` curriculum，Trot/Jump 与 teacher/custom 任务保留。
- custom observation noise 已按源配置区分：DreamWaQ 关节位置噪声 `0.02`，TS/AMP-TS student 为 `0.03` 且角速度噪声 `0.3`，其余 custom 保持 `0.01/0.2`；配置值检查通过。
- 已补齐源项目的关节/root reset 分布：Spring-Jump 使用默认关节姿态，Backflip 使用 `default_q×U(0.5,1.25)`，Handstand/Leggedstand 使用 `default_q×U(0.5,1.5)`，CTS 使用 `default_q×U(0.9,1.1)`，其余 custom 使用 `default_q×U(0.5,1.5)`；站立和 custom 任务的 reset root 6D 速度均使用 `U(-0.5,0.5)`，并通过独立 `reset_joints_by_scale` 事件接入。
- CTS 的 `hip_pos` 已改为源项目四髋关节误差平方和；AMP-DreamWaQ 已补充源项目的 rear-hip 越界惩罚（`|q|>0.4`）。
- custom 任务的 torque-multiplier 已通过 MuJoCo position actuator 的 `gainprm/biasprm` 接入：CTS/DreamWaQ/AMP 变体使用 `U(0.9,1.1)`，TS 变体使用 `U(0.8,1.2)`，每个目标共享同一系数并在独立 PD-gain 随机化之后执行。
- DreamWaQ VAE、actor 与确定性 ONNX wrapper 的 code 顺序已恢复为源项目的
  `code_vel(3) || code_latent(16) || current_obs(45)`，decoder 和策略条件输入保持一致，
  避免仅维度相同但语义错位。
- TS/AMP-TS student runner 已恢复源配置的 `50` steps/env rollout，并将 action
  standard deviation 下限改为源配置的逐关节值 `0.05×(关节上限−关节下限)`；该约束也用于
  AMP-CTS、AMP-DreamWaQ、AMP-TS 和 TS teacher，并已通过数值下限检查。
- 两个 TS-Student 注册入口现使用专用 `VelocityDistillationRunner`：三层 LSTM hidden/cell
  跨 50 个采样步持续传递，episode reset 时按环境清零；第 0 轮由 teacher 驱动，后续由
  student 驱动；每轮联合优化 teacher/student 的 32 维 latent 与 12 维确定性动作误差。
- 源项目策略没有额外的运行观测归一化层；14 个源任务的 actor/critic 已改为直接消费
  环境中完成 scale/noise/clip 的观测。旧的、带归一化统计的 teacher checkpoint 仍可作为
  student 蒸馏源加载，runner 会保留该 checkpoint 自身的输入变换。
- 14 个源任务已按各自源配置恢复训练 learning rate、seed、默认最大 iteration、保存间隔，
  并统一使用源策略的 `clip_actions=100`；公共 Velocity 基线仍保留 mjlab 自身默认配置。
- Trot、Jump、Spring-Jump、Handstand 已接回源 PPO 的左右镜像约束：沿用各任务原始的
  signed observation/action permutation，按 10×47 或 45 维契约生成镜像，并通过当前 RSL-RL
  mirror loss（系数 `1`）训练；Backflip 与 Leggedstand 按源开关保持关闭。三套变换均通过
  两次镜像还原检查，Trot/Jump/Handstand CLI 短训均记录到非零 symmetry loss。
- Jump 已恢复源项目的楼梯 terrain curriculum、地形原点附近 x/y `±1 m` reset；所有启用
  `dof_acc` 的任务使用源项目跨 policy step 的关节速度有限差分，Jump 保留其不除 `dt` 的
  特例。Jump 的角速度核也恢复为源实现的二维 L2 norm。
- 特殊动作与 custom 任务的 aggregate collision 已按各源 `penalize_contacts_on` 区分
  thigh/calf/base；base 接触仍独立用于 episode termination，不再把所有自碰撞或终止接触
  无条件重复计入 collision reward。
- Backflip 已恢复源代码在飞行阶段叠加基础项和 `3×` 额外 pitch-rate 项（合计 `4×`），
  并按源实现对全三维 body linear velocity 做全程惩罚；Spring-Jump 的着陆姿态 gate 保留
  源表达式的 signed Euler sum 语义。
- TS teacher 的 187 维地形输入已恢复为源公式
  `clip(base_z-terrain_z-0.5,-1,1)×5`，不再误用公共 height-scan 的 `×0.2` 缩放；运行时检查
  已确认独立 terrain group 与 critic 尾部 187 维逐元素相同，最大误差为 `0`。
- DreamWaQ 辅助更新已恢复源 VAE 目标：显式速度误差、当前观测重建误差和随机 latent 的
  KL（权重 `1`），辅助优化器 learning rate 为 `1e-3`，并按 PPO epoch/mini-batch 次数更新；
  重建目标读取 critic 最新特权帧末尾的无噪声 45 维状态，done 样本按源 `live_batch` 屏蔽，
  KL 对 16 个 latent 维先求和再做 batch mean。
- DreamWaQ CENet encoder 已恢复第二个线性层后的 ELU；VAE 参数已从 PPO optimizer 中剔除，
  只由源项目独立的 `1e-3` velocity/reconstruction/KL optimizer 更新。修正后 2 环境×4 步×1
  iteration 的真实 DreamWaQ CLI 更新通过。
- CTS actor/critic 已共享同一组 teacher/student encoder，critic value 路径对 latent 停止梯度；
  teacher/student 的 PPO surrogate 按源项目分别求均值的语义做 3:1 分组权重，蒸馏更新只抽取
  student 环境；更新顺序也恢复为先完成 PPO、再训练 student encoder，避免先蒸馏后改变 rollout
  action-conditioning latent。最终 CTS CLI 短迭代已通过。
- CTS/DreamWaQ 的环境历史组内部保留“前 5 帧 + 当前帧”，模型在编码前丢弃当前帧，因而
  encoder 实际输入仍为源项目的前 5 帧（225 维）；DreamWaQ 导出接口继续只暴露 225 维源历史。
- 观测历史 reset 语义已加入显式的 `history_fill="zero"`：Trot/Jump/Spring-Jump/Backflip
  的 actor/critic 历史，以及 CTS/DreamWaQ/custom history 均在 episode 开始时以前部零帧、
  newest 当前帧初始化，不再沿用 mjlab 默认的“把首帧复制到全部槽位”；公共 velocity 任务仍
  保持原默认行为。custom history 还会直接复用 actor 已处理帧，确保历史中的 noise realization
  与当时策略实际收到的观测逐元素相同；DreamWaQ 两个连续步的对应最大误差均为 `0`，环形缓冲
  与观测历史共 32 项定向测试通过。
- AMP 判别器更新已恢复源项目与对应 PPO 共享的 learning rate（AMP-CTS/AMP-DreamWaQ 为
  `1e-3`，AMP-TS 为 `1e-5`）、trunk/head weight decay、PPO
  epoch/mini-batch 更新次数及每个 mini-batch 的 policy/expert 归一化统计更新。
- AMP 判别器现在在首轮 rollout 前创建，因此第 0 轮也使用源项目的随机判别器奖励；replay
  直接记录每个真实 `current→next` transition（含末步和 terminal state 替换），不再通过平移
  rollout 丢失末步或跨 episode 配对。归一化统计在当前 mini-batch 更新完成后再写入。
- AMP expert loader 已固定使用源环境 policy step 的 `0.02 s` transition 间隔，并恢复源 loader
  的 `p×N` 帧索引和 `duration×U-step-frame_duration` 尾部采样公式；13 条源轨迹的 64 组
  `31→31` 状态对已检查为有限值，定点插值与源公式最大误差为 `0`。
- AMP 组合算法的 policy transition replay capacity 已恢复为源配置的 `1,000,000`；
  buffer 仍在首次有效 transition 后惰性分配，普通 PPO 和环境初始化不会额外占用该内存。
- 源任务的刚体摩擦随机化已改为对每个环境全部 Go2 碰撞几何体共享一个 MuJoCo
  `geom_friction[0]` 样本，并按任务恢复 `0.2–1.2`、`0.3–1.0`、`0.2–1.25`、
  `0.2–1.0` 或 `0.05–3.0` 范围；已移除源项目未使用的扭转/滚动摩擦随机事件。
- Handstand 及八个 custom 任务已固定为源项目的全量 heading 命令（`rel_heading_envs=1`），并关闭 mjlab 公共模板的 standing/forward 混合采样。
- 八个 custom 任务的 episode termination 已统一恢复为 source 的 base 接触；AMP-CTS 与两个
  student 的落脚计时分别使用 `0.5`/`0.3` 秒 offset，足端 clearance 使用 source 的 body-frame
  相对足高与横向速度公式及各变体目标高度，不再沿用公共 rough task 的近似 kernel。
- custom 特权观测已把 PD `kp`、`kd` 与完整 torque multiplier 作为三个独立随机字段保存，
  避免从最终 MuJoCo gain 反推时把它们相乘混淆；TS 的 28 维 URDF link-mass block 按源槽位
  放置 12 个可表示的活动连杆，MuJoCo 中已折叠的固定连杆使用中性倍率 `1`。
- 标准任务的命令重采样已接入源项目的全零/仅线速度归零概率和小速度阈值：Trot/Jump/Spring-Jump/Backflip 为 `0.05/0.05/0.1`，Handstand/Leggedstand 为 `0.20/0.10/0.1`；CTS 采用 `0.1`、其余 custom 采用 `0.2` 的线速度阈值；TS 四个变体另接入源回调中的 `0.05/0.05` 命令掩码。
- 新增 `mjlab/scripts/run_go2_validation.sh`，按源任务顺序执行约 `1024` 环境、`1000`
  iteration 的训练验证；粗糙地形 custom 任务默认使用 `sap_segmented` broadphase
  与 35 contacts/world，避免 1024 环境初始化时的 Warp 数 GiB 碰撞缓冲分配。AMP 任务
  默认读取源 `datasets/mocap_motions_go2`，TS-Student 变体在统一验证日志目录中自动查找
  对应的 AMP-TS/TS teacher checkpoint（训练顺序也按两类 teacher 分开）。

## 14. 验收结果与边界

代码迁移、任务入口和必要的有限验证已经完成；以下边界用于区分“迁移完成”和后续性能调参：

1. 六个标准动作的源可见状态、奖励、终止、触发和 reset 语义已逐项映射，并均完成契约
   step 与 1024×1000 训练验证。MuJoCo-Warp 与 PhysX 的接触求解不同，源项目 restitution
   随机化和少数接触数值边界不能表述为逐点物理等价；这属于跨引擎校准边界，不是缺失任务逻辑。
2. AMP、AMP-CTS、AMP-DreamWaQ、AMP-TS 及其组合已经分别完成真实动捕目录下的 CLI
   短迭代验收；这些验证证明更新链路可运行，不等同于收敛质量或 sim-to-sim 性能结论。
3. 自定义算法的 ONNX/部署输入输出现已固定为版本 1 契约并完成五种实际接口的重新加载前向；
   AMP 组合复用相同 actor 契约。Policy Bundle runtime 已在独立 mjlab 环境完成 50 Hz、
   2 环境×8 tick 的连续控制；更长时间的行为和数值稳定性仍属于后续性能评估。
4. TS-Student 的三层 LSTM 已能训练、导出，并由 Bundle runtime 在连续控制中保存 hidden/cell、
   按环境选择性清零。Go2 `physics_dt=0.005`、`control_dt=0.02` 的频率契约已校验；真实 Unitree
   SDK、关节映射和硬件安全闭环仍是独立验收范围。
5. 当前验证严格遵循“少量 smoke、不过度测试、无需哈希检查”：`ruff`、契约检查和必要的
   runner smoke 已通过；最终一轮 `lainloco validate contracts` 为 14/14 通过，14 个任务已有一次
   1024 环境、1000 iteration 完整记录。该长训发生在
   后续源语义复核之前，因此只证明当时的端到端稳定性；最新修正使用契约检查、1024 环境容量
   单迭代和对应算法短迭代验证，不把旧长训曲线表述为最新实现的收敛结果，也不做硬件闭环测试。

当前 custom 任务已经接入 PPO+辅助更新钩子、策略条件输入、主要随机化、动捕和 teacher
checkpoint 通路。后续若要追求源视频级动作质量，应在当前实现上针对 MuJoCo 接触参数和奖励
曲线继续调参；目录注册、基础调用链、算法迁移及本次约定的训练验收均已闭合。

最近一次源代码对照还修正了三处特殊动作细节：Jump 的相位接触/摆脚奖励恢复为半周期
阈值 `0.5`；Spring-Jump/BackFlip 的起跳竖直速度改用世界坐标系 Z 分量；BackFlip 的
触发窗口恢复为源项目的 50–60 个控制步，并在离地后补上一次性俯仰冲量及随训练步数
衰减的冲量概率。修正后 14 个任务的观测/动作/critic 契约检查仍全部通过，BackFlip
另完成了 4 环境、80 步的无界面触发路径 smoke。

Handstand/Leggedstand 的奖励表也已按源配置去重：删除重复的接触与目标姿态别名，关闭
通用模板遗留的足端 clearance、swing-height、stand-still 和 contact-force 项；
Leggedstand 不再错误启用 Handstand 专属的 orientation/foot-height symmetry 与 alive
奖励。两个站立入口均通过 1 环境、4 步的无界面 zero-action smoke。

`scripts/run_go2_validation.sh` 曾完成一轮 `1024×1000` 批量训练：14 个注册任务（六个
标准动作、DreamWaQ/AMP-DreamWaQ、CTS/AMP-CTS、AMP-TS/TS，以及 AMP-TS-Student/
TS-Student）均以退出码 `0` 结束，并生成对应的 `model_999.pt`。student
启动过程中发现的旧 teacher 别名选择和训练期 ONNX 导出显存峰值问题已修正；批量验证默认
跳过训练期 ONNX 导出，独立 ONNX smoke 仍保留。状态明细记录在
`mjlab/logs/go2_validation/validation_status.tsv`，完整运行输出在
`mjlab/logs/go2_validation/batch_stdout.log`；状态文件中保留的早期失败记录均已被后续成功
运行覆盖，不代表当前实现失败。源代码复核后，两个 student 已从 PPO+辅助 loss 改为源项目
专用的纯递归行为蒸馏 runner；1024 环境、50 steps/env 的单迭代容量验证已通过。曾启动的
student 定向长训在发现 teacher terrain 缩放不一致后主动停止，状态文件中的 `exit=130` 是该次
人工中止记录，不是崩溃。修正后的 CTS、DreamWaQ、AMP-TS 和 TS-Student 更新链路均已完成
必要短测；依据“不要过度测试”的约束，不因每次语义修正自动重复整套 14×1000 长训。

该轮 1024×1000 日志中，14 个任务从 iteration 0 到 999 的末尾 mean reward 均有改善，
可作为用户要求的“简单效果验证”，但不解释为全部动作已经达到源视频质量：

| 任务 | iter 0 reward | iter 999 reward | iter 999 episode length |
|---|---:|---:|---:|
| Trot | -0.79 | 26.89 | 910.99 |
| Jump | -380.03 | -6.93 | 12.99 |
| Spring-Jump | 3.88 | 48.96 | 220.95 |
| Backflip | 39.34 | 194.98 | 176.91 |
| Handstand | -1.36 | -0.48 | 12.65 |
| Leggedstand | -1.21 | -0.26 | 12.83 |
| DreamWaQ | -0.89 | 6.00 | 644.94 |
| AMP-DreamWaQ | -1.26 | 10.50 | 996.02 |
| CTS | -0.87 | 5.35 | 569.47 |
| AMP-CTS | 0.10 | 21.92 | 837.45 |
| AMP-TS | -1.04 | 5.25 | 758.23 |
| TS | -0.95 | 2.96 | 503.84 |
| AMP-TS-Student | -0.56 | 7.47 | 889.22 |
| TS-Student | -0.56 | 0.97 | 70.10 |

其中 Jump、Handstand、Leggedstand 在 1000 iteration 下 episode length 仍短，说明这些特殊
动作还需要更长训练或 MuJoCo 专项调参；它们的 reward 已改善且训练链路、checkpoint 与契约均
正常。本表数据来自语义复核前的完整长训，之后的修正只做了相应定向短测，未重复过度训练。
