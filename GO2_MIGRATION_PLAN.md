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

### 7.3 AMP

迁移动作参考数据加载、AMP observation、replay buffer、判别器和生成器更新。
动捕数据需要按 MuJoCo 的 joint/body 顺序重新整理，不能直接假定 Isaac Gym
的刚体索引仍然适用。

### 7.4 DreamWaQ

迁移 VAE/CENet、历史观测编码、显式速度估计、latent 重建和 KL loss。辅助
优化器、样本 mask、live 样本处理和 checkpoint 保存封装在自定义 algorithm 中。

### 7.5 AMP-TS、TS 和组合算法

先分别完成基础 AMP、CTS、DreamWaQ，再组合成 AMP-CTS、AMP-DreamWaQ 和
AMP-TS。TS student 需要单独处理 LSTM hidden state、蒸馏 runner 和 ONNX 导出；
纯 TS 不启用 AMP 判别器和动捕数据。

## 8. 观测、动作和 checkpoint 契约

建立固定契约表，至少包含：

- joint name 到 action index 的映射；
- 每个 observation term 的维度、顺序、scale、noise 和 clip；
- privileged/critic 额外字段及拼接顺序；
- action scale、default joint position、PD 参数和 torque limit；
- history frame 数、拼接方向和 reset 填充规则；
- student ONNX 输入维度及 hidden-state 处理方式。

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
6. 现有 MuJoCo sim-to-sim 或 Unitree 部署接口的输入输出语义保持不变。

