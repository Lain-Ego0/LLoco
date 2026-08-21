# 机器人快速适配规范

## 1. 适配目标

新增机器人不应再复制一整套 `deploy/robots/<robot>` 和任务代码。首版允许仅凭公开 XML/MJCF 建立 simulation-only Robot Profile；硬件成熟后再补 Driver、传感器、标定和安全实现，而不改变 Skill 或训练接口。

```text
Skill canonical joints
          │
          ▼
Robot Profile mapping
  ├── MJCF qpos/qvel/actuator indexes
  ├── policy observation/action indexes
  └── hardware motor IDs/signs/offsets
          │
          ▼
Simulation Adapter / Runtime Driver
```

## 2. Robot Profile 最小内容

```yaml
apiVersion: robolab.dev/v1alpha1
kind: RobotProfile

metadata:
  id: unitree.g1.29dof
  version: 1.0.0
  vendor: Unitree
  model: G1
  variant: 29dof

description:
  mjcf: model/g1.xml
  rootBody: pelvis
  feet: [left_ankle_roll_link, right_ankle_roll_link]
  imuFrame: imu_in_pelvis
  jointSet: g1.29dof.canonical.v1

control:
  mode: joint_position
  frequencyHz: 50
  actionScaleProfile: default
  stateEstimator: onboard

targets:
  simulation:
    type: mujoco
    enabled: true
  physical:
    enabled: false
    driver: null

safety:
  fallbackState: damping
  commandTimeoutMs: 100
  stateTimeoutMs: 100
  limitsRef: safety/limits.yaml
  gainsRef: safety/gains.yaml

capabilities:
  simulation: true
  training: true
  motorCommunication: false
  sensorStreaming: false
  calibration: false
  physicalDeployment: false

bindings:
  simulation: bindings/simulation.yaml
  hardware: bindings/hardware.yaml
  tasks: bindings/tasks.yaml
```

具体网络接口、IP 等机器相关值不进入版本化 Profile，放在本地 Device Instance 配置中。`RobotProfile` 描述型号，`RobotInstance` 描述某一台设备。只有 `physicalDeployment` 及其必需 capability 全部就绪时，WebUI 才能激活 physical target。

## 3. Simulation-only 快速接入

对于仓库已有或公开获得的 XML/MJCF，首版接入流程是：

1. 导入模型及 mesh，并记录原始 URL、revision、许可证和文件哈希；
2. 加载 MuJoCo，自动枚举 body、joint、actuator、sensor、keyframe 和 collision；
3. 生成 Robot Profile 草稿与模型诊断报告；
4. 映射 canonical joint、足端/末端、root body 和 IMU frame；
5. 完成默认姿态、关节 sweep、静态接触和执行器检查；
6. 绑定现有 velocity/tracking task，并生成 observation/action schema；
7. 达到 L2 后允许安装兼容 MotionSkill，达到 L3 后允许“一键仿真部署”。

没有真实机器人或厂商 SDK 不影响 L0-L3。模型来源和资产许可证必须随 Robot Profile 保存，不能因为 XML 可公开下载就假设 mesh、纹理和模型都可再分发。

## 4. 关节映射是核心契约

每个关节至少记录：

- canonical name；
- MJCF joint、qpos、qvel 和 actuator 名称/索引；
- policy observation/action 索引；
- hardware motor ID；
- 正负方向、零位 offset、减速比；
- position/velocity/torque 限制；
- 推荐与绝对最大 kp/kd；
- 所属语义组，如 left_leg、waist、right_arm。

适配器必须自动检查：名称唯一、映射双射、索引连续性、数组长度、限位方向、默认姿态合法性、动作维度和 ONNX 输入输出维度。任何隐式“第 N 个关节应该对应第 N 个电机”的假设都应被消除。

机器可读形态已冻结为 `JointSet v1alpha1`（`packages/schemas/src/robolab_schemas/data/joint_set.v1alpha1.schema.json`，2026-08-20 B1）：每关节一条记录，包含上列全部字段；`robolab check` 对 JointSet 执行上述自动检查（ONNX 维度实测除外，属于样板 Skill contract test），逐条输出可解释原因。

## 5. 适配分层

### A. Description

- 导入 MJCF（首选）或通过转换流程导入 URDF；
- 验证 mesh、惯量、关节范围、碰撞体、site、sensor 和 keyframe；
- 定义标准 frame、接触部位与 canonical joint set。

### B. Simulation

- 配置 actuator、armature、stiffness/damping、action scale；
- 配置初始姿态、碰撞策略、地形和域随机化边界；
- 运行模型加载、静态站立、关节 sweep 和接触测试。

### C. Training Binding

- 将 Robot Profile 绑定到稳定的 RoboLab `TaskDefinition`，由 backend 选择具体实现；
- 迁移期可以绑定 `unitree_compat` 的现有 velocity/tracking task factory；
- 只覆盖机器人特有参数，复用通用 observation/reward/termination；
- 注册稳定 task id，并导出 observation/action schema hash。

### D. Runtime Driver

- 实现连接、状态读取、命令写入、模式检查、健康度和安全关闭；
- 把厂商消息转换为统一 `RobotState` / `RobotCommand`；
- 驱动内不实现 Skill 逻辑，Skill runner 不直接包含厂商 SDK 类型。

### E. Safety & Calibration

- 校验关节方向、零位、顺序、限位和 IMU frame；
- 设置命令/state timeout、速度和姿态边界；
- 明确 Passive/FixStand/Damping/E-stop 的进入条件；
- 首次上机使用吊装、低增益、低幅度逐关节测试。

## 6. 预留的实机扩展接口

为了适配未来的自制机器人，Platform/Edge SDK 预留以下接口：

| 接口 | 责任 | 不应承担的责任 |
|---|---|---|
| `MotorBus` | 发现电机、读状态、批量写命令、总线健康度 | 策略推理、UI 逻辑 |
| `SensorAdapter` | IMU、编码器、力/触觉、相机等时间戳数据 | 隐式修改控制命令 |
| `CalibrationProvider` | 零位、方向、offset、传感器外参和标定报告 | 绕过硬限制 |
| `StateEstimator` | 将同步传感器转换为统一机器人状态 | 厂商通信细节 |
| `DeploymentTarget` | prepare/start/stop/status/safe 生命周期 | 直接实现某个 Skill |
| `SafetyController` | watchdog、限位、姿态/速度边界、故障降级 | 依赖 WebUI 保持在线 |

统一数据契约至少包括：

- `JointState`：position、velocity、effort、temperature、fault、timestamp；
- `RobotState`：joint state、base pose/twist、IMU、contacts、health；
- `JointCommand`：mode、position/velocity/effort target、kp/kd、deadline；
- `SensorFrame`：sensor id、frame id、monotonic timestamp、payload schema；
- `CalibrationRecord`：设备序列号、profile version、结果、操作者、时间和哈希。

接口允许 C++ 原生实现或由厂商 SDK adapter 实现。实时控制路径优先 C++；Python Skill 只能通过受控平台 action 发起标定流程或读取结果，不能进入高频电机循环。

## 7. 一键部署的准确含义

“一键部署”是执行已生成并通过检查的 Deployment Plan，不是跳过步骤：

```text
resolve Skill/Profile -> compatibility -> prepare target -> validate
                      -> start session -> monitor -> stop/safe
```

- simulation target 可在 L3 后由一个按钮完成准备、启动和监控；
- physical target 在 Driver、MotorBus、Sensor、Calibration 和 Safety capability 不完整时保持 disabled；
- 以后实机适配完成后，同一个按钮仍需执行本地连接检查、安全门禁和失败回退；
- 标定是独立、可复用且有报告的 Job，不隐藏在部署启动脚本中。

## 8. 适配成熟度等级

| 等级 | 验收标准 | 可解锁能力 |
|---|---|---|
| L0 Model | MJCF 加载、名称/索引/限位检查通过 | UI 中可见 |
| L1 Static | 静态、关节 sweep、默认姿态通过 | 训练配置 |
| L2 Policy Sim | MJLab play 和离线 schema 检查通过 | Skill 仿真 |
| L3 Sim-to-Sim | 独立 runtime + simulator 通过 | 一键仿真部署 |
| L4 Hardware Safe | 吊装、低增益、急停和 timeout 通过 | 受限实机 |
| L5 Validated | 指标、时长、回归和操作文档通过 | 正式兼容标记 |

兼容矩阵应显示“Robot Profile × Skill × Artifact hash”的实际验证等级，而不是笼统显示某型号受支持。

## 9. WebUI Robot Onboarding Wizard

建议向导分为：

1. 默认选择 Generic MuJoCo-only，硬件存在时再选择/创建 Driver；
2. 导入 MJCF/mesh 并显示模型诊断；
3. 自动发现 joints/actuators/sensors；
4. 可视化绑定 canonical joints；hardware IDs 在 physical target 中单独补充；
5. 编辑控制周期、默认姿态、增益和硬限制；
6. 运行 L0/L1 自动检查；
7. 绑定 velocity/tracking 等任务模板；
8. 生成 Profile、验证报告和未实现 capability 清单。

自动生成只能形成草稿。关节方向、硬件 ID、增益和安全上限必须由了解设备的人复核。

## 10. 对现有代码的渐进重构

`vendor/unitree_rl_mjlab/deploy/robots/*` 作为受支持的 legacy 后端保留，再分三步消除复制：

1. 先用 `unitree_compat` 完成 G1 黄金基线并固定 checkpoint/配置/指标；
2. 提取共享的 FSM、ONNX runner、observation/action 构造和参数加载；
3. 将各机器人 `Types.h`、mode check、DDS topic 和关节映射变成 Driver/Profile；
4. 建立 RoboLab-native MJLab task/train/play/export，使用双后端等价性报告验证后切换默认；
5. 用 Skill manifest 动态配置 runner，FSM 只管理通用生命周期和安全转换。

首个纵向样板建议只选一台机器人完成全链路，再推广到其他型号；不要同时重构所有机器人。
