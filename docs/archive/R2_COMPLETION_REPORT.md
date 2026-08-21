# R2 完成报告：FireDog 2.2 simulation-first 接入

状态：`COMPLETED — simulation-only scope`（2026-08-21）

本报告只覆盖 R2，不启动 GPU 训练、不加入真实 Driver/Runtime、不恢复 Unitree/G1 active 栈。R2 阶段标记仍保留为 active，避免在本次交付中提前激活 R3。

## 1. 已实现的 R2.x

| 工作项 | 实现与证据 |
|---|---|
| R2.1 来源与 Profile | `robots/firedog2.2.SLDASM/profile.yaml`，公共 ID `community.firedog2_2@1.0.0`；`provenance.yaml` 记录 24 个原始输入文件的大小/SHA-256、SW2URDF 版本、来源和授权边界 |
| R2.2 Inspector | `robolab robot inspect`；覆盖 body/joint/actuator/sensor/site/mesh/keyframe/geom、引用存在性、重复名称、非法限位、非法 root body；诊断规则为稳定 `INSPECT_*` ID，JSON 证据在 `diagnostics/` |
| R2.3 Profile vNext | 新增 `RobotProfile v1beta1`，不修改冻结的 `v1alpha1`；增加正例、反例和兼容说明 |
| R2.4 Mapping | 复用现有 `JointSet v1alpha1` 语义检查；新增 `ActuatorSensorMapping v1alpha1`，覆盖 16 actuator、34 sensor、单位/方向/限位/gear/频率/延迟/MJCF 名称索引/policy 索引/hardware ID |
| R2.5 转换与 snapshot | `robolab robot convert` 确定性 URDF→MJCF；`robolab robot snapshot` 输出排序 JSON，当前 snapshot SHA-256 为 `78e710edcbe48372539c272307e281972e937b91540431a81f65464ac2860b42` |
| R2.6 最小仿真 | 独立 MuJoCo CPU smoke 与 Customized MJLab 1.6 `ManagerBasedRlEnv` factory；完成 load/reset/default pose/limits/named action/observation/action-observation dimensions/task binding |

## 2. 关键入口

- [FireDog README](../../robots/firedog2.2.SLDASM/README.md)
- [Profile v1beta1](../../robots/firedog2.2.SLDASM/profile.yaml)
- [JointSet](../../robots/firedog2.2.SLDASM/joint_set.yaml)
- [MJCF](../../robots/firedog2.2.SLDASM/model/firedog2_2.xml)
- [Provenance](../../robots/firedog2.2.SLDASM/provenance.yaml)
- [Resolved snapshot](../../robots/firedog2.2.SLDASM/snapshots/resolved_config.snapshot.json)
- [Inspector core](../../packages/core/src/robolab_core/robot_inspector.py)
- [Config/snapshot core](../../packages/core/src/robolab_core/robot_config.py)
- [Simulation smoke core](../../packages/core/src/robolab_core/robot_simulation.py)
- [MJLab task factory](../../mjlab/src/mjlab/tasks/firedog2_2/firedog2_2_env_cfg.py)

CLI 入口：

```bash
PYTHONPATH=apps/cli/src:packages/core/src:packages/schemas/src \
python -m robolab_cli.main robot inspect <model.xml> --json
python -m robolab_cli.main robot convert <robot.urdf> --output <model.xml> --json
python -m robolab_cli.main robot snapshot robots/firedog2.2.SLDASM/profile.yaml \
  --repo-root . --task community.firedog2_2.velocity.flat \
  --output robots/firedog2.2.SLDASM/snapshots/resolved_config.snapshot.json --json
```

## 3. 来源、revision、许可证和不确定性

- 原始来源：`/home/lxy/RoboLab/robots/firedog2.2.SLDASM`，是用户提供的目录资产，不是单文件 SolidWorks 二进制。
- 原始 revision：无外部仓库 revision；provenance 明确记录为 `null`，以 URDF/CSV/package.xml/export.log 和全部输入 SHA-256 固定。
- RoboLab revision：`3b9ef4a7561faff4d2ab80351fe20262e1d5ff71`；MJLab upstream revision：`0fb8a681136be94ffc636a3dd423cabb97d91f10`。
- SolidWorks URDF Exporter：commit `1.6.0-4-g7f85cfe`，build `1.6.7995.38578`。
- `package.xml` 声明 BSD，但作者和维护者仍为 TODO；不能据此推断 STL、URDF、CSV、CAD 或日志可再分发。
- 当前授权状态为 `unclear_local_only`：只允许当前本地 simulation-only 使用，不宣称许可证已完全确认。原始 `export.log` 中的 Windows 路径和 CSV 列写入错误均被保留记录，派生 MJCF 不依赖这些路径。

## 4. URDF → MJCF 链

输入：

- URDF SHA-256：`9a98cffad4c2559e881bc9b36a596aea58f1ab0658ee386a5f44e1ccf1119ba1`
- 17 个 STL mesh，全部被引用且存在；
- URDF 实测 16 joints：RF/RR/LR/LF 各 4 个，前三个 revolute，wheel 为 continuous。

命令：

```bash
robolab robot convert urdf/火狗2.2.SLDASM.urdf \
  --output model/firedog2_2.xml --json
```

输出 MJCF SHA-256：`55a2852da8f3c4aa02ed46f7cdab085ccc5401454cd38afb5f26bb979a6eb516`。

RoboLab simulation 假设已记录：position actuator（kp=40、kv=4）、effort-derived force range、joint state/IMU sensor、foot site、collision geom；continuous wheel 在 MJCF 保持 unlimited，JointSet 只使用有限 inspection window。

## 5. 验证结果

CPU contract：

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with PyYAML --with jsonschema \
  --with fastapi --with httpx --with pytest pytest -q tests/contract
126 passed, 1 skipped, 1 warning
```

定向 R2：

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with PyYAML --with jsonschema \
  --with mujoco --with numpy --with pytest \
  pytest -q tests/contract/test_r2_robot_onboarding.py tests/contract/test_r2_mujoco.py
10 passed
```

Customized MJLab 1.6 R2 task：

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest \
  pytest -q tests/test_firedog2_2_r2.py
1 passed
```

MJLab CPU smoke：

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest -q tests/smoke_test.py
1 passed
```

Ruff：

```
uv run --with ruff ruff check <R2 root Python files>
All checks passed!

uv run --with ruff ruff check <R2 MJLab Python files>
All checks passed!
```

模型/仿真实测：

- MuJoCo compile：`nbody=18, njnt=17, nu=16, nsensor=34, nq=23, nv=22`；
- reset：default keyframe，time `0.0`；
- named action：`RF_joint -> act_RF_joint`，`0.05`；
- observation：`32 -> 32`，finite，action 后值发生变化；
- 关节限位：12 个 revolute position limits 检查通过，4 个 continuous wheel 明确标记为无限位置关节；
- MJLab factory：action `16`，actor observation `32`。

## 6. CPU/GPU/Viewer/硬件证据边界

- CPU：contract、Inspector、schema/mapping、URDF→MJCF、snapshot、MuJoCo smoke、MJLab task factory 均有通过证据。
- GPU：未启动训练，未声称通过。
- Viewer：真实命令尝试：

  ```
  timeout 8s uv run --with mujoco python <launch_passive script>
  viewer_started
  exit 124 (intentional evidence timeout)
  ```

  Native MuJoCo viewer 已进入运行态；没有伪造截图。8 秒后由 timeout 主动结束进程。
- 硬件：未接入 Driver、Motor ID、SDK、MotorBus、Calibration 或 physical Runtime，未声称通过。
- 策略：没有 zero-policy、随机动作或占位 checkpoint；R2 仅验证 named action 接口和物理响应，不宣称策略完成。

## 7. R2 退出条件

R2 的 simulation-first 退出条件满足：FireDog Profile、来源追溯、Inspector 正反例、vNext schema、mapping、可复现转换、snapshot、Customized MJLab 1.6 load/reset/action/observation、通用 task binding、Viewer 启动证据和测试记录均已具备。

许可证授权仍不清晰，因此本结果不能扩展为资产再分发许可或硬件兼容声明。下一阶段最小行动是：由资产权利人补充 CAD/STL/URDF/CSV 的明确授权或可再分发许可；若需要进入实机，则另行实现并验收 Driver/Calibration/Safety，不属于 R2。
