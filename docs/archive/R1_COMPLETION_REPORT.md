# R1 完成报告：MJLab 定制工具链基础

状态：`COMPLETED`（2026-08-21）

R1 建立了 RoboLab 自己的 `robolab-motion-v1` 命令契约。CLI、API 和未来 Worker 可以共享同一份
JobCommand schema；当前实现只构造和持久化 motion job，不启动训练、不创建真实部署会话，也不登记真实机器人。

## R1.1–R1.8 证据

| 项目 | 实现证据 | 测试证据 |
|---|---|---|
| R1.1 Toolchain identity | `robolab_core.motion.ToolchainIdentity`；Job metadata、JobCommand、evaluate evidence 和 export metadata 均记录 `robolab_mjlab@1`、MJLab upstream SHA、RoboLab SHA | `test_toolchain_identity_is_stable_and_embedded`、CLI train JSON 检查 |
| R1.2 Robot Registry | `RobotBinding` / `RobotRegistry`，使用 id+SemVer 和 import-style MJLab entity/config；默认 registry 为空，R1 不选择真实机器人 | registry lookup、版本错误、重复注册和空默认 registry 测试 |
| R1.3 Task Registry | `TaskDefinition` / `TaskRegistry`，task 与 RobotBinding 分离，声明 JSON Schema 并校验 resolved config | task discovery、配置错误、版本和 unknown entry 测试 |
| R1.4 Train | `build_train_command` 和 `robolab train`，包含 resolved config、seed、resume hash、资源和 output directory | CPU command construction、persisted Job metadata、CLI train 测试 |
| R1.5 Play | `build_play_command` 和 `robolab play`，包含 checkpoint hash、viewer、recording、deterministic 和 Customized MJLab backend | checkpoint hash、deterministic 和 CLI play 测试 |
| R1.6 Evaluate | `build_evaluate_command` 和 `make_evaluation_result`，包含 scene、episodes、metrics、thresholds、machine-readable evidence | metric/threshold/result 测试 |
| R1.7 Export | `build_export_command` 和 `write_artifact_metadata`，包含 source hash、observation/action schema、频率、scale、joint order、toolchain metadata | export metadata 和 CLI export 测试 |
| R1.8 Regression | motion schema、registry、command、CLI、API 和 smoke 测试 | root contract suite 与 MJLab CPU smoke 通过 |

## 关键修改文件

- `packages/core/src/robolab_core/motion.py`：R1 identity、registry、JobCommand、artifact/evaluation helpers 和 CPU smoke probe。
- `packages/schemas/src/robolab_schemas/data/motion_command.v1.schema.json`：`robolab-motion-v1` 结构 schema。
- `packages/schemas/src/robolab_schemas/__init__.py`：motion schema 加载和校验入口。
- `packages/core/src/robolab_core/jobs.py`：Job input 的结构化 metadata 扩展。
- `apps/cli/src/robolab_cli/main.py`：`train`、`play`、`evaluate`、`export` 和 `motion list`。
- `services/api/src/robolab_api/app.py`：toolchain、task、robot discovery API。
- `tests/contract/test_motion_toolchain.py`、`tests/contract/test_motion_cli.py`：R1 核心回归。

## 验证记录

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with PyYAML --with jsonschema --with fastapi --with httpx --with pytest pytest -q tests/contract
117 passed, 1 warning

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/contract/test_motion_toolchain.py tests/contract/test_motion_cli.py
12 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/smoke_test.py   # cwd=mjlab
1 passed

uv run --with ruff ruff check --select I001,F401,RUF022,UP035 <R1 changed files>
All checks passed
```

MJLab 全部非 slow 测试曾尝试运行，但在本地 120 秒时限内未完成；没有把该尝试记录为通过。R1 的退出条件要求的
MJLab CPU smoke 已通过。

## 证据边界与已知限制

- CPU：已验证 registry、schema、四类 command construction、Job metadata、API discovery 和 MJLab cartpole smoke。
- GPU：本机检测到 CUDA/RTX 4060，但没有启动 GPU 训练；没有 GPU 训练完成证据。
- Viewer：没有启动 native/Viser viewer；play 仅验证结构化命令构造。
- 硬件：没有真实机器人、厂商 SDK、Driver 或部署 Runtime 证据。
- Robot Registry 当前为空；R2 才选择并接入正式自研/社区 Robot Profile。
- command 的 `result.status` 初始为 `pending`，不会用 zero-policy、随机动作或占位 checkpoint 宣称真实策略完成。
- 当前仓库既有上游 MJLab 文档/fixture 中可能出现历史示例名称；R1 active RoboLab registry、CLI 和 API 没有 Unitree adapter、vendor path、`--vendor-root` 或 G1 task ID。

## 阶段边界

本报告不宣称 R2–R6 功能完成：没有实现真实机器人接入、完整 WebUI Train/Validate、Deployment Runtime、watchdog/FSM、实机或厂商 SDK。
