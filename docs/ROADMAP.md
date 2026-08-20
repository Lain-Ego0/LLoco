# RoboLab 实施路线图

## Phase 0：来源与契约基线

- 以已验证的 upstream commit `1425b15f` 建立可追踪导入历史；
- 以独立 curated vendor commit 导入 `vendor/unitree_rl_mjlab/`，使用 manifest 排除宣传文档、预编译 runtime 和 Skill 产物；
- 选择仓库内一个公开 Unitree MJCF 作为首个 simulation-only 样板；
- 冻结 `RobotProfile v1alpha1` 和统一 `SkillPackage v1alpha1` 最小 schema；
- 为现有 velocity 产物生成 MotionSkill，并创建一个可执行 PlatformSkill；
- 建立基础 lint、schema、license 和 artifact hash 检查。

退出条件：任何人能解释一份 Skill 为什么与某 Robot Profile 兼容或不兼容。

## Phase 1：本地平台闭环

- FastAPI API、SQLite、artifact store 和本地 Worker；
- 发现现有 MJLab registry tasks；
- Job 日志、取消、状态和配置 snapshot；
- Skill catalog 安装、版本固定、权限审查、Conda prepare、contract test 与卸载；
- `robolab-job-v1` 子进程协议和统一 action registry；
- AgentSkill 校验和 Codex `.agents/skills/` export 原型；
- 最小 React WebUI：Robots、Skills、Jobs、Artifacts；
- MJLab play 与验证报告。

退出条件：MotionSkill 和 PlatformSkill 都能从 catalog 安装后直接调用，且不导入 API 进程并保留完整 lineage。

## Phase 2：仿真部署与 Edge

- 抽取共享 C++ runtime；
- 完成首个 simulation target，并定义但不强制实现实机 Driver/Profile；
- 接入 unitree_mujoco sim-to-sim；
- Edge heartbeat、watchdog、session token、start/stop/safe；
- WebUI Validate 与 Deploy 页面；
- 自动 compatibility 与 safety gates。

退出条件：实现一键仿真部署；physical target 缺少 Driver/Calibration/Safety 时明确阻止激活。

## Phase 3：训练与机器人向导

- 训练配置 schema、资源选择、指标和 checkpoint；
- Robot onboarding wizard 与自动模型诊断；
- 导入第二个公开 MJCF，验证 simulation-only 抽象；
- 定义 MotorBus、SensorAdapter、CalibrationProvider 和 StateEstimator 接口；
- Skill 回滚和兼容矩阵回归测试。

退出条件：第二个机器人不复制整套 runtime 即可完成 L0-L3 适配。

## Phase 4：受控实机与生态

- 为实际自制机器人实现驱动、传感器、标定、安全控制和本地确认；
- Skill 签名、可信发布者和权限声明；
- 可选 Docker、远端 Worker 和多人模式；
- CompositeSkill 与稳定扩展 SDK；
- 发布贡献、兼容认证和安全披露流程。

## 首个迭代建议

首个两周迭代建议实现三条最小样例：“G1 velocity MotionSkill -> MJLab play”、“MJCF Inspector PlatformSkill -> 诊断报告”，以及统一安装/调用页面。AgentSkill 是否同时进入首版，等待第二轮 QA 确认。
