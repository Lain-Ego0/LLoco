# RoboLab 实施路线图

> 当前状态、依赖、验收和工作项以 [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md) 为唯一权威。
> 本路线图只描述阶段目标，不替代逐项验收。

## 总路线

```text
MJLab 1.6 基座固定
  -> RoboLab 定制运动工具链
  -> 非 Unitree 自研机器人接入
  -> 训练/回放/评测/导出
  -> Skill/Artifact/Validation
  -> Runtime/sim-to-sim/stop-safe
  -> 自研机器人 + 成品机器人验证
```

Unitree RL MJLab 是 legacy/reference 路线，不是总路线的根节点。

## 历史完成部分

- ✅ 建立 `vendor/unitree_rl_mjlab/` 的来源追踪、许可证和精选导入记录；
- ✅ 冻结 `RobotProfile`、`JointSet` 和 `SkillPackage` 最小 schema；
- ✅ 实现基础 lint、兼容性检查、Skill 安装、Job、API、Worker 和 WebUI 骨架；
- ✅ 建立 G1 29DoF simulation-only Profile 和旧 MJLab/Unitree 兼容入口。

这些成果继续有效，但不能等同于 MJLab 1.6 深度定制已经完成。

## 当前主线阶段

### R0：MJLab 1.6 基座固定

固定 upstream commit、默认依赖环境、许可证、修改账本、smoke test 和同步规则。

### R1：定制 MJLab 工具链

实现 Robot/Task Registry、统一 train/play/evaluate/export、schema、Artifact metadata 和回归测试。

### R2：自研机器人接入

引入非 Unitree 参考机器人，完成 MJCF/Profile 诊断、执行器/传感器映射、reset、observation 和 Viewer。

### R3：真实运动策略

完成首个 velocity/tracking task 的训练或可信 checkpoint 回放、独立指标评测和策略导出。

### R4：Skill 与平台闭环

让 MotionSkill、API、CLI、Worker 和 WebUI 调用 Customized MJLab 1.6，并保存完整 Artifact lineage。

### R5：部署适配与 Runtime

实现 ONNX runner、FSM、heartbeat、watchdog、simulation driver、DeploymentSession 和 stop/safe。

### R6：多机器人验证

使用至少一个自研机器人和一个成品机器人完成共享契约验证；G1 在此作为普通成品机器人适配器。

## 明确不再采用的旧门禁

- 不再要求先完成 G1 黄金 checkpoint 才能开发 MJLab 1.6；
- 不再把 `unitree_compat` 和 `mjlab_native` 作为长期对等产品后端；
- 不再要求 MJLab 1.6 与旧 MJLab 1.2 逐帧等价；
- 不再把自研机器人排到第二机器人或后期生态阶段；
- 不再把“修改 MJLab”视为架构违规。

## 后续阶段

R6 完成后，再评估真实硬件 Driver、Calibration、Safety、Skill 签名、第二类训练后端、远程 Worker、Docker 和多人模式。
