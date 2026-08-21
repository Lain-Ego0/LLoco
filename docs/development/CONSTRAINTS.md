# 开发约束

本文是跨阶段强制约束。当前阶段和退出条件见 [plans/README.md](../plans/README.md)；本文件不记录阶段进度。

## 技术边界

- `mjlab/` 是 MJLab 1.6 下游定制基座，允许记录和维护 RoboLab-specific 修改；
- WebUI、API、Worker、Skill Registry、Artifact Store 和 Edge Runtime 不全部塞入 MJLab 核心；
- 新机器人必须通过 Robot Profile、Task、Artifact 和 Runtime 公共契约；
- 不复制厂商目录，不把 SDK、DDS topic、motor ID 或厂商 task ID 加入公共 schema；
- 新增第二机器人不得复制完整 task、runner、FSM 或 Skill 工作流。

## 可复现性

每个训练、评测或部署 Artifact 必须固定：代码 revision、MJLab upstream revision、环境/lock、Robot Profile、Task、seed、
最终配置、输入输出 schema、Artifact hash 和验证证据。

## 安全

- WebUI/API 断开不能阻止 Runtime 进入安全状态；
- heartbeat、命令超时、限幅、状态机和故障回退必须在 Edge/Runtime；
- physical target 缺少 Driver、Calibration 或 Safety 时 fail closed；
- 未通过仿真验证的 Skill 不得激活实机。

## 证据与状态

- 代码、测试和证据路径齐全后才可标记完成；
- CPU contract、GPU 训练、人工 Viewer 和硬件验收分开记录；
- 不使用 zero-policy、随机动作或只有页面的占位实现冒充运动能力；
- 不修改或删除历史事实，只在 archive 中保留并明确其不可执行性。

## Agent 执行规则

自动 Agent 必须先读取当前阶段计划和本约束，不得依据 archive 中的旧命令执行；不得自行跨阶段；发现计划冲突时先报告并
更新决策文档，不得恢复 Unitree-first 路线。
