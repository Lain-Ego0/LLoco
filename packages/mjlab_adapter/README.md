# MJLab Adapter（Unitree Legacy 兼容入口）

状态：已废止，等待 R0.6 删除。本目录曾通过受控子进程运行 `vendor/unitree_rl_mjlab/` 的旧 discovery/play 入口。
历史行为由 Git 记录保存；不得继续为本包增加功能、修复兼容性或将其改名伪装成通用 backend。

## 职责

- 读取固定 Unitree vendor registry，提供 legacy task discovery；
- 把经过 schema 校验的兼容配置转换为旧脚本参数；
- 构造受控的 legacy play/train/export JobCommand；
- 让 Local Worker 负责进程隔离、日志、取消、退出码和 Artifact 登记；
- 提供迁移报告所需的旧 task、模型、配置和 checkpoint 信息。

## 明确边界

- 本目录不修改 `vendor/unitree_rl_mjlab/`，也不把其 task ID 变成公共 Skill/API ID；
- 本目录不负责实现 MJLab 1.6 的 Task/Robot/Train/Play 工具链；
- 新平台功能进入 `mjlab/` 的定制代码、`packages/mjlab_tasks/` 或 Platform Core；
- 厂商专用 Runtime/SDK 逻辑进入 `integrations/<vendor>/` 和 `runtime/`；
- 旧环境中的 `mjlab==1.2.0` 只在显式 legacy 环境使用。

## 后续处理

R0.6 将删除本目录、对应 CLI/API 依赖和专用测试。不会迁移为 `backends/unitree_compat/`。未来 Customized MJLab 1.6
的任务入口必须重新实现，不能导入或调用本包。
