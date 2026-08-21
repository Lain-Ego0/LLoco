# MJLab Adapter（Unitree Legacy 兼容入口）

状态：B3 过渡版已实现。本目录通过受控子进程运行 `vendor/unitree_rl_mjlab/` 的旧 discovery/play 入口，
用于保存 G1 迁移资料、旧 checkpoint 诊断和 Unitree sim-to-sim 兼容性。它不是 Customized MJLab 1.6 主线工具链。

## 职责

- 读取固定 Unitree vendor registry，提供 legacy task discovery；
- 把经过 schema 校验的兼容配置转换为旧脚本参数；
- 构造受控的 legacy play/train/export JobCommand；
- 让 Local Worker 负责进程隔离、日志、取消、退出码和 Artifact 登记；
- 提供迁移报告所需的旧 task、模型、配置和 checkpoint 信息。

## 明确边界

- 本目录不修改 `vendor/unitree_rl_mjlab/`，也不把其 task ID 变成公共 Skill/API ID；
- 本目录不负责实现 MJLab 1.6 的 Task/Robot/Train/Play 工具链；
- 新平台功能进入 `vendor/mjlab/` 的定制代码、`packages/mjlab_tasks/` 或 Platform Core；
- 厂商专用 Runtime/SDK 逻辑进入 `integrations/<vendor>/` 和 `runtime/`；
- 旧环境中的 `mjlab==1.2.0` 只在显式 legacy 环境使用。

## 后续处理

RoboLab 可以将本目录改名或迁移为 `backends/unitree_compat/`，但这只是代码整理，不是新的主线后端。任何 Unitree
兼容测试都必须标记 `toolchain: unitree_legacy_mjlab_1_2`，并且不能阻塞 MJLab 1.6 定制、非 Unitree 机器人接入或 Runtime 开发。
