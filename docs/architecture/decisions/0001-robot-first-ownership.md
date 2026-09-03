# ADR 0001：机器人优先的能力归属

- 状态：Accepted
- 日期：2026-09-03

## 背景

Go2 的任务、观测、控制约束和部署逻辑原先分散在 mjlab 的 velocity 命名空间中。按仿真框架类别组织虽然便于复用底层设施，但开发者无法从一个稳定边界看到单个机器人的完整能力，也容易把机器人专用逻辑继续写入通用框架。

## 决定

机器人专用资产、MDP、技能任务、训练参数绑定、PolicyContract 和部署 FSM 由 `lainloco.robots.unitree.go2` 拥有。真正与机器人无关且已有复用证据的算法实现位于 `lainloco.learning`，仿真生命周期仍由 mjlab 提供。

## 替代方案

- 继续按 `mjlab.tasks.velocity` 组织所有 Go2 能力：改动较少，但会扩大上游 fork 和所有权混淆。
- 为每个技能建立独立顶层包：局部清晰，但会复制 RobotSpec、MDP 和部署契约。
- 立即提取所有看似通用的代码：边界过早，尚无第二种机器人证明其抽象稳定。

## 后果

Go2 能力集中可发现，mjlab 同步冲突减少；代价是需要 Catalog、entry point 和兼容 alias 连接新领域边界与旧入口。共享代码只有在出现跨机器人复用证据后才提升为通用模块。

## 迁移影响

Go2 环境工厂和专用 MDP 已迁入机器人目录，mjlab 中的实现已删除。旧 Python 导入路径由扩展 bootstrap 临时映射，旧 Task ID 继续可用。
