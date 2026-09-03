# ADR 0002：分离 Task 与 TrainingProfile

- 状态：Accepted
- 日期：2026-09-03

## 背景

历史入口把 DreamWaQ、CTS、AMP 和 Teacher-Student 等算法名称编码进 Task ID，使“机器人做什么”与“怎样优化策略”绑定。相同环境目标因此产生重复注册项，也难以判断算法变化是否改变了任务语义。

## 决定

`TaskSpec` 只描述技能、地形和环境目标；`TrainingSpec` 描述算法、模型、runner、storage 和训练默认值；`ExperimentSpec` 显式组合 Robot、Task、TrainingProfile 与 PolicyContract。普通训练使用 `lainloco train <task> --profile <profile>`，蒸馏使用必须提供 teacher checkpoint 的独立 workflow。

## 替代方案

- 保留每个“任务×算法”独立 ID：兼容直接，但组合数量持续膨胀。
- 让 CLI 根据名称隐式猜测算法：命令短，但无法静态校验和复现。
- 把蒸馏视为 PPO 参数开关：会掩盖 teacher 依赖和独立 recurrent 更新循环。

## 后果

任务和训练能力可独立演进，Catalog 能在启动前拒绝无效组合；代价是所有运行必须携带明确 profile，旧算法型 ID 只能作为兼容接口存在。

## 迁移影响

当前有 8 个技能、8 个 TrainingProfile 和 16 个有效 Experiment 绑定。新 Task ID 不含算法名称；旧 `Mjlab-*` ID 映射到同一环境与训练配置。TS student 由 `lainloco distill` 启动。
