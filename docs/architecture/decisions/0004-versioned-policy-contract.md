# ADR 0004：版本化 PolicyContract 与 Policy Bundle

- 状态：Accepted
- 日期：2026-09-03

## 背景

单独的 ONNX 文件无法可靠表达关节顺序、动作缩放、控制周期、观测字段、history 顺序、normalization 或 recurrent state。训练、sim-to-sim 和实机若各自猜测这些信息，即使 tensor shape 相同也可能执行错误策略。

## 决定

部署的最小交付物是版本化六件套 Policy Bundle：`policy.onnx`、`contract.json`、`normalization.npz`、`robot.yaml`、`task.yaml` 和 `manifest.json`。加载器同时校验摘要、RobotSpec、TaskSpec、PolicyContract、ONNX 图与元数据，并拒绝未知版本或语义不一致。sim-to-sim 与未来硬件端共享同一 Bundle 读取语义。

## 替代方案

- 只依赖 ONNX 输入输出名称：无法覆盖动作和时序语义。
- 把所有配置嵌入 ONNX metadata：工具链支持不一致，也不适合大型 normalization 数据。
- 由部署代码硬编码每种 profile：会重复训练端知识并导致版本漂移。

## 后果

错误策略会在控制循环前失败，产物可审计并支持 history、teacher conditional 与外部 recurrent state。代价是导出和版本迁移必须维护严格 schema，任何破坏性变更都需要提升 contract version。

## 迁移影响

PolicyContract v1 最初固化 Go2 的 12 维动作、基础观测、条件字段、50 Hz 控制与 TS student recurrent state；同一结构现也表达 G1 29-DoF 的 Flat/Rough PPO 契约。`lainloco export` 从 checkpoint 直接生成完整 Bundle，旧式裸 ONNX 仍可显式通过 `bundle create` 封装。
