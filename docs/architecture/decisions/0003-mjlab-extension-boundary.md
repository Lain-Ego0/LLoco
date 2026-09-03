# ADR 0003：以扩展包依赖 mjlab

- 状态：Accepted
- 日期：2026-09-03

## 背景

项目需要 mjlab 的 Manager-based 环境、MuJoCo Warp、注册表和 RSL-RL 集成，同时又需要大量 Go2 专用任务和算法。直接把这些实现长期保存在 mjlab fork 中，会增加上游同步冲突并模糊两个项目的发布身份。

## 决定

LainLoco 是独立 workspace distribution，并通过 `mjlab.tasks` entry point 注册任务。依赖方向固定为 `lainloco → mjlab`；mjlab 不静态导入 LainLoco。只有可被非 Go2 任务复用的仿真能力才进入 mjlab。

## 替代方案

- 永久维护完整 mjlab fork：入口简单，但升级和发布成本最高。
- 复制 mjlab 到 LainLoco：获得控制权，却造成底层框架分叉和重复维护。
- 立即拆成远端独立仓库：最终边界清晰，但会在重构期间破坏本地可运行基线。

## 后果

Go2 迭代与 mjlab 上游演进解耦，包元数据能验证单向依赖。兼容旧导入需要运行时 alias，且在兼容周期结束前不能假设仅安装 mjlab 即可获得 Go2 任务。

## 迁移影响

workspace 根包即 `src/lainloco`，entry point 自动注册新旧任务入口；本地后端 fork 明确位于 `vendor/mjlab`，mjlab 的 Go2 专用脚本、环境实现和 learning 实现已迁出。现存 mjlab 通用改动仍需按上游贡献或最小 fork 策略逐步收敛。
