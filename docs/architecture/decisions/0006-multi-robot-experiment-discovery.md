# ADR 0006：用全局 Catalog 和后端 Binding 接入多个机器人

- 状态：Accepted
- 日期：2026-09-03

## 背景

Go2 最初同时拥有机器人事实、实验 Catalog 和 mjlab 工厂绑定。训练、回放、导出与 CLI
直接解析 Go2 Catalog，因此新增 G1 会要求复制工作流或继续把通用入口绑定在 Go2
namespace。另一方面，mjlab 1.6 已维护 G1 29-DoF 资产和 Velocity Flat/Rough 环境，
本地 `unitree_rl_mjlab-main` 还可用于核对后续 Tracking 与部署能力边界。

## 决定

建立仓库级 `lainloco.experiments`，统一发现 RobotSpec、TaskSpec、TrainingSpec 与
Experiment binding。每个机器人仍拥有自己的事实与组合；mjlab 专有的 registry ID、
配置 factory、runner 和导出 metadata 统一封装为
`lainloco.integrations.mjlab.MjlabExperimentBinding`。CLI 与通用 workflow 只解析全局
Catalog，不导入具体机器人。

G1 首批范围固定为 29-DoF Flat/Rough Velocity + PPO，并复用 mjlab 1.6 维护的环境工厂。
LainLoco 为其增加 RobotSpec、PolicyContract、canonical Task ID、Bundle 与 sim-to-sim
验收。Tracking、G1-23DoF 和硬件部署不在首批范围内。

## 替代方案

- 为 G1 复制 Go2 CLI/workflow：实现直接，但每增加机器人都会分叉训练与部署路径。
- 把所有机器人组合写进单个注册模块：可集中枚举，但机器人事实和后端细节会重新耦合。
- 将参考仓库整体并入 `vendor/`：能一次得到更多机器人，却会引入 mjlab 1.2/1.6 API
  差异、重复资产和额外维护边界。

## 后果

新增机器人只需增加自身 Catalog 与一个明确的后端 binding，现有 train/play/export/
Bundle/sim-to-sim 工作流无需复制。代价是 bootstrap 必须处理“mjlab 已注册的源任务”与
“LainLoco 自己构造的任务”两种来源，并持续用 fresh-interpreter 测试防止 entry-point
导入循环。

## 迁移影响

G1 源 ID 继续由 mjlab 提供，LainLoco 为其注册两个 canonical ID；Go2 旧 ID 和兼容
Python alias 保持不变。全局 Catalog 当前包含 18 个实验组合。参考仓库只用于语义和能力
核对，首批 G1 代码与资产均来自当前 mjlab 1.6 API/资产，没有复制其部署程序或策略文件。
