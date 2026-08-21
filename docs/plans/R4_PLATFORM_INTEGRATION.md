# R4：Skill、平台与验证整合

状态：`⬜ NOT ACTIVE`。

## 目标

将现有 Skill、Worker、API、CLI、WebUI 和 Artifact Store 连接到 Customized MJLab 1.6 工具链和 R3 PolicyArtifact。

## 进入条件

- R3 已生成通过独立评测的固定 PolicyArtifact；
- MotionSkill、Robot Profile、Task 和 Artifact 的版本绑定字段已确定；
- 当前阶段已在 `plans/README.md` 激活。

## 本阶段不做

- 不在 API 或 WebUI 进程内运行训练、Viewer 或策略推理；
- 不让 WebUI 生成一套与 CLI 不同的隐式配置；
- 不把浮动路径、未固定 checkpoint 或无 hash 文件登记为正式 Artifact；
- 不提前实现 R5 的硬实时 Runtime 或 physical activation。

## 工作项

| ID | 工作项 | 交付物 | 验收 |
|---|---|---|---|
| R4.1 | MotionSkill binding | robot/task/toolchain/artifact/evaluate/export | manifest 无厂商路径 |
| R4.2 | Worker Job | train/play/evaluate/export 隔离执行 | 日志、事件、取消和输出完整 |
| R4.3 | Platform API | Task、Recipe、Artifact、Validation 资源 | 只接受稳定 ID/revision/schema |
| R4.4 | ValidationRun | fixed inputs、metrics、thresholds、evidence、status | 追溯到 Skill/Profile/Artifact hash |
| R4.5 | WebUI Train/Validate | schema 表单、raw config、状态和错误原因 | 与 CLI 等价 |
| R4.6 | Artifact lineage | checkpoint、ONNX、config、video、report | 禁止浮动 revision 或无 hash 产物 |

## 退出条件

用户可从安装 MotionSkill 开始，通过 API/CLI/WebUI 创建 MJLab 1.6 Job，并得到固定 PolicyArtifact 和 ValidationRun；三种入口
必须引用同一 schema、resolved config 和 lineage，不得只完成页面演示。
