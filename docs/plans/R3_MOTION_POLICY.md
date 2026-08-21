# R3：真实运动策略工作流

状态：`🔄 ACTIVE`；真实训练和策略证据正在实施。

## 目标

为 R2 参考机器人完成一个真实 velocity 或 tracking 任务的 train/play/evaluate/export，并生成固定 PolicyArtifact。

## 进入条件

- R2 完成；
- observation/action schema 可从任务生成；
- 训练配置和 Artifact metadata 接口稳定；
- 当前阶段被明确激活。

## 本阶段不做

- 不用 zero-policy、随机动作或仅能加载的 checkpoint 代替真实策略；
- 不为了提高单一机器人指标而把机器人专用字段写入公共 Task/Artifact schema；
- 不启动实机控制；本阶段只固定仿真策略、评测与部署所需 metadata；
- GPU 不可用时可以完成配置、指标和 CPU 测试，但不得把真实训练项标记完成。

## 工作项

| ID | 工作项 | 交付物 | 验收 |
|---|---|---|---|
| R3.1 | 首个通用任务 | `motion.velocity.flat@1` 或经决策确认的 tracking task | task 与 robot binding 分离 |
| R3.2 | Observation/Action schema | name、shape、dtype、unit、frame、history、scale | 进入 Artifact |
| R3.3 | TrainingRecipe | seed、runner、network、reward、randomization、资源 | resolved config 可重放 |
| R3.4 | 训练或可信 checkpoint | policy、日志、代码和环境 revision | 不使用 zero-policy 替代 |
| R3.5 | Trained play | Viewer/视频和 episode 结果 | 有效运动证据 |
| R3.6 | Independent evaluation | 速度、姿态、接触、终止、扰动指标 | 阈值版本化 |
| R3.7 | Export | ONNX 和 deploy metadata | 固定输入下按阈值一致 |
| R3.8 | PolicyArtifact | checkpoint/ONNX/config/schema/hash/lineage | lint、compatibility、复现通过 |

## 退出条件

至少一个真实策略完成 train/play/evaluate/export 并形成固定 PolicyArtifact。GPU 不可用时不得虚假关闭本阶段。
