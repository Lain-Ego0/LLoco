# MJLab Adapter

状态：B3 最小版已实现。它以受控子进程运行 vendor 的只读
`scripts/list_envs.py` 发现 task，并构造 `scripts/play.py <task>` 的等价命令；
实际执行、日志、退出码与取消由 Local Worker 管理。`train.py` 不在 MVP 范围。

## 职责

把平台 Job（train / play / export / evaluate）映射到 `vendor/unitree_rl_mjlab/` 中现有的脚本与任务 registry：

- 从 vendor 的 registry 发现 task，而不是在 WebUI 手写任务列表；
- 把经过 schema 校验的配置转换为 CLI 参数或配置文件；
- 以受控子进程启动 train/play/export，采集 stdout、指标和退出码；
- 将 checkpoint、ONNX、deploy.yaml、视频和日志登记为 Artifact；
- 不把训练进程嵌入 API 进程。

## 边界

- 本目录是 RoboLab 自有代码，不写入 `vendor/`；
- 第一版通过子进程调用 vendor 现有 `scripts/train.py`、`play.py` 等入口，等接口稳定后再重构内部实现；
- 厂商专用逻辑不属于本目录，放 [`integrations/`](../../integrations/)；部署运行时不属于本目录，放 [`runtime/`](../../runtime/)。

设计细节见 [ARCHITECTURE §4.1](../../docs/reference/ARCHITECTURE.md)。
