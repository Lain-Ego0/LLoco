# MJLab Adapter（接口边界，未实现）

状态：本目录目前只声明所有权与接口边界，没有平台代码。实现顺序见 [ROADMAP](../../docs/ROADMAP.md) Phase 1。

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

设计细节见 [ARCHITECTURE §4.1](../../docs/ARCHITECTURE.md)。
