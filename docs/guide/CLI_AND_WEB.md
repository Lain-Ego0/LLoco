# CLI 与 WebUI

## 启动本地平台

```bash
cd /home/lxy/RoboLab
conda activate robolab-mjlab16
python -m pip install -e packages/schemas -e packages/core \
  -e services/api -e services/worker -e apps/cli
robolab serve
```

服务只监听 `127.0.0.1`。当前 CLI/API 提供平台检查、Skill、Job 和服务管理基础能力；新的 MJLab 1.6 motion commands
按 R1 计划实现，不得恢复已删除的 Unitree `--vendor-root`、旧 discovery 或旧 play 命令。

## WebUI 原则

- UI 管理对象、Job、Artifact、验证和部署状态，不承担硬实时控制；
- 每个动作必须有等价的结构化 Job 输入、配置快照和结果；
- 页面显示 `installed`、`compatible`、`validated` 和 `active` 的真实区别；
- physical target 在 Driver、Calibration 和 Safety 完成前保持禁用。

详细产品和视觉约束见 [产品设计](../development/PRODUCT.md) 与 [UI 规范](../specifications/UI.md)。
