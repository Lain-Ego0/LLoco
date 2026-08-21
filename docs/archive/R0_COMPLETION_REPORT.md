# R0 完成报告

状态：完成，2026-08-21。本文是不可执行的结果摘要，完整变更由 Git diff/history 保存。

## 目标

固定 MJLab 1.6 下游基座，移除 Unitree-first 技术路线，建立可维护环境、来源、测试和修改规则。

## 完成结果

- `vendor/mjlab/` 已整体迁移到仓库一级目录 `mjlab/`；
- upstream 固定为 `mujocolab/mjlab@0fb8a681136be94ffc636a3dd423cabb97d91f10`，tag `v1.6.0`；
- `mjlab/UPSTREAM.md` 和 `mjlab/ROBOLAB_CHANGES.md` 已建立；
- Unitree vendor、MJLab 1.2 adapter、G1 Profile、Unitree Integration 和专用测试已从 active tree 删除；
- CLI/API 已移除 `vendor_root`、旧 discovery/play 和 vendor health check；
- G1 fixture 已替换为明确的中性 test fixture；
- 当前 NOTICE 只保留 active MJLab 和实际分发内容；
- 默认环境为 `robolab-mjlab16`。

## 环境

- Python 3.11.15
- Torch 2.9.0
- CUDA 12.8
- Warp 1.14.0
- MuJoCo 3.11.0
- MuJoCo-Warp 3.11.0
- RSL-RL 5.4.2

## 验证

- `python -m pip install -e mjlab`：通过；
- `pip check`：通过；
- MJLab import 和 editable path：通过；
- `list_envs`：发现 6 个上游任务；
- cartpole 最小模型加载：通过；
- contract suite：`105 passed, 1 warning`；
- Markdown link、破损 symlink、禁止引用、路径断言和 `git diff --check`：通过。

GPU 只做只读资源检查。RTX 4060 当时被外部 IsaacLab 进程占用，未启动 RoboLab 训练、Viewer 或随机策略验收。

## 后续

R0 不证明 RoboLab 已经拥有 train/play/evaluate/export 工具链，也不证明真实机器人或策略闭环。后续从 R1 开始，见
[`plans/README.md`](../plans/README.md)。
