# 当前开发计划

状态：R0、R1 已完成，当前 `ACTIVE_STAGE: R2`。本文是阶段状态、依赖关系和执行入口的唯一权威来源。

## 阶段顺序

```text
R0 基座固定与旧栈退役 ✅
  ↓
R1 MJLab 定制工具链基础 ✅
  ↓
R2 自研机器人接入 ✅
  ↓
R3 运动策略工作流 ⬜
  ↓
R4 Skill/平台/验证整合 ⬜
  ↓
R5 Runtime/部署适配 ⬜
  ↓
R6 多机器人证明与发布 ⬜
```

## 当前事实

- MJLab 1.6 位于根目录 `mjlab/`，upstream 为 `v1.6.0` / `0fb8a681136be94ffc636a3dd423cabb97d91f10`；
- Unitree active 栈已删除；不能恢复旧 adapter、G1 Profile 或 `--vendor-root`；
- 默认环境为 `robolab-mjlab16`：Python 3.11.15、Torch 2.9.0、CUDA 12.8、Warp 1.14.0、MuJoCo/MuJoCo-Warp 3.11.0、RSL-RL 5.4.2；
- R0 smoke、路径、symlink、NOTICE、Markdown link 和 contract 结果已记录；R1 基线与退出证据见归档报告；
- R2 的唯一参考机器人是 FireDog 2.2，目标为 simulation-first 接入；GPU 训练当前不启动。

## 当前阶段

R2 是当前已完成、尚未切换到 R3 的阶段。R3–R6 只用于理解后续目标，不得提前实现阶段性大功能。

## 执行与切换规则

1. 执行者先读取 [开发主线](../development/MAINLINE.md)、[开发约束](../development/CONSTRAINTS.md) 和当前阶段文件；
2. 只实现当前阶段工作项。后续阶段允许做只读调研，不允许创建其正式实现或把占位结果登记为完成；
3. 工作项只有同时具备实现、测试和可定位证据时才能标记完成；CPU、GPU、Viewer 与硬件证据必须分别记录；
4. 阶段退出条件全部满足后，先更新完成报告与本页，再激活下一阶段；不能只修改某个阶段文件中的状态；
5. 阶段状态发生冲突时，以本页为准，并立即修正文档冲突；归档、聊天记录和 Git 历史不覆盖本页状态；
6. 当前 GPU 被占用只阻止需要 GPU 的验收，不阻止 R2 的 CPU contract、Inspector、配置和仿真闭环工作。

## 计划文件

- [R1：MJLab 定制工具链基础](R1_MJLAB_TOOLCHAIN.md)
- [R2：自研机器人接入](R2_ROBOT_ONBOARDING.md)
- [R3：运动策略工作流](R3_MOTION_POLICY.md)
- [R4：Skill、平台与验证整合](R4_PLATFORM_INTEGRATION.md)
- [R5：部署 Runtime 与适配](R5_DEPLOYMENT_RUNTIME.md)
- [R6：多机器人证明与发布](R6_MULTI_ROBOT.md)

已完成阶段和废止路线只保留在 [压缩归档](../archive/README.md)；旧版完整计划从 Git 历史审计，不在当前树中保留，避免产生第二套执行入口。
