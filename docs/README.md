# RoboLab 文档入口

本目录严格按阅读和执行顺序组织。除 `plans/README.md` 外，其他文档不定义当前阶段状态；`archive/` 不具有执行权。

## 推荐阅读顺序

```text
guide/使用指南
  ↓
development/开发主线
  ↓
development/开发约束与稳定规范
  ↓
plans/当前阶段与后续阶段计划
  ↓
archive/历史完成报告
```

## 1. 使用指南

- [使用指南总览](guide/README.md)
- [环境与安装](guide/ENVIRONMENT.md)
- [CLI 与 WebUI](guide/CLI_AND_WEB.md)
- [Skill 与机器人](guide/SKILLS_AND_ROBOTS.md)

## 2. 开发主线

- [RoboLab 开发主线](development/MAINLINE.md)：项目定义、技术路线和最终证明目标
- [总体架构](development/ARCHITECTURE.md)：Platform、MJLab、Runtime 和 Robot Adaptation 分层
- [产品设计](development/PRODUCT.md)：用户、页面、工作流和产品边界
- [MJLab 维护规则](development/MJLAB_MAINTENANCE.md)：upstream、下游修改、测试和回滚

## 3. 开发约束与稳定规范

- [开发约束](development/CONSTRAINTS.md)：跨阶段强制规则，自动 Agent 必须遵守
- [Skill Package](specifications/SKILL.md)
- [Robot Profile](specifications/ROBOT_PROFILE.md)
- [Runtime 与部署](specifications/RUNTIME_AND_DEPLOYMENT.md)
- [Agent 集成](specifications/AGENT.md)
- [UI 规范](specifications/UI.md)
- [上游、许可证与致谢](legal/UPSTREAM_AND_ACKNOWLEDGEMENTS.md)

这些规范定义稳定对象和接口；它们不能覆盖当前阶段计划，也不能把未实现能力描述成已完成。

## 4. 当前与后续计划

- [当前开发计划](plans/README.md)：唯一阶段状态来源，当前 `ACTIVE_STAGE: R2`
- [R1：MJLab 定制工具链基础](plans/R1_MJLAB_TOOLCHAIN.md)
- [R2：自研机器人接入](plans/R2_ROBOT_ONBOARDING.md)
- [R3：运动策略工作流](plans/R3_MOTION_POLICY.md)
- [R4：Skill、平台与验证整合](plans/R4_PLATFORM_INTEGRATION.md)
- [R5：部署 Runtime 与适配](plans/R5_DEPLOYMENT_RUNTIME.md)
- [R6：多机器人证明与发布](plans/R6_MULTI_ROBOT.md)

## 5. 历史归档

- [归档规则](archive/README.md)
- [R0 完成报告](archive/R0_COMPLETION_REPORT.md)
- [Phase 0–1 平台骨架摘要](archive/PHASE_0_1_SUMMARY.md)
- [Unitree 早期路线摘要](archive/LEGACY_UNITREE_SUMMARY.md)
- [早期决策摘要](archive/EARLY_DECISIONS.md)

历史完整文件、旧命令和原始决策可从 Git 历史查看，不复制到当前执行文档。
