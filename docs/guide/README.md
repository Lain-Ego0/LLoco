# 使用指南

本目录回答“用户现在如何使用 RoboLab”。先阅读本页，再按需要进入环境、CLI/WebUI、机器人和 Skill 指南。

## 当前可用边界

- RoboLab 是本地优先工作台，默认服务监听 `127.0.0.1`；
- MJLab 1.6 已固定在仓库根目录 `mjlab/`，但 RoboLab 自有 train/play/evaluate/export 主线仍在 R1；
- Unitree 旧栈已从 active tree 删除，不提供旧 G1 play 或 Unitree legacy 安装入口；
- 当前平台控制面、Skill、CLI、Worker、API、WebUI 和基础 Artifact 能力可用；
- 真实策略训练、验证和部署 Runtime 以 `docs/plans/README.md` 的当前阶段为准。

## 阅读路径

1. [环境与安装](ENVIRONMENT.md)
2. [CLI 与 WebUI](CLI_AND_WEB.md)
3. [机器人与 Skill](SKILLS_AND_ROBOTS.md)
4. [开发主线](../development/MAINLINE.md)
5. [当前阶段计划](../plans/README.md)
