# Phase 0–1 平台骨架摘要

状态：历史完成摘要。完整 B1–B7 工作表和旧验收命令仅保留在 Git 历史中。

## 已形成能力

- RobotProfile、JointSet、SkillPackage v1alpha1 schema；
- 关节映射、Skill/Profile 兼容性和 lint；
- MotionSkill、PlatformSkill、AgentSkill 基础模型；
- Skill 安装、固定 revision、权限审查、独立进程 Job 和 action registry；
- FastAPI、SQLite、Artifact Store、Local Worker 和最小 React WebUI；
- CLI check/skill/agent/serve 等平台入口；
- CPU contract suite 和样板 Skill 骨架。

## 未完成内容

当时基于 Unitree/MJLab 1.2 的 G1 trained play、远程 catalog CI、sim-to-sim 和部署 Runtime 没有形成主线闭环。R0 后这条
路线已经退役，不能继续执行旧 B7 命令补齐。

## 当前继承

保留平台控制面、Skill、schema、Worker、API、WebUI 和 Artifact 基础；MJLab motion toolchain 从 R1 基于根目录 `mjlab/`
重新建立。
