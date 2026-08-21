# Unitree 早期路线摘要

状态：已退役。本文只记录事实，不提供当前安装或运行命令。

## 历史来源

- Repository: `https://github.com/unitreerobotics/unitree_rl_mjlab`
- Revision: `1425b15f73bd4095f0df53709d7c389c3eb9e790`
- License: Apache-2.0

早期仓库曾精选导入 Unitree 任务、机器人资产、sim-to-sim 和 deployment 源码，并对外部 ONNX Runtime/MuJoCo 发现做过
局部补丁。该路线用于快速验证 G1、Skill 和平台 Job，但把较旧 MJLab、厂商 task ID、vendor path 和部署目录带入了平台。

## 退役结果

R0 删除了 Unitree vendor、adapter、G1 Profile、Integration、CLI/API 默认参数和专用测试。当前发布物不承诺旧 checkpoint、
旧 task 或 Unitree sim-to-sim 兼容。

未来如果支持 G1，必须作为新的商业机器人适配器，通过当前 Robot Profile、Task、Artifact、Skill 和 Runtime 契约重新接入，
并重新确认资产与模型许可证。
