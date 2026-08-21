# Skill 与机器人使用指南

RoboLab 将“机器人是什么”和“机器人能做什么”分开：

- Robot Profile 描述 MJCF、关节、执行器、传感器、控制周期、能力和安全边界；
- MotionSkill 描述策略、任务、输入输出 schema 和部署参数；
- PlatformSkill 提供检查、转换或评测工具；
- AgentSkill 提供 Agent 工作流，但不能绕过平台直接发送电机命令。

Unitree G1 不再是当前 Profile 样板。R2 选定真实非 Unitree 参考机器人前，文档示例中的 `test.reference_biped` 或
`community.reference_biped` 都只是占位符，不能当作已安装或已验证的机器人。

兼容性必须同时检查 Profile、Task、JointSet、observation/action schema、控制模式、频率、Artifact hash 和安全能力。
不得仅凭机器人名称或关节数量判断兼容。
