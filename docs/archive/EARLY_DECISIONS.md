# 早期决策摘要

状态：历史摘要。完整 D-001–D-048 表格和问答过程由 Git 历史保存。

## 继续有效

- 本地优先、单用户、loopback 服务；
- Conda 环境，首版不引入 Docker、云平台或多租户；
- Robot Profile 与 Skill 分离；
- 可执行 Skill 由 Worker 独立进程运行；
- WebUI 不承担硬实时控制；
- Artifact、配置和部署输入必须固定 revision/hash；
- 根项目 MIT 与第三方许可证分别管理；
- physical target 缺少安全能力时不可激活。

## 已被取代

- Unitree G1 作为默认黄金基线；
- MJLab 1.2/Unitree adapter 作为长期 backend；
- G1 checkpoint 阻塞所有后续开发；
- 双后端等价后才允许 MJLab native；
- 不直接修改 MJLab；
- 自研机器人后置。

新的有效方向是根目录 `mjlab/` 的 MJLab 1.6 下游定制、自研机器人主线、独立指标验收和安全 Runtime。规范性约束见
[`development/CONSTRAINTS.md`](../development/CONSTRAINTS.md)。
