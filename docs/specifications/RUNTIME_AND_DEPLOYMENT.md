# Runtime 与部署规范

Runtime 是独立于 WebUI/API 和具体厂商 SDK 的部署数据面。它负责策略推理、FSM、watchdog、遥测、Driver 边界和安全回退。

## 状态机

```text
DRAFT -> COMPATIBLE -> OFFLINE_VALIDATED -> SIM_VALIDATED
      -> OPERATOR_ARMED -> EDGE_CONFIRMED -> ACTIVE
      -> STOPPING -> SAFE
```

任意 heartbeat loss、命令超时、姿态越界、Driver 断开或人工停止都必须进入 `SAFE` 或明确的安全降级状态。

## DeploymentPlan 必须固定

- Robot Profile/version；
- Task 和 PolicyArtifact hash；
- observation/action/deployment schema；
- Runtime 参数、控制频率和 fallback；
- required gates、target 和 capability；
- 代码、MJLab upstream 和环境 revision。

在 R5 完成独立 Runtime、simulation driver、故障注入和 stop/safe 之前，不开放 physical motor command。
