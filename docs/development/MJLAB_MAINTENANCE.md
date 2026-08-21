# MJLab 维护规则

RoboLab 的 MJLab 下游代码位于根目录 `mjlab/`。上游记录见 [`mjlab/UPSTREAM.md`](../../mjlab/UPSTREAM.md)，修改账本见
[`mjlab/ROBOLAB_CHANGES.md`](../../mjlab/ROBOLAB_CHANGES.md)。

## 修改分类

- `upstream-import`：同步上游文件，不改变 RoboLab 语义；
- `robolab-specific`：RoboLab 运动控制工具链需要的下游行为；
- `upstreamable`：有候选上游价值的修改；
- `temporary-compatibility`：有明确删除条件的临时兼容。

## 强制规则

1. 上游同步、RoboLab 行为修改和平台代码修改使用可区分提交；
2. 每项行为变化写入 change ledger，包含文件、目的、兼容影响、测试和回滚；
3. 运行 MJLab smoke、contract 和代表性 task 回归；
4. 不把平台 API、Skill Registry、数据库或厂商 SDK 放进 MJLab 核心；
5. 升级前固定旧 revision，升级后保留冲突、回归和回滚记录。
