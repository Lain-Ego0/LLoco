# Unitree 整栈退役与 MJLab 根目录迁移方案

状态：2026-08-21 已确认，等待代码执行。本文是 R0 中删除 Unitree 旧路线和迁移 MJLab 目录的强制执行说明。

## 1. 最终决策

RoboLab 不再在 active working tree 中保留 Unitree RL MJLab 兼容栈。Unitree 早期选型的源码、adapter、G1 Profile、
CLI/API 默认参数和专用测试将整体退役。历史来源仍可从 RoboLab Git 历史和固定上游 commit 恢复，不另建归档副本。

同时，MJLab 1.6 下游定制基座从：

```text
vendor/mjlab/
```

迁移为仓库一级目录：

```text
mjlab/
```

“放在根目录”严格表示 `/home/lxy/RoboLab/mjlab/` 是一级子目录。不得把 MJLab 的 `pyproject.toml`、`src/`、`tests/`、
`docs/` 等文件打散或合并到 `/home/lxy/RoboLab/` 根层，因为 RoboLab Platform 与 Customized MJLab 是同一仓库中的两个
独立发布和测试边界。

## 2. 为什么不保留 Unitree legacy

当前 `vendor/unitree_rl_mjlab/` 约 227 MiB，其中绝大部分是多种 Unitree 机器人资产。继续保留会产生：

- CLI/API 默认值继续把 Unitree 目录当作 MJLab；
- G1 Profile 通过软链接依赖 vendor 资产；
- adapter 和 tests 继续鼓励旧 task ID 与 vendor path；
- 小模型执行计划时容易把“兼容参考”误解为“先完成 Unitree”；
- 发布包需要继续携带与主线无关的第三方源码和 NOTICE；
- 项目对外仍容易被理解为 Unitree RL MJLab 的平台包装。

既然 MJLab 1.6 主线允许重新实现 G1 适配，将旧栈保留在 active tree 的收益低于维护和语义成本。

## 3. 必须整体删除的 active 路径

代码执行批次必须删除：

```text
vendor/unitree_rl_mjlab/
packages/mjlab_adapter/
robots/unitree.g1.29dof/
integrations/unitree/
```

如果 `vendor/mjlab/` 已迁移到 `mjlab/`，并且 `vendor/` 不再包含其他内容，则同时删除空的 `vendor/` 目录。

不得只删除 `vendor/unitree_rl_mjlab/` 而保留失效的 G1 symlink、adapter 或默认参数。不得从旧 vendor 私自复制 G1 mesh
到其他目录来延长旧耦合。如果未来重新支持 G1，必须作为 R6 成品机器人适配，通过 MJLab 1.6 公共 Robot Profile、Task、
Artifact 和 Runtime 契约重新导入，并重新核查资产许可证。

## 4. 必须删除或重构的代码依赖

### 4.1 CLI

移除仅服务旧 vendor 的：

- 全局或子命令 `--vendor-root` 默认值；
- `robolab mjlab tasks` 中对 `robolab_mjlab_adapter.discover_tasks` 的调用；
- `robolab mjlab play` 中对 vendor `scripts/play.py` 的调用；
- legacy Job 中将 vendor root 加入 `PYTHONPATH` 和 allowed paths 的逻辑。

R1 的 Customized MJLab 1.6 CLI 未完成前，可以暂时不提供 `robolab mjlab tasks/play`，但不得保留一个名称正常、内部仍调用
Unitree 的假主线命令。后续同名命令必须直接绑定 `mjlab/` 的 RoboLab toolchain。

### 4.2 API 和 health

移除：

- `create_app(..., vendor_root=...)` 参数和 `app.state.vendor_root`；
- 通过 `vendor_root/scripts/list_envs.py` 判断 MJLab 健康度的逻辑；
- API 测试中的 `missing-vendor` 参数。

新的 MJLab health 应在 R0/R1 检查：

- `mjlab` Python package 能否导入；
- toolchain/upstream revision；
- registry/minimal smoke test 状态；
- 依赖和 GPU 状态。

### 4.3 Python 包和安装入口

- 从根安装命令、环境文档和任何 workspace 配置中移除 `packages/mjlab_adapter`；
- 删除 `robolab-mjlab-adapter` 的 editable package；
- 不把 adapter 改名为一个看似通用、实际仍依赖 Unitree 的包；
- R1 新包必须直接围绕 `mjlab/` 和 RoboLab task/toolchain 契约建立。

### 4.4 Tests 和 fixtures

删除或重构：

- `tests/contract/test_mjlab_adapter.py`；
- 依赖 `robots/unitree.g1.29dof/` 或 vendor G1 XML 的测试；
- 公共 schema/CLI 测试中的 `Unitree-G1-Flat`、`unitree.g1.29dof` 和 G1-only fixture；
- 依赖外部 `RoboLab-Skill` G1 velocity 样板的强制 e2e 路径。

替代原则：

- schema、compatibility 和 lint 单元测试使用明确标注为 test fixture 的中性机器人 ID，例如 `test.reference_biped`；
- test fixture 不得被文档称为已经接入的真实机器人；
- 需要真实 MJCF 的集成测试必须等待 R2.1 选定合法参考机器人，或使用 MJLab 上游明确允许测试的内置资产；
- 不为维持原测试数量而制造假 checkpoint、假 MotionSkill 或假部署证据。

## 5. 文档和许可证处理

### 5.1 保留的历史事实

以下内容可以保留：

- Unitree 上游仓库 URL 和 commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`；
- 2026-08-20 精选导入、补丁和 B7 运行证据；
- Unitree 早期选型为何被取消的决策记录；
- Git 历史中的原始第三方文件和许可证。

历史文档必须明确写明对应路径已从 active tree 删除，不能提供已失效的当前安装命令。

### 5.2 当前发布文档

目录删除后必须更新：

- `README.md`；
- `THIRD_PARTY_NOTICES.md`；
- `docs/reference/UPSTREAM_AND_ACKNOWLEDGEMENTS.md`；
- `docs/reference/ENVIRONMENT_SETUP.md`；
- `docs/reference/ARCHITECTURE.md`；
- `docs/reference/SIMULATION_BACKEND_STRATEGY.md`；
- `docs/project/DEVELOPMENT_PLAN.md` 和 `ROADMAP.md`；
- `robots/README.md`、Skill 示例和文档索引。

如果发布物不再分发 Unitree vendor、cnpy、LodePNG 和 joystick-derived code，则当前 `THIRD_PARTY_NOTICES.md` 不应继续声称
这些文件存在于 active tree。历史归属可移动到历史维护记录；实际仍使用的第三方依赖才保留在当前 NOTICE。

## 6. MJLab 根目录迁移规则

应使用可追踪的 Git move 将整个目录从 `vendor/mjlab/` 移到 `mjlab/`，保持文件内容和历史可比较性。迁移提交只包含路径移动、
引用更新和必要的构建配置更新，不同时修改 MJLab 行为。

迁移后目标结构：

```text
RoboLab/
├── mjlab/
│   ├── UPSTREAM.md
│   ├── ROBOLAB_CHANGES.md
│   ├── LICENSE
│   ├── pyproject.toml
│   ├── src/mjlab/
│   └── tests/
├── packages/
├── services/
├── apps/
├── robots/
├── runtime/
└── docs/
```

迁移后：

- 默认 editable install 使用 `python -m pip install -e mjlab`；
- 所有 upstream 和 change ledger 路径改为 `mjlab/UPSTREAM.md`、`mjlab/ROBOLAB_CHANGES.md`；
- 根 RoboLab 和 `mjlab/` 可以有各自 `pyproject.toml`、测试和发布生命周期；
- `mjlab/src/mjlab` 继续保持上游 Python package 名称；
- 不创建 nested `.git`，也不把 MJLab 当作未固定 submodule；
- 是否未来拆为独立 `RoboLab-MJLab` 仓库，留到 R6 upstream sync rehearsal 后决定。

## 7. 强制执行顺序

1. 运行并记录删除前 contract suite；
2. 用单独提交将 `vendor/mjlab/` 移动为 `mjlab/`，更新所有路径引用，不修改行为；
3. 移除 CLI/API/health 对 Unitree vendor 的默认依赖；
4. 删除 adapter、G1 Profile、Unitree integration 和相关测试；
5. 删除 `vendor/unitree_rl_mjlab/`，必要时删除空 `vendor/`；
6. 更新当前 NOTICE、环境、README 和活动文档；
7. 将中性 test fixture 与真实 R2 参考机器人严格区分；
8. 运行 contract tests、Markdown link check、全仓 Unitree path 检索和 `git diff --check`；
9. 单独提交 Unitree retirement，不与 MJLab 行为定制混在同一个提交。

第 2 步和第 3–9 步可以是两个独立提交。不得先删除 vendor 目录、再让仓库处于 API/CLI 和 symlink 全部断裂的中间状态并
宣称任务完成。

## 8. 完成验收

全部满足后才可关闭退役任务：

```text
exists(mjlab/pyproject.toml) == true
exists(mjlab/UPSTREAM.md) == true
exists(vendor/mjlab) == false
exists(vendor/unitree_rl_mjlab) == false
exists(packages/mjlab_adapter) == false
exists(robots/unitree.g1.29dof) == false
exists(integrations/unitree) == false
```

并且：

- active code 中不存在 `vendor_root`、`Unitree-G1-*` 或 `robolab_mjlab_adapter`；
- active docs 中不存在可执行的 Unitree legacy 安装命令；
- 历史文档中的 Unitree 引用被清楚标记为已退役路径；
- 当前 contract suite 通过，测试数量允许因删除 legacy 专用测试而变化；
- `python -m pip install -e mjlab` 使用锁定依赖成功；
- MJLab 1.6 import 和最小 smoke test 通过；
- 当前工作树没有破损软链接；
- 当前 NOTICE 只列实际仍被分发或使用的第三方内容。

## 9. 小模型执行禁令

执行 Agent 不得：

- 只删 `vendor/unitree_rl_mjlab/` 而跳过代码去耦；
- 把 G1 mesh 复制到新 `mjlab/` 或 `robots/` 以规避删除；
- 把 `packages/mjlab_adapter` 改名后继续调用旧 Unitree 脚本；
- 为保持旧 contract test 数量而保留 Unitree fixture；
- 把 `test.reference_biped` 当成真实 R2 机器人；
- 在目录迁移提交中顺便修改 MJLab 运行行为；
- 删除历史来源、许可证事实或 Git 追踪记录；
- 将 `mjlab/` 的文件打散到 RoboLab 仓库根层。
