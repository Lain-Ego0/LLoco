# Unitree RL MJLab 精选 Vendor 方案

状态：已确认并已实施（历史基线记录）。精选 vendor 于 2026-08-20 导入
`vendor/unitree_rl_mjlab/` 并纳入 Git；保留技术价值和来源追踪，不把 RoboLab
建成 Unitree 上游仓库的完整镜像。当前开发进度以 `DEVELOPMENT_PLAN.md` 为准。

## 1. 当前事实

`vendor/unitree_rl_mjlab/` 已按 `VENDOR_MANIFEST.yaml` 完成精选导入，来源基线为 `unitreerobotics/unitree_rl_mjlab@1425b15f`（导入前已对本地完整工作副本做过完整 blob 哈希比对）。导入为独立 commit，随后以单独 commit 让 deploy/simulate 改为发现外部 ONNX Runtime 与 MuJoCo，替代被排除的预编译包。

该上游仓库不是通用 `mujocolab/mjlab` 引擎本体，而是基于 MJLab 构建的 Unitree 机器人训练、资产、仿真和部署项目。原样放入 RoboLab 主树会产生三个问题：

- 仓库内容和视觉素材过度 Unitree 化，容易让 RoboLab 看起来像品牌二次包装；
- 上游演示 GIF、预编译 runtime 和样例产物增加体积，但不构成平台核心；
- Unitree 专用部署目录容易被误认为 RoboLab 的通用机器人接口。

来源追踪并不要求完整携带上游宣传内容。Apache-2.0 允许在满足许可证、notice 和修改标记要求的前提下分发经过选择和修改的来源。

## 2. 最终选择

采用 **精选 vendor 导入（curated vendor import）**：

- 仍在 RoboLab 单一主仓库中管理，保持 one-stop；
- 只导入平台实际依赖或迁移所需的技术内容；
- 将上游放入明确命名空间 `vendor/unitree_rl_mjlab/`；
- 使用 manifest 记录 upstream commit、包含路径、排除路径和理由；
- RoboLab 通用代码不继续写入 vendor 目录。

这取代此前“完整 squash subtree 快照”的方案。因为完整快照一旦进入 Git 历史，即使之后删除 GIF/二进制，普通 clone 的历史体积仍然存在。

## 3. 目标目录边界

```text
vendor/
└── unitree_rl_mjlab/         # 精选的 Unitree 上游技术来源
    ├── src/                  # Unitree tasks 与所需机器人资产
    ├── scripts/              # 当前仍由 adapter 调用的训练/回放工具
    ├── simulate/             # 经依赖清理后的 sim-to-sim 源码/配置
    ├── deploy/               # 经精简的 Unitree legacy deploy 源码
    ├── LICENCE
    ├── UPSTREAM.md
    └── VENDOR_MANIFEST.yaml

integrations/
└── unitree/                  # RoboLab Unitree Driver/Profile 适配

packages/
└── mjlab_adapter/            # RoboLab 对通用 MJLab/任务的稳定适配层

runtime/                      # 与厂商无关的部署、推理、安全和遥测（仅有入口说明）
robots/                       # 与 Skill 解耦的 Robot Profile（仅有入口说明）
docs/                         # RoboLab 自己的正式文档
```

上述目录边界中，`vendor/`、`packages/`、`robots/` 与 `docs/` 已存在；
`packages/mjlab_adapter/` 已实现首版 task 发现与受控 play 调用，`integrations/`
和 `runtime/` 仍主要是后续迁移边界，具体进度按 `DEVELOPMENT_PLAN.md` 跟踪。

## 4. 首次导入保留清单

首次导入已按下列原则实施，实际路径集合以 `VENDOR_MANIFEST.yaml` 的 `includedPaths` 为准。保留项以“当前平台真实依赖或迁移参考”为标准：

| 内容 | 处理 | 理由 |
|---|---|---|
| `src/tasks/` | 保留 | 现有 velocity/tracking task 与配置基础 |
| `src/assets/robots/` | 按首批支持模型保留 | G1 等 simulation-only Profile 和训练需要 |
| `src/assets/motions/` | 仅保留开发源数据或迁入 Skill | 运行时动作属于 MotionSkill |
| `scripts/` | 保留 | MVP adapter 需要 train/play/list/convert 入口 |
| `simulate/` 源码和 Unitree 配置 | 保留 | sim-to-sim 和 viewer 复用基础 |
| `deploy/include/`、机器人部署源码 | 暂时保留 | 提取通用 runtime 和 Unitree Driver 的迁移参考 |
| `deploy/thirdparty/cnpy` | 需要时保留源码和许可证 | 当前 mimic 数据读取依赖 |
| `setup.py`、`.gitignore` | 按实际适配保留 | 安装和开发辅助 |
| 根 `LICENCE` 与相关第三方许可证 | 必须保留 | 法律和来源要求 |

“按首批支持模型保留”不意味着立即删除所有非 G1 模型。首次实施前应由 import manifest 列出实际包含集合，避免无意破坏已有 task；但新增非 Unitree 模型不得继续进入该 vendor 的 Unitree assets 命名空间。

## 5. 首次导入排除清单

实际排除路径与理由以 `VENDOR_MANIFEST.yaml` 的 `excludedPaths` 为准。

| 内容 | 处理 | 替代方案 |
|---|---|---|
| 上游 `README.md`、`README_zh.md` | 不作为 active vendor 文档导入 | `UPSTREAM.md` 链接原始仓库和固定 commit |
| `doc/`，尤其 `doc/gif/` | 排除 | RoboLab 根 `docs/`；演示需要时链接上游页面 |
| `deploy/thirdparty/onnxruntime-linux-*` | 排除预编译分发包 | Conda/安装器下载固定 ONNX Runtime 版本并校验 |
| `simulate/mujoco/lib/` 等预编译 runtime | 原则上排除 | Conda/安装器提供固定 MuJoCo 依赖；具体构建先验证 |
| `deploy/robots/**/config/policy/**/exported/` | 排除 | 策略放入 RoboLab-Skill 的 MotionSkill |
| mimic/动作运行产物 `.npz` | 迁入相应 Skill | manifest 固定哈希和机器人兼容性 |
| 上游品牌演示媒体、重复许可证副本 | 排除或整合 | 根第三方声明和 vendor LICENSE/NOTICE |
| build、logs、wandb、outputs | 排除 | 本地 `var/` 或构建目录 |

排除预编译库前必须先让 setup/doctor 能安装并发现等价版本，避免为了仓库形象破坏可运行性。必要时可以在短期迁移分支保留，但不能进入最终 curated vendor commit。

## 6. `deploy/` 的长期定位

`vendor/unitree_rl_mjlab/deploy/` 只作为 **Unitree legacy deployment backend / migration reference**：

- 不把新自制机器人复制到 `deploy/robots/<name>`；
- Unitree 专用 DDS、mode、motor ID 和 message type 迁入 `integrations/unitree/`；
- 共享 FSM、ONNX runner、observation/action、安全和遥测迁入根 `runtime/`；
- policy、motion 和 deploy params 由 Skill 安装器提供；
- 在新 runtime 达到功能等价前保留必要上游源码；
- 功能迁移完成后可以进一步缩减 legacy deploy。

因此保留部署技术不等于把 RoboLab 定位为 Unitree 平台。Unitree 是首个适配器，不是通用 API 的命名来源。

## 7. 品牌与文档边界

- 根 README 以 RoboLab 工作流、Skill、Robot Profile 和 WebUI 为主体；
- 不在根 README/WebUI 使用 Unitree GIF、Logo 或宣传式实机矩阵；
- `docs/` 不复制上游产品文案，只写 RoboLab 当前行为；
- Unitree 只在兼容矩阵、integration、致谢和第三方声明中出现；
- G1/Go2 与未来自制机器人通过同一 Robot Profile/Skill API 展示；
- 不暗示 Unitree 对 RoboLab 的官方认可或合作关系。

## 8. `VENDOR_MANIFEST.yaml` 记录内容

`vendor/unitree_rl_mjlab/VENDOR_MANIFEST.yaml` 已建立，记录 name、source、revision、license、importMode、实际 includedPaths/excludedPaths（含排除理由）、robolabPatches 和 verifiedAt。

导入时已验证所有被包含文件与固定 upstream commit 一致。排除项不是“文件丢失”，而是 manifest 中可审计的产品决策。当前 `robolabPatches` 包含：新增 `deploy/cmake/FindONNXRuntime.cmake`，以及让 deploy/simulate 的 CMake 发现外部 ONNX Runtime/MuJoCo 而不是使用被排除的预编译包。

## 9. Git 历史

```text
1. docs: freeze RoboLab platform baseline                       # 已完成
2. vendor(unitree): import curated unitree_rl_mjlab@1425b15f    # 已完成
   build(unitree): discover external simulation runtimes        # 已完成（RoboLab patch 独立 commit）
3. skill: publish G1 velocity artifacts in RoboLab-Skill        # 未开始
4. platform: add schemas, CLI, worker and adapters              # schemas and CLI completed in B1; worker and adapters remain pending
5. runtime: migrate generic deployment capabilities out of vendor  # 未开始
```

vendor commit 不混入 RoboLab 重构。平台修改和路径迁移放后续独立 commit，便于审查上游来源与本项目贡献。

## 10. 后续同步

1. 获取新的 upstream commit/range；
2. 根据 `VENDOR_MANIFEST.yaml` 生成同一 include/exclude 规则下的候选树；
3. 审核上游许可证、依赖和目录变化；
4. 形成纯 vendor sync commit；
5. 再修复 RoboLab adapter/runtime 冲突；
6. 运行 task discovery、MJCF load、Skill compatibility 和部署编译回归；
7. 更新 revision、排除理由和第三方声明。

禁止用新的 ZIP 直接覆盖 vendor 目录，也禁止把被排除的宣传媒体或二进制在同步时悄悄带回主仓库。
