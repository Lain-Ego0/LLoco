# B7 MVP 验收记录

状态：进行中；创建日期 2026-08-20。此记录区分已经完成的 CPU 闭环和依赖
本地 MJLab/Viser 环境的手动验证，禁止用前者替代后者。

## 环境基线

- 工作目录：`/home/lxy/RoboLab`
- Conda 环境：`robolab`
- 平台包：`robolab-schemas`、`robolab-core`、`robolab-mjlab-adapter`、
  `robolab-api`、`robolab-worker`、`robolab-cli` 均以 editable 方式安装。
- 启动命令：`robolab serve`；服务仅输出 loopback URL。

## 已完成的演练

| 路径 | 结果 | 证据 |
|---|---|---|
| G1 MotionSkill 安装与兼容性 | 通过 | policy ONNX/deploy SHA-256、`unitree.g1.29dof` Profile 与 `Unitree-G1-Flat` task 契约测试 |
| MJCF Inspector | 通过 | 从 API 安装 PlatformSkill，经 Worker 运行并产生 `report.json`、`report.md`、`robot_profile.draft.yaml` |
| Robot Onboarding AgentSkill | 通过 | 安装、manifest 校验与 `.agents/skills/robot-onboarding/` Codex export 测试 |
| API/WebUI | 通过 | Jobs、logs、cancel、Artifact lineage、静态 WebUI 资源 API 测试 |

最新 CPU 测试：`112 passed`（2026-08-20）。

本地启动烟测：已通过 `/home/lxy/miniconda3/envs/robolab/bin/robolab serve`，自动
分配端口并监听 `http://127.0.0.1:33045`（测试超时后正常关闭）。

## CI 发布前置条件

`.github/workflows/cpu-contract.yml` 会同时 checkout RoboLab 与官方
`RoboLab-Skill` catalog 并运行全部 contract tests。当前工作站的后者含有尚未提交
的 MJCF Inspector 与 Robot Onboarding 包改动；在这些 catalog 改动发布到其远程
仓库前，GitHub CI 无法复现本地的三个样板 Skill 路径。因此 B7.2 保持“进行中”，
尽管本地 CPU suite 已通过。

## MJLab/Viser 本地验证

`robolab` 环境现已包含 MJLab 运行依赖，检测到：

- `mjlab==1.2.0`、`mujoco==3.5.0`、`torch==2.13.0+cu130`、`viser==1.1.0`、
  `warp-lang==1.12.0`、`scipy==1.17.1`；
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU；
- task 发现已通过：使用 `robolab mjlab tasks --keyword G1` 发现
  `Unitree-G1-Flat` 及其他 11 个 G1 task。

此前 `warp==1.16.0` 缺少 vendor 所调用的 `warp.context` API；现已在 `robolab`
环境固定为 `warp==1.12.0`，最小 zero-policy play 已进入 CUDA kernel 编译阶段，未再
出现该 API 崩溃。首次编译超过 45 秒的烟测窗口，尚未人工确认 Viser 页面。

以下项目仍待完成，故不能标记 MVP 完成：

1. 等首次 CUDA kernel 编译完成后，确认 Viser URL/窗口正常启动，检查 viewer session、
   模型标识和退出后的 Job 状态；
2. 从 WebUI 或等价 CLI 创建 G1 velocity `play` Job，确认日志、停止与结果归档；
3. 保存本次 play 的 config snapshot、stdout/stderr、events、result 和可选视频的
   artifact hash；
4. 记录最终 GPU/CPU、MuJoCo、MJLab、Viser、Warp 的具体版本与结果。

完成所有条目后，将本表与 `docs/DEVELOPMENT_PLAN.md` 的 B7.1/B7.3 状态更新为
`✅`；在此之前 MVP 仅处于 CPU 闭环就绪状态。
