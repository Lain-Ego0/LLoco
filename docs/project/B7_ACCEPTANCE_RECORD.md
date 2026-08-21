# B7 MVP 验收记录

状态：进行中；创建日期 2026-08-20。本文是 B7 的验收证据记录，当前进度以
[`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md) 为准。此记录区分已经完成的 CPU 闭环和依赖
本地 MJLab/Viser 环境的手动验证，禁止用前者替代后者。

执行顺序严格对应 `DEVELOPMENT_PLAN.md`：**B7.1 → B7.2 → B7.3**。
本文只记录证据，不重新定义批次、依赖或完成状态。

## 环境基线

- 工作目录：`/home/lxy/RoboLab`
- Conda 环境：`robolab`
- 平台包：`robolab-schemas`、`robolab-core`、`robolab-mjlab-adapter`、
  `robolab-api`、`robolab-worker`、`robolab-cli` 均以 editable 方式安装。
- 启动命令：`robolab serve`；服务仅输出 loopback URL。

## 详细执行步骤

以下步骤用于重跑或补齐 B7 验收。每一步都应保存终端输出、Job 路径和产物哈希；
不要用 CPU contract tests 代替 MJLab/Viser 手动验证。

### 1. 固定环境

```bash
cd /home/lxy/RoboLab
conda activate robolab

python -m pip install -e packages/schemas -e packages/core -e packages/mjlab_adapter \
  -e services/api -e services/worker -e apps/cli
python -m pip install -e "packages/mjlab_adapter[mjlab-runtime]" \
  -e vendor/unitree_rl_mjlab

python -c 'from importlib.metadata import version; print("mjlab", version("mjlab"))'
python -c 'import mujoco; print("mujoco", mujoco.__version__)'
python -c 'import torch; print("torch", torch.__version__)'
python -c 'import viser; print("viser", viser.__version__)'
python -c 'import warp; print("warp", warp.__version__)'
python -c 'import scipy; print("scipy", scipy.__version__)'
python -c 'import torch; print("cuda", torch.cuda.is_available())'
python -c 'import torch; print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
git rev-parse HEAD
```

记录 Python、GPU、CUDA、MJLab、MuJoCo、Viser、Warp、Torch、SciPy 版本；
`warp-lang` 应固定为 `1.12.0`。

### 2. B7.1：三条 CPU 样板路径

```bash
cd /home/lxy/RoboLab
export B7_ROOT="$PWD/var/b7-acceptance-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$B7_ROOT"/{workspace,runs,skills,logs}
export CATALOG=/home/lxy/RoboLab-Skill
export PROFILE="$PWD/robots/unitree.g1.29dof/profile.yaml"
export G1="$CATALOG/skills/motion/g1_velocity"
export INSPECTOR="$CATALOG/skills/platform/mjcf_inspector"
export ONBOARDING="$CATALOG/skills/agent/robot_onboarding"
```

先运行 CPU contract suite：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/contract \
  |& tee "$B7_ROOT/logs/contract-tests.txt"
```

#### 3.1 G1 MotionSkill

```bash
robolab check "$G1/skill.yaml" --package-dir "$G1" \
  |& tee "$B7_ROOT/logs/g1-check.txt"
robolab check --skill "$G1/skill.yaml" --profile "$PROFILE" \
  |& tee "$B7_ROOT/logs/g1-compatibility.txt"
robolab skill install "$G1" --installed-root "$B7_ROOT/skills/installed" --json \
  | tee "$B7_ROOT/logs/g1-install.json"
robolab mjlab tasks --keyword G1 --json \
  | tee "$B7_ROOT/logs/g1-tasks.json"
```

必须确认 schema、许可证、artifact SHA-256、`unitree.g1.29dof` 兼容性和
`Unitree-G1-Flat` task 均通过，并记录安装内容 hash。

#### 3.2 MJCF Inspector

```bash
robolab check "$INSPECTOR/skill.yaml" --package-dir "$INSPECTOR" \
  |& tee "$B7_ROOT/logs/inspector-check.txt"
robolab skill install "$INSPECTOR" --installed-root "$B7_ROOT/skills/installed" --json \
  | tee "$B7_ROOT/logs/inspector-install.json"
cat "$INSPECTOR/schemas/inspect.input.json"
```

将待检验的 MJCF/XML 放入 `$B7_ROOT/workspace/`，按输入 schema 填写参数后运行：

```bash
robolab skill run "$INSPECTOR/skill.yaml" --runs-root "$B7_ROOT/runs" \
  --params '{"mjcf_path":"/absolute/path/to/model.xml"}' --wait \
  |& tee "$B7_ROOT/logs/inspector-run.txt"
```

检查 run 目录必须包含 `events.jsonl`、`result.json`、
`artifacts/report.json`、`artifacts/report.md` 和
`artifacts/robot_profile.draft.yaml`，并保存哈希：

```bash
find "$B7_ROOT/runs" -type f -print0 | sort -z | xargs -0 sha256sum \
  | tee "$B7_ROOT/logs/inspector-artifact-sha256.txt"
```

#### 3.3 Robot Onboarding AgentSkill

```bash
robolab check "$ONBOARDING/skill.yaml" --package-dir "$ONBOARDING" \
  |& tee "$B7_ROOT/logs/onboarding-check.txt"
robolab skill install "$ONBOARDING" --installed-root "$B7_ROOT/skills/installed" --json \
  | tee "$B7_ROOT/logs/onboarding-install.json"
robolab agent export "$ONBOARDING" --target codex \
  --target-root "$B7_ROOT/agents/skills" --json \
  | tee "$B7_ROOT/logs/onboarding-export.json"
find "$B7_ROOT/agents/skills/robot-onboarding" -maxdepth 2 -type f \
  | sort | tee "$B7_ROOT/logs/onboarding-exported-files.txt"
```

确认导出目录包含 `SKILL.md`、`skill.yaml` 和 `references/`。

### 3. B7.2：发布 catalog 并运行 CPU CI

在同级 `../RoboLab-Skill` 仓库确认三个样板 Skill 已提交：

```bash
cd /home/lxy/RoboLab-Skill
git status --short
git log -1 --oneline
git tag --list
```

确认以下目录存在并已推送到远程：

```text
skills/motion/g1_velocity/
skills/platform/mjcf_inspector/
skills/agent/robot_onboarding/
catalog.yaml
```

如尚未发布，提交并推送 catalog；MotionSkill 的 `g1-velocity-v0.1.0` tag
必须指向包含 `policy.onnx` 和 `deploy.yaml` 的提交：

```bash
git add catalog.yaml skills
git commit -m "feat: publish MVP sample skills"
git push origin main
git tag -a g1-velocity-v0.1.0 -m "G1 velocity 0.1.0"
git push origin g1-velocity-v0.1.0
```

在 RoboLab 提交或推送后，等待 `.github/workflows/cpu-contract.yml` 成功。
记录 GitHub Actions URL、RoboLab commit、RoboLab-Skill commit 和测试结果。

### 4. B7.3：MJLab / Viser 手动验证

先验证 task 发现：

```bash
robolab mjlab tasks --keyword G1 \
  | tee "$B7_ROOT/logs/mjlab-g1-tasks.txt"
```

运行最小 zero-policy play。首次运行可能编译 CUDA kernel，必须等待进程完成或稳定
启动 viewer，不能用短超时判定失败：

```bash
robolab mjlab play Unitree-G1-Flat --agent zero --viewer viser \
  --num-envs 1 --runs-root "$B7_ROOT/runs" --wait \
  |& tee "$B7_ROOT/logs/g1-zero-play.txt"
```

确认 Viser URL、viewer session、G1 模型和 Job 状态；保存截图或 session 记录。

若策略/checkpoint 已准备好，再执行实际 trained velocity 路径：

```bash
robolab mjlab play Unitree-G1-Flat --agent trained --viewer viser \
  --num-envs 1 --checkpoint-file /absolute/path/to/model.pt --video \
  --runs-root "$B7_ROOT/runs" --wait \
  |& tee "$B7_ROOT/logs/g1-trained-play.txt"
```

若 checkpoint 不在本地，可将 `--checkpoint-file /absolute/path/to/model.pt` 替换为
`--wandb-run-path <WandB-run-path>`。`--video` 是 RoboLab CLI 的布尔开关，adapter 会向
vendor 传递其所需的 `--video True`。

`trained` 路径未成功时，不得用 zero-policy 代替 G1 Velocity MotionSkill 通过；
应将其记录为 B7 阻塞项。

保存所有 run 证据：

```bash
find "$B7_ROOT/runs" -type f | sort | tee "$B7_ROOT/logs/all-run-files.txt"
find "$B7_ROOT/runs" -type f -print0 | sort -z | xargs -0 sha256sum \
  | tee "$B7_ROOT/logs/all-run-sha256.txt"
find "$B7_ROOT/runs" -name result.json -exec sh -c \
  'echo "--- $1"; cat "$1"' _ {} \; | tee "$B7_ROOT/logs/all-results.txt"
```

还需从 WebUI 或等价 CLI 验证运行中 Job 的日志刷新、停止、最终状态和 artifact
归档；记录 `config snapshot`、stdout/stderr、`events.jsonl`、`result.json`、
可选视频及 SHA-256。

## 最终判定与回填

B7 只有在以下三项全部完成后才能关闭：

- B7.1：三类样板路径均完成，且 G1 使用真实 trained velocity play；
- B7.2：远程 GitHub Actions 在两个远程仓库上通过 contract suite；
- B7.3：Viser 手动验证、Job 日志/取消/状态、artifact hash 和环境版本齐全。

完成后，在本记录补充每条命令、执行时间、退出码、Job/run 路径、结果摘要、artifact
hash、GPU/依赖版本、Viser 截图位置和 CI URL；然后将
`docs/project/DEVELOPMENT_PLAN.md` 中 B7.1、B7.2、B7.3 更新为 `✅ 日期`。

## 历史状态快照（截至 2026-08-20）

| 路径 | 结果 | 证据 |
|---|---|---|
| G1 MotionSkill 安装与兼容性 | 通过 | policy ONNX/deploy SHA-256、`unitree.g1.29dof` Profile 与 `Unitree-G1-Flat` task 契约测试 |
| MJCF Inspector | 通过 | 从 API 安装 PlatformSkill，经 Worker 运行并产生 `report.json`、`report.md`、`robot_profile.draft.yaml` |
| Robot Onboarding AgentSkill | 通过 | 安装、manifest 校验与 `.agents/skills/robot-onboarding/` Codex export 测试 |
| API/WebUI | 通过 | Jobs、logs、cancel、Artifact lineage、静态 WebUI 资源 API 测试 |

最新 CPU 测试：`113 passed, 1 warning`（2026-08-21）。

本地启动烟测：已通过 `/home/lxy/miniconda3/envs/robolab/bin/robolab serve`，自动
分配端口并监听 `http://127.0.0.1:33045`（测试超时后正常关闭）。

## 历史 CI 阻塞说明（截至 2026-08-20）

`.github/workflows/cpu-contract.yml` 会同时 checkout RoboLab 与官方
`RoboLab-Skill` catalog 并运行全部 contract tests。当前工作站的后者含有尚未提交
的 MJCF Inspector 与 Robot Onboarding 包改动；在这些 catalog 改动发布到其远程
仓库前，GitHub CI 无法复现本地的三个样板 Skill 路径。因此 B7.2 保持“进行中”，
尽管本地 CPU suite 已通过。

## 历史 MJLab/Viser 环境快照（截至 2026-08-20）

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

完成所有条目后，将本表与 [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md) 的 B7.1/B7.2/B7.3 状态更新为
`✅`；在此之前 MVP 仅处于 CPU 闭环就绪状态。

## 最新复测（2026-08-21）

- 代码版本：RoboLab `6027dae7a3dea9ac49a02e058e32a08fd529562f`；本地 catalog
  `32d112be710f984f7e048cbfbb9e82fbe559d625`。完整日志、run 目录和 SHA-256 位于
  `/home/lxy/RoboLab/var/b7-acceptance-20260821-114501/`。
- 环境：Python 3.11.15，MJLab 1.2.0，MuJoCo 3.5.0，Torch 2.13.0，Viser 1.1.0，
  warp-lang 1.12.0，SciPy 1.17.1；CUDA 可用，GPU 为 NVIDIA GeForce RTX 4060 Laptop GPU。
- B7.1 CPU：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/contract` 通过，
  `113 passed, 1 warning`。G1 MotionSkill 的 schema、artifact SHA-256、
  `unitree.g1.29dof` 兼容性、安装和 G1 task 发现均通过；MJCF Inspector 与 Robot
  Onboarding 的 manifest 校验、安装以及 Codex export 均通过。
- MJCF Inspector：首次按当时文档中的错误参数 `mjcfPath` 运行失败（`KeyError: 'mjcf_path'`，
  run `3d62c404-818a-4b18-84cf-374154406a10`）。以 schema/实现实际读取的 `mjcf_path` 重试成功，
  run `c7405c22-a638-4ffd-9915-44b1591eaaeb` 产生 `report.json`、`report.md`、
  `robot_profile.draft.yaml`；三个 artifact SHA-256 分别为
  `70c00d2c2f5db35ab347c129bd36c095e895017e8b64160e03b700c14c51715b`、
  `1d6764c0979e31dd5626faff4dd7c20de8a49efc151fe93ac4198f3294c74b5d`、
  `bab54621bd80e31294f7ebfcbf6b406417b42a01ead4d2e7dd5a1131b7e446c6`。验收命令已修正为
  `mjcf_path`。
- B7.3 zero-policy：`Unitree-G1-Flat` 在 1 个环境中完成 CUDA 编译并启动 Viser，日志显示
  `http://localhost:8080`。该 viewer 是持续运行的交互进程；为使自动化复测可返回，180 秒
  后由保护超时终止（run `0decd9e0-1a36-4db1-a0d3-9f0e095665e6`），因此没有最终 result。
- B7.3 trained：修复前首次实际 trained play 失败，run
  `e33baf4d-c773-41e5-a29e-4b25ab1688a5` 退出码 1，原因是 vendor
  当时的 `PlayConfig` 缺少 `wandb_run_path` 属性。带 `--video` 的原命令还会失败（run
  `0f95bf54-9d65-46b5-a8df-2c05c20d34c1`，退出码 2），因为 adapter 将布尔 flag 转为
  `--video`，但当时 adapter 未传递显式布尔值；这些问题已在下方“问题修复复测”中修复。

本轮因此维持 B7.1/B7.2/B7.3 为 `🔶`：CPU 闭环和 Viser 启动证据已更新，但远程 CI、
真实 trained velocity play，以及人工 viewer/Job 取消与归档验收均未完成。

## 问题修复复测（2026-08-21）

- 已将验收命令中的 Inspector 参数统一为 schema 实际要求的 `mjcf_path`。
- `robolab_mjlab_adapter` 已按 vendor tyro CLI 约定将布尔参数序列化为显式值，
  例如 `--video True`；CLI 同时支持 `--checkpoint-file` 和 `--wandb-run-path` 透传。
- vendor `PlayConfig` 已声明此前被读取但未声明的 `wandb_run_path` 和 `registry_name`，
  trained play 现在会给出可理解的“缺少 checkpoint/WandB 路径”错误，而不是
  `AttributeError`。当前仓库没有可供 MJLab RSL-RL 使用的 checkpoint，因此 trained
  play 仍需提供真实 checkpoint 或 WandB run 才能继续。
- 修复后 contract suite：`113 passed, 1 warning`。zero + `--video` 已通过 CLI 参数解析并
  进入 Viser 运行阶段（持续 viewer 进程按预期由测试超时结束）；不再出现“Missing value
  for argument '--video'”。
