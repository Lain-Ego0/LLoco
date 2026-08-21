# RoboLab 本地环境安装

状态：2026-08-21 修订。默认环境服务于 RoboLab 定制 MJLab 1.6 主线；Unitree/MJLab 1.2 不再是可安装的当前路线。

## 1. 平台控制面

在 RoboLab 根目录执行：

```bash
conda activate robolab-mjlab16
python -m pip install -e mjlab -e packages/schemas -e packages/core \
  -e services/api -e services/worker -e apps/cli
```

首次创建环境时还需要：

```bash
python -m pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" "httpx>=0.27"
```

## 2. 默认 MJLab 1.6 主线环境

默认环境名为 `robolab-mjlab16`。首次创建并安装仓库内固定 revision：

```bash
conda create -n robolab-mjlab16 python=3.11
conda activate robolab-mjlab16
python -m pip install -e mjlab
```

MJLab 1.6 的确切依赖以 `mjlab/pyproject.toml`、`mjlab/uv.lock` 和 `mjlab/UPSTREAM.md` 为准。当前锁文件关键版本为
Torch 2.9.0、Warp 1.14.0、MuJoCo-Warp 3.11.0、MuJoCo 3.11.0 和 RSL-RL 5.4.2；不得与旧 MJLab 1.2/Warp 1.12 混装。

在需要按 lock file 重建现有 Python 3.11 环境时，使用 `uv` 导出并同步依赖，再重新安装 editable 项目：

```bash
uv export --locked --python "$CONDA_PREFIX/bin/python" --no-dev --no-emit-project \
  --format requirements.txt -o /tmp/robolab-mjlab16-py311.txt
uv pip sync --python "$CONDA_PREFIX/bin/python" /tmp/robolab-mjlab16-py311.txt
python -m pip install -e mjlab
```

默认环境必须能够运行 Customized MJLab 的 import、registry、最小模型加载和 CPU smoke test。R1 的 RoboLab train/play/
evaluate/export 入口尚未实现，当前 CLI 不提供伪造的 MJLab 子命令。

## 3. 历史 Unitree 环境说明

Unitree/MJLab 1.2 环境、adapter 和 vendor 已从 active tree 删除。历史 Artifact 如需复查，必须从 Git 历史和历史记录恢复
原始环境信息；本文件不提供可执行的 legacy 安装命令。

## 4. 启动本地服务

```bash
cd /home/lxy/RoboLab
conda activate robolab-mjlab16
robolab serve
```

服务只监听 `127.0.0.1`。WebUI 的生产构建已随仓库保存，普通启动不需要 Node.js；仅修改前端时才在 `apps/web/` 执行：

```bash
npm install
npm run build
```

## 5. 测试

若 shell 已加载 ROS 等系统 Python 路径，运行测试前禁用外部 pytest 插件：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/contract
```

contract 测试在平台 CPU 环境运行；MJLab 1.6 smoke 在 `robolab-mjlab16` 独立环境运行。CPU 与 GPU 检查必须分开记录。

## 6. R0.7 smoke evidence

2026-08-21 在 `robolab-mjlab16` 中完成以下 CPU 可执行检查，均退出码 0：

```bash
python -m pip install -e mjlab
python -c 'import mjlab; print(mjlab.__file__)'
python -m mjlab.scripts.list_envs
python mjlab/tests/smoke_test.py
```

editable 路径为 `/home/lxy/RoboLab/mjlab/src/mjlab/__init__.py`；`list_envs` 注册并列出 6 个不依赖 Unitree 的上游任务，
cartpole 最小模型加载 smoke 通过。环境版本为 Python 3.11.15、Torch 2.9.0（CUDA 12.8）、Warp 1.14.0、
MuJoCo-Warp 3.11.0、MuJoCo 3.11.0 和 RSL-RL 5.4.2。

GPU 检查仅做只读记录：GPU 可见但当前由外部 IsaacLab 进程占用（约 4765 MiB）；本批次未启动 RoboLab 训练、viewer 或
随机策略验收。GPU 训练/性能验收等待外部资源。

## 7. 环境记录要求

任何训练、评测或部署 Artifact 必须记录：

- Python/Conda environment identifier；
- MJLab upstream revision；
- RoboLab revision；
- Torch、MuJoCo、MuJoCo-Warp、Warp、RSL-RL 和 ONNX Runtime 版本；
- GPU/CUDA 信息（如适用）；
- 安装命令或 lock 文件 hash。

Unitree legacy 不属于当前默认安装路线；历史结果必须明确标记为已退役工具链。
