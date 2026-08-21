# RoboLab 本地环境安装

状态：2026-08-21 修订。默认环境服务于 RoboLab 定制 MJLab 1.6 主线；Unitree/MJLab 1.2 仅作为隔离的 legacy 环境。

## 1. 平台控制面

在 RoboLab 根目录执行：

```bash
conda activate robolab
python -m pip install -e packages/schemas -e packages/core -e packages/mjlab_adapter \
  -e services/api -e services/worker -e apps/cli
```

首次创建环境时还需要：

```bash
python -m pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" "httpx>=0.27"
```

## 2. 默认 MJLab 1.6 主线环境

R0 完成后，默认环境必须安装仓库内固定 revision 的 `vendor/mjlab/` 及其依赖：

```bash
python -m pip install -e vendor/mjlab
```

MJLab 1.6 的确切 Torch、MuJoCo-Warp、Warp、RSL-RL 和 Python 版本以 `vendor/mjlab/pyproject.toml`、环境锁文件和
`vendor/mjlab/UPSTREAM.md` 为准。不要仅凭 README 中的版本号自行升级或降级；每次依赖变更必须有 smoke 和回归记录。

默认环境必须能够运行 Customized MJLab 的 import、registry、最小模型加载、CPU smoke test 和平台 Job。默认安装不应
把 `vendor/unitree_rl_mjlab` 作为所有后端的隐式依赖。

## 3. Unitree legacy 环境

只有需要复现旧 G1 play、旧 checkpoint 或 Unitree sim-to-sim 时，才安装 legacy 依赖：

```bash
python -m pip install -e "packages/mjlab_adapter[mjlab-runtime]" -e vendor/unitree_rl_mjlab
```

这条命令属于 `unitree_legacy_mjlab_1_2` 环境路径，不能作为 RoboLab 1.6 默认环境的安装说明。当前 legacy 适配器固定：

- `mjlab==1.2.0`；
- `mujoco-warp==3.5.0`；
- `warp-lang==1.12.0`；
- 该环境特有的 Torch、Viser 和其他依赖。

旧 vendor 代码仍调用 `warp.context`，因此不要让 pip 在 legacy 环境中自动升级到不兼容的 Warp 版本。legacy 环境应与
MJLab 1.6 环境分开创建或至少使用独立 lock 文件，禁止混装两个版本的 MJLab/Warp/MuJoCo-Warp。

## 4. 启动本地服务

```bash
cd /home/lxy/RoboLab
conda activate robolab
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

测试应明确标记 `cpu`、`mjlab_1_6`、`unitree_legacy`、`gpu` 和 `hardware`。Unitree legacy 测试通过不代表 MJLab 1.6
定制工具链通过；反之亦然。

## 6. 环境记录要求

任何训练、评测或部署 Artifact 必须记录：

- Python/Conda environment identifier；
- MJLab upstream revision；
- RoboLab revision；
- Torch、MuJoCo、MuJoCo-Warp、Warp、RSL-RL 和 ONNX Runtime 版本；
- GPU/CUDA 信息（如适用）；
- 安装命令或 lock 文件 hash。

如果环境属于 Unitree legacy，Artifact 中必须写入：

```text
toolchain: unitree_legacy_mjlab_1_2
product_default: false
```
