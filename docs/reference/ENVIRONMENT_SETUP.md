# RoboLab 本地环境安装

在 RoboLab 根目录执行。以下命令将仓库内的所有 Python 包以 editable
方式安装到已有的 `robolab` Conda 环境，之后无需设置 `PYTHONPATH`：

```bash
conda activate robolab
python -m pip install -e packages/schemas -e packages/core -e packages/mjlab_adapter \
  -e services/api -e services/worker -e apps/cli
```

首次创建环境时还需要 API 依赖：

```bash
python -m pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" "httpx>=0.27"
```

如需在同一个 `robolab` 环境运行 MJLab/Viser（而不仅是使用 WebUI/API），安装
经验证的 runtime extra 与 vendor checkout：

```bash
python -m pip install -e "packages/mjlab_adapter[mjlab-runtime]" -e vendor/unitree_rl_mjlab
```

`mjlab==1.2.0` 的 vendor 代码仍调用 `warp.context`，因此固定
`warp-lang==1.12.0`。不要让 pip 自动升级到 `warp-lang==1.16.0`：该版本会在
G1 play 初始化前报 `AttributeError: module 'warp' has no attribute 'context'`。
`scipy` 也是 MJLab task registry 的隐式运行依赖，已纳入此 extra。

启动本地服务：

```bash
cd /home/lxy/RoboLab
conda activate robolab
robolab serve
```

服务只监听 `127.0.0.1`。WebUI 的生产构建已随仓库保存，普通启动不需要
Node.js；仅修改前端时才在 `apps/web/` 执行 `npm install && npm run build`。

若 shell 已加载 ROS 等系统 Python 路径，运行测试前禁用外部 pytest 插件以避免环境
污染：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/contract
```
