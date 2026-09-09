# LLoco 独立工作台

工作台位于 `src/lloco/workbench/`。主训练包 `lloco.tasks` 和原有 train/play CLI 保持独立；HTTP 服务不导入 torch、mjlab 或训练环境，耗时工作全部在子进程中运行。

## 启动

```bash
cd /path/to/LLoco
uv run --extra workbench lloco-workbench
# 或指定项目、端口，不自动打开浏览器
uv run --extra workbench lloco-workbench --project /path/to/LLoco --port 7861 --no-browser
```

`webui`、`lloco-webui` 和 `python -m lloco.webui` 指向同一新工作台。
页面状态保存在浏览器 localStorage；任务信息和日志位于被 git 忽略的 `.lloco-workbench/`。
训练写入 `logs/workbench/`，与原有训练输出分开。运行任务可查看日志、停止；退出服务时停止其子进程。重启保留历史状态，不自动恢复训练。

## 五步流程

1. 精简查看器由 LLoco 自行编写，使用 MuJoCo 原生编译 URDF/MJCF，通过 Three.js 显示模型。支持项目路径、目录或多文件导入，STL/OBJ 网格和基本几何体；提供视角适配、关节调节、重置、坐标系、关节轴、碰撞线框、惯量椭球和连杆质量/主惯量数值。URDF 保留固定连杆和视觉几何。输入必须可由 MuJoCo 编译；当前不显示纹理、高度场，不包含 USD、编辑器、动画、测量或仿真。项目路径支持模型同目录及内置 `xmls/`、`urdf/` 的父目录资源。模型只在内存中缓存，不创建训练配置。
2. Velocity 和 Tracking 二选一，未选择分支折叠。Tracking 的 GMR 支持 LAFAN1 骨架 BVH：读取原始 Frame Time，项目内 GMR 求解 IK，校验 G1 关节顺序，转换为 50 Hz NPZ。也可输入已有 GMR PKL，或直接选择已有 NPZ。输出路径必须尚不存在。当前重定向限定 G1 29-DoF，23-DoF 使用独立转换的 NPZ；不宣称任意 BVH、SMPL-X 或视频已支持。
3. Velocity / Tracking 中选择真实注册任务的 PPO 配置；Go2 Skill 作为 Velocity 下的独立四足入口，提供 DreamWaQ、AMP-DreamWaQ、CTS、TS Teacher 和五种动作模板。算法不会跨任务任意拼接。
4. 输入正整数的迭代轮次、保存间隔和环境数。工作台强制使用 TensorBoard logger，保留各任务其他默认参数。GPU 选择遵循既有训练 CLI。TensorBoard 监听本机 6006，在训练页面内显示现有 logs 下的奖励、损失及性能曲线。
5. 扫描 logs 下的 PT/ONNX，提供下载、checkpoint 选择和导出。ONNX 位于 checkpoint 所在目录的 `exported/<checkpoint名>/policy.onnx`。导出在 CPU 上创建单环境并使用任务实际 runner（Tracking 会使用自己的导出器）。CPU Viser 监听 8080，通过 checkpoint 策略执行回放，Tracking 需要匹配的 motion；当前不包含 ONNX Runtime 回放。端口占用时显示错误，避免错误嵌入已有服务。

服务默认供本机浏览器使用；TensorBoard 绑定 loopback，远程访问需自行配置隧道。各任务同一类操作只允许同时启动一个；进程退出码和启动失败原因显示在任务日志中。

## 模块

| 路径 | 职责 |
| --- | --- |
| `server.py` | HTTP 路由、文件资源和静态页面服务 |
| `catalog.py` | 无训练依赖的任务 / 算法目录 |
| `services.py` | 参数校验、命令构建、进程组管理、历史日志 |
| `worker.py` | 调用原有训练、回放和任务 runner 导出 |
| `retarget.py`、`gmr/` | 项目内 GMR BVH 重定向及带许可证的上游子集 |
| `static/` | 五步工作台页面和查看器构建产物 |
| `viewer/` | LLoco 自编精简 Three.js 查看器 |
| `models.py` | MuJoCo 模型编译、几何提取和正向运动学；不导入训练环境 |

GMR 第三方归属见 `gmr/NOTICE.md`、`gmr/LICENSE` 及 LAFAN loader 许可证。查看器未沿用 robot_viewer 应用源码；Three.js 的 MIT 许可证见 `viewer/public/THREE_LICENSE.txt`。

## 查看器开发与验证

已包含构建产物，首次启动无需 Node.js。修改查看器后重建：

```bash
cd src/lloco/workbench/viewer
npm ci
npm run build
```

构建输出到 `../static/viewer/`，由同一个工作台 HTTP 服务提供；不启动外部 viewer 服务。工作台的 `static/app.js`、`style.css` 可直接编辑。

```bash
# 避免本机 ROS 的 pytest 自动插件混入 Python 3.13 环境
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/test_workbench.py tests/test_workbench_models.py -q
```

测试覆盖命令参数、任务分支校验、路径约束、子进程日志与历史、静态资源和同源请求边界。页面浏览器测试与 GMR/训练/导出 smoke 是额外的实际运行验证；不代表完整动作质量或策略收敛验收。

模型接口测试覆盖 URDF 导入、MJCF include、Go2 网格、关节限位与重置、惯量椭球和无效输入。模型解析时才加载 MuJoCo，不涉及训练依赖。
