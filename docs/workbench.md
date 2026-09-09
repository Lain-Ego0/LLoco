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

1. 精简查看器由 LLoco 自行编写，使用 MuJoCo 原生编译 URDF/MJCF，通过 Three.js 显示模型。支持项目路径、目录或多文件导入，STL/OBJ 网格和基本几何体；提供视角适配、关节调节、重置、坐标系、关节轴、碰撞线框、惯量椭球和连杆质量/主惯量数值。URDF 保留固定连杆和视觉几何。输入必须可由 MuJoCo 编译；当前不显示纹理、高度场，不包含 USD、代码或动画编辑器、测量或仿真。项目路径支持模型同目录及内置 `xmls/`、`urdf/` 的父目录资源。模型只在内存中缓存，不创建训练配置。
2. Velocity 和 Tracking 二选一，未选择分支折叠。Tracking 的 GMR 支持 LAFAN1 骨架 BVH：读取原始 Frame Time，项目内 GMR 求解 IK，校验 G1 关节顺序，转换为 50 Hz NPZ。也可输入已有 GMR PKL，或直接选择已有 NPZ。输出由工作台生成唯一文件名，避免覆盖已有动作。当前重定向限定 G1 29-DoF，23-DoF 使用独立转换的 NPZ；不宣称任意 BVH、SMPL-X 或视频已支持。
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

## 重定向过程与效果预览

Tracking 页面从统一资产库选择 BVH / PKL 输入，点击开始后自动打开预览，NPZ 输出由工作台自动命名并入库。任务会依次显示读取动作、求解重定向、生成 NPZ 和完成状态；帧数来自实际处理进度。重定向尚未结束时，勾选“跟随进度”可查看最新已求解姿态。

- BVH：左侧橙色为原始人体骨架，右侧为同一时刻的 G1 模型。
- PKL：只显示 G1；已有 GMR PKL 不包含原始人体骨架。
- 支持播放 / 暂停、时间轴定位、0.25× / 0.5× / 1× / 2× 速度。“固定根位置”便于原地对比，关闭后显示根位置移动。
- 预览保留 GMR 求解后的 qpos，以约 15 Hz 抽样传输，按源动作时间回放；训练 NPZ 仍按原转换流程生成 50 Hz 数据。预览用于检查运动姿态，不是训练策略回放，也不是重定向误差评测。
- 每个任务的进度和预览位于 `.lloco-workbench/previews/<任务ID>/`。任务列表的“动作预览”可重新打开结果；页面的“查看上次预览”记住最近选择。停止或失败后可查看已生成部分，界面明确显示状态。旧版本任务若没有保存预览，需要重新运行。

前端采用紧凑的浅色工具界面，主工作台与模型/动作查看器使用一致的表单和边框样式。

新增 `motion_preview.py` 负责原子进度更新和增量姿态读取，`viewer/src/motion.js` 负责骨架对照、时间轴与回放。预览使用工作台模型接口进行 CPU 正向运动学，不启动训练环境。

## 统一动作资产库

工作台只从 `src/lloco/assets/motions/` 枚举动作。重定向输入和训练 NPZ 均使用下拉选择，不再填写路径；“浏览资产库”可按文件直接选择用途，“导入动作文件”将本机 BVH / PKL / CSV / NPZ / TXT 复制入库。

| 子目录 | 内容 |
| --- | --- |
| `g1/`、`g1_23dof/`、`go2_amp/` | 原有项目动作 |
| `converted/` | 已有训练 NPZ 的归档副本 |
| `examples/` | 原工作台验证用 BVH / PKL / NPZ 片段，非完整动作数据集 |
| `imports/<导入ID>/` | 用户导入文件；同名文件不会覆盖 |
| `generated/g1/`、`generated/g1_23dof/` | 自动命名的转换结果 |

历史文件已归档到资产库；旧位置保留副本以兼容已有命令和日志引用，界面只展示资产库内文件。新转换统一写入 `generated/`，成功后自动选为当前训练动作；失败或停止不会替换已选动作。

BVH / GMR PKL 支持 G1 29-DoF；CSV 可选择 G1 29 或 23-DoF，并按 30 Hz 输入转换为 50 Hz NPZ。NPZ 可直接选用于 Tracking。Go2 TXT 显示为算法内置动作，不进入 GMR 转换。

CSV 转换在按钮下方显示实际进度：初始化、逐帧转换（已处理/总输出帧数及百分比）、保存 NPZ、完成。转换器每处理 10 帧以及末帧更新状态；完成以 NPZ 写入成功为准，失败或停止会显示相应状态。最近一次 CSV 进度可在刷新页面后恢复查看。

## 查看 CSV 转换后的效果

CSV 完成后点击进度条旁的“播放转换结果”；也可从“训练动作 · NPZ”选择资产，再点击“播放所选动作”。打开后点击播放器的“播放”，或拖动时间轴逐帧检查，支持倍速和固定根位置。此入口直接读取实际保存的 LLoco G1 29/23-DoF NPZ（根姿态与原生顺序的关节角），无需重定向或训练。
