# LLoco Model Inspector

LLoco 的精简模型检查页面，由本项目编写。参考 robot_viewer 的检查项划分思路，未沿用其应用源码或 UI。

- `../models.py`：MuJoCo 原生编译、几何提取、正向运动学和惯量计算。
- `src/main.js`：Three.js 场景、检查图层、关节与导入交互。
- `src/style.css` / `index.html`：工作台风格的精简布局。

只实现 URDF/MJCF 导入、基本几何与 STL/OBJ 网格、轨道视角、关节调节、坐标系、关节轴、碰撞线框和惯量椭球。没有代码编辑、动画编辑、测量、USD 或物理仿真功能。Tracking 的动作预览由 `src/motion.js` 提供：源骨架与 G1 同步显示，支持实时进度和结果时间轴回放。输入须可由 MuJoCo 编译；不显示纹理或高度场。

`npm ci && npm run build` 输出到 `../static/viewer/`。运行依赖 LLoco 工作台的模型 API，而非额外查看器进程。

Three.js 使用 MIT 许可证，见 `public/THREE_LICENSE.txt`。LLoco 自编代码遵循项目根许可证。
