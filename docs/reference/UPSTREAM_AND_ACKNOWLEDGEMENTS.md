# 上游来源、许可证与致谢

## 1. 直接代码来源

`vendor/mjlab/` 是 RoboLab 的 MJLab 1.6 深度定制基座。它必须记录固定的 upstream repository、commit、导入日期、
RoboLab 修改和回归证据。MJLab 源码及其派生修改继续遵守上游许可证；RoboLab 不会把下游修改重新标记为根项目 MIT。

`vendor/unitree_rl_mjlab/` 精选导入自 Unitree Robotics 的 [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)，是本项目首批 Unitree 训练、回放、机器人资产与部署迁移参考，已纳入 RoboLab Git。该上游使用 Apache License 2.0。来源基线已经通过完整 blob 哈希比对固定为 [`1425b15f`](https://github.com/unitreerobotics/unitree_rl_mjlab/commit/1425b15f73bd4095f0df53709d7c389c3eb9e790)。导入采用精选 vendor，未携带上游宣传媒体、冗余预编译 runtime 和归属 Skill 的产物；include/exclude 规则见 `vendor/unitree_rl_mjlab/VENDOR_MANIFEST.yaml`。

以后同步上游时应使用单独 commit，记录 upstream range、冲突和 RoboLab 修改，避免无法区分原创、修改与上游更新。

## 2. 应重点致谢的项目

### 直接基础与机器人生态

- [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)：RoboLab 内 `vendor/unitree_rl_mjlab/` 的来源、历史 G1 样板和 Unitree legacy 兼容基础；不作为长期公共平台 API。
- [mujocolab/mjlab](https://github.com/mujocolab/mjlab)：上游训练环境与高层 API 基础。
- [unitreerobotics/unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)：Unitree 实机通信与控制接口。
- [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)：现有 sim-to-sim 流程的重要组成。

### 仿真、学习与动作跟踪

- [MuJoCo](https://github.com/google-deepmind/mujoco)：物理仿真基础。
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)：GPU 加速仿真后端。
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)：上游 API 和 manager-based 环境设计的重要来源。
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)：强化学习算法与 runner 生态。
- [whole_body_tracking / BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)：动作预处理与全身动作跟踪方法来源。

### 通信与运行依赖

- [Eclipse Cyclone DDS](https://github.com/eclipse-cyclonedds/cyclonedds)：DDS 通信实现。
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)：现有 C++ 部署程序的推理依赖；curated vendor 不含预编译包，由安装器提供固定版本（deploy/simulate 的 CMake 已改为发现外部安装）。
- [cnpy](https://github.com/rogersce/cnpy)：现有部署程序读取 NumPy `.npy/.npz` 数据。
- [LodePNG](https://lodev.org/lodepng/)：现有 MuJoCo 模拟器包含的 PNG 编解码实现。
- [drewnoakes/joystick](https://github.com/drewnoakes/joystick)：现有 joystick 工具注明的代码基础。

## 3. 致谢与法律声明要分开

README 的“致谢”用于表达技术来源；`THIRD_PARTY_NOTICES.md`、各依赖许可证和发布包中的 NOTICE 用于满足分发义务。两者不能互相替代。

发布前至少完成：

- 保留 `vendor/unitree_rl_mjlab/LICENCE`；
- 对修改过的 Apache-2.0 来源文件添加显著修改说明或用可追踪变更记录满足要求；
- 保留上游已有 copyright、patent、trademark 和 attribution notice；
- 如果上游/依赖包含 NOTICE，将其可读副本带入发布物；
- 为模型、动作数据、mesh、图片和文档分别核查许可，它们不一定自动继承代码许可证；
- Skill manifest 声明自身许可证，平台不得把未知许可的 artifact 当作 MIT 内容再分发。

本文是工程清单，不构成法律意见。

## 4. 商标与官方关系

RoboLab 是独立项目，不是 Unitree Robotics、MuJoCo、MJLab 或其他上游项目的官方发行版。名称和商标只应用于描述兼容性与来源，不暗示认可、认证或合作关系。
