# 上游来源、许可证与致谢

本文区分当前分发内容、依赖生态和已退役历史来源。只有当前仓库实际包含或发布的第三方内容才进入根目录
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)。历史使用过的项目不因此成为当前依赖。

## 1. 当前直接代码来源

`mjlab/` 是 RoboLab 的 MJLab 1.6 下游定制基座：

- upstream repository：[`mujocolab/mjlab`](https://github.com/mujocolab/mjlab)；
- tag：`v1.6.0`；
- commit：`0fb8a681136be94ffc636a3dd423cabb97d91f10`；
- active location：`mjlab/`；
- license：Apache License 2.0，见 [`mjlab/LICENSE`](../../mjlab/LICENSE)。

固定来源、同步规则和下游修改分别记录在 [`mjlab/UPSTREAM.md`](../../mjlab/UPSTREAM.md) 与
[`mjlab/ROBOLAB_CHANGES.md`](../../mjlab/ROBOLAB_CHANGES.md)。RoboLab 根项目的 MIT 许可证不覆盖或替换 MJLab 及其派生文件的
上游许可证、版权、NOTICE 和修改记录。

## 2. 当前依赖生态

RoboLab/MJLab 使用或解析的主要依赖包括 MuJoCo、MuJoCo-Warp、Warp、RSL-RL、PyTorch、ONNX/ONNX Runtime 等。确切依赖、
版本和传递依赖以 `mjlab/pyproject.toml`、`mjlab/uv.lock` 及各 RoboLab package manifest 为准；本文不单独声明一个可能漂移的
版本清单。

Isaac Lab 等项目对 MJLab 的 manager-based 环境设计有技术影响，但“设计来源”不等于当前仓库直接分发其源码。模型、动作
数据、mesh、图片和文档也必须分别审查许可证，不能自动继承代码许可证。

## 3. 已退役历史来源

R0 前曾使用 `unitreerobotics/unitree_rl_mjlab@1425b15f73bd4095f0df53709d7c389c3eb9e790` 及相关部署来源进行早期
G1 验证。对应 vendor、adapter、Profile、Integration、sim-to-sim 和部署源码已从 active tree 删除，不是当前工具链、依赖或
兼容承诺。历史 revision、许可证和删除事实见 [Unitree 早期路线摘要](../archive/LEGACY_UNITREE_SUMMARY.md) 与 Git 历史。

cnpy、LodePNG、joystick-derived source 等随旧部署树删除的内容同样不属于当前分发物。若未来重新引入任何代码、资产或
机器人 SDK，必须按新的固定 revision 重新完成来源、许可证、NOTICE、修改和分发边界审查，不能沿用历史结论。

## 4. 发布检查

发布前至少确认：

- `THIRD_PARTY_NOTICES.md` 与实际分发树一致；
- 修改过的上游文件保留适用的 copyright、license、NOTICE，并在账本中有显著修改记录；
- 上游同步记录 upstream range、冲突、回归测试和回滚 revision；
- 机器人模型、mesh、动作数据、策略权重、图片和文档分别具有可分发依据；
- Skill manifest 声明自身许可证，未知许可证 Artifact 不作为 MIT 内容再分发；
- 成品机器人 SDK 或协议实现进入 `integrations/` 前完成独立审查。

本文是工程清单，不构成法律意见。

## 5. 商标与官方关系

RoboLab 是独立项目，不是 MJLab、MuJoCo、Unitree Robotics 或其他上游项目的官方发行版。名称和商标只用于描述来源或
兼容性，不暗示认可、认证或合作关系。
