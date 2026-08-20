# Local Skill Workspace

RoboLab 在这里发现本机 Skill：

- `builtin/`：随 RoboLab 源码发布的内置 Skill，纳入 Git；
- `installed/`：从 RoboLab-Skill catalog 安装的固定版本，由平台管理；
- `dev/`：本地开发 checkout 或链接，内容可变。

用户不应手工修改 `installed/` 中的版本。下载、哈希校验、Conda 环境准备、contract test 和注册应由未来的 Skill Manager 完成。开发中的 Skill 放在 `dev/`，完成后发布到独立的 RoboLab-Skill 仓库。

统一包格式见 [Skill Package 规范](../docs/SKILL_SPEC.md)。当前目录只是仓库边界和安装位置，Skill Manager 尚未实现。

