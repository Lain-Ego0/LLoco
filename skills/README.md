# Local Skill Workspace

RoboLab 在这里发现本机 Skill：

- `builtin/`：随 RoboLab 源码发布的内置 Skill，纳入 Git；
- `installed/`：从 RoboLab-Skill catalog 安装的固定版本，由平台管理；
- `dev/`：本地开发 checkout 或链接，内容可变。

用户不应手工修改 `installed/` 中的版本。`robolab skill list` 可扫描三类来源；`robolab skill install` 执行固定 revision、校验、内容寻址复制和注册；`robolab skill prepare` 只做权限审查并生成显式环境准备计划，不自动执行第三方脚本。开发中的 Skill 放在 `dev/`，完成后发布到独立的 RoboLab-Skill 仓库。

统一包格式见 [Skill Package 规范](../docs/reference/SKILL_SPEC.md)。
