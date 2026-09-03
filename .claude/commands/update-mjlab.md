---
allowed-tools: Bash(uv lock:*), Bash(uv run:*), Edit, Read
description: Update LLoco's pinned mjlab dependency
---

Update the exact `mjlab` version in `pyproject.toml` to `$ARGUMENTS`.

1. Require a semantic version; ask for one when `$ARGUMENTS` is empty.
2. Change only the `mjlab==...` dependency.
3. Run `uv lock --upgrade-package mjlab`.
4. Run `make check` and report any 1.6 API compatibility breaks instead of hiding them.
5. Add the dependency update to `CHANGELOG.md`.

Do not pin or update mjlab's transitive MuJoCo Warp dependency directly from LLoco.
