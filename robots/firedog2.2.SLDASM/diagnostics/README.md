# FireDog 2.2 diagnostics

生成入口：

```bash
PYTHONPATH=apps/cli/src:packages/core/src:packages/schemas/src \
uv run --with PyYAML --with jsonschema python -m robolab_cli.main robot inspect \
  model/firedog2_2.xml --json
```

Inspector 规则 ID 稳定以 `INSPECT_*` 开头，错误包含对象名称和 XML 路径。合法模型是 `model/firedog2_2.xml`；非法反例位于
`tests/contract/test_r2_robot_onboarding.py` 的临时 fixture，覆盖重复名称、缺失 mesh、缺失引用和非法 root body。

