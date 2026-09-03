# Integration tests

This tier exercises component boundaries on CPU: ONNX Runtime execution,
state/history reset behavior, checkpoint reload/export, and finite simulator
control loops. Tests must remain bounded and may not claim policy convergence.

Run it with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --package mjlab --extra cpu \
  pytest -q tests/integration
```
