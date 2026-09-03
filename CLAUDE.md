# LLoco development guide

- Always run Python tools through `uv run`.
- Keep mjlab as an external, pinned dependency; do not copy its framework source here.
- Put robot models in `src/lloco/assets/robots/<robot>/`.
- Put shared task behavior in mjlab or a single LLoco adapter, not per-robot copies.
- Do not commit checkpoints, exported ONNX policies, or prebuilt third-party runtimes.
- Use two-space indentation and an 88-character line limit for Python.

Before finishing a change, run:

```bash
make check
```

Add user-facing changes to `CHANGELOG.md` under `Unreleased`.
