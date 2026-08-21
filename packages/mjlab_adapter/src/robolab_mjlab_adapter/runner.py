"""Command construction for the vendor scripts; execution belongs to Worker."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping


def build_play_command(vendor_root: str | Path, task_id: str, parameters: Mapping[str, object] | None = None, *, python: str | None = None) -> list[str]:
    """Build an equivalent ``play.py`` command without executing it."""
    if not task_id or any(char in task_id for char in "\r\n\x00"):
        raise ValueError("task_id 不能为空且不能包含控制字符")
    script = Path(vendor_root).resolve() / "scripts" / "play.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    # play.py selects the task as its first positional argument.
    command = [python or sys.executable, str(script), task_id]
    for key, value in (parameters or {}).items():
        if not key or key.startswith("_"):
            raise ValueError(f"非法 play 参数: {key!r}")
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                # The vendored tyro CLI represents booleans as explicit
                # values (e.g. ``--video True``), rather than store_true
                # flags.  Emitting the value keeps the adapter compatible
                # with that contract and avoids an argparse-style mismatch.
                command.extend([flag, "True"])
        elif value is not None:
            command.extend([flag, str(value)])
    return command
