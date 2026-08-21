"""Non-invasive local startup health probes."""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path


def health(data_dir: str | Path) -> dict[str, object]:
    root = Path(data_dir)
    disk = shutil.disk_usage(root)
    gpu = {"available": False, "detail": "未检测到 CUDA"}
    try:
        import torch
        gpu = {"available": bool(torch.cuda.is_available()), "detail": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "PyTorch 未检测到 CUDA"}
    except ImportError:
        gpu["detail"] = "未安装 PyTorch"
    conda_prefix = os.environ.get("CONDA_PREFIX") or (sys.prefix if (Path(sys.prefix) / "conda-meta").is_dir() else None)
    mjlab_import = {"ok": False, "importOk": False, "status": "unavailable", "detail": "MJLab 1.6 未导入"}
    try:
        import mjlab

        mjlab_import.update({"ok": True, "importOk": True, "status": "ready", "version": getattr(mjlab, "__version__", "unknown")})
    except ImportError as exc:
        mjlab_import["detail"] = f"无法导入 mjlab: {exc}"
    return {"python": {"ok": True, "version": sys.version.split()[0], "executable": sys.executable}, "conda": {"ok": conda_prefix is not None, "environment": os.environ.get("CONDA_DEFAULT_ENV") or (Path(conda_prefix).name if conda_prefix else None), "prefix": conda_prefix}, "mujoco": {"ok": importlib.util.find_spec("mujoco") is not None}, "mjlab": mjlab_import, "gpu": gpu, "disk": {"ok": disk.free > 1024 ** 3, "freeBytes": disk.free, "totalBytes": disk.total}}
