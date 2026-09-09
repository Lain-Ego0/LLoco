"""One project-local catalog for imported, bundled and generated motions."""

import base64
import uuid
from pathlib import Path

from .services import inside

LIBRARY = Path("src/lloco/assets/motions")
FORMATS = {".bvh", ".pkl", ".npz", ".csv", ".txt"}


def catalog(root: Path):
  return [
    dict(
      path=str(p.relative_to(root)),
      name=str(p.relative_to(root / LIBRARY)),
      format=p.suffix[1:],
      size=p.stat().st_size,
    )
    for p in sorted((root / LIBRARY).rglob("*"))
    if p.is_file()
    and p.suffix.lower() in FORMATS
    and p.resolve().is_relative_to((root / LIBRARY).resolve())
  ]


def import_file(root: Path, body: dict):
  name = body.get("name", "")
  if not name or Path(name).name != name or Path(name).suffix.lower() not in FORMATS:
    raise ValueError("请选择 BVH、PKL、NPZ、CSV 或 TXT 动作文件")
  content = base64.b64decode(body.get("data", ""), validate=True)
  if not content:
    raise ValueError("动作文件为空")
  directory = inside(root, str(LIBRARY / "imports" / uuid.uuid4().hex[:12]))
  directory.mkdir(parents=True)
  path = directory / name
  path.write_bytes(content)
  return dict(path=str(path.relative_to(root)))


def conversion(root: Path, body: dict):
  source = inside(root, body.get("source", ""))
  if not source.is_relative_to((root / LIBRARY).resolve()) or not source.is_file():
    raise ValueError("请选择资产库中的动作")
  action = {".bvh": "gmr-retarget", ".pkl": "gmr-convert", ".csv": "csv-convert"}.get(
    source.suffix.lower()
  )
  if not action:
    raise ValueError("请选择 BVH、PKL 或 CSV 输入")
  robot = body.get("robot", "g1")
  if robot not in ("g1", "g1_23dof") or (action != "csv-convert" and robot != "g1"):
    raise ValueError("BVH / PKL 重定向仅支持 G1 29-DoF")
  output = LIBRARY / "generated" / robot / f"{source.stem}_{uuid.uuid4().hex[:12]}.npz"
  return action, dict(source=str(source), output=str(output), robot=robot)
