"""Asset catalog, safe import and automatic conversion destinations."""

import base64

import pytest

from lloco.workbench.motion_library import LIBRARY, catalog, conversion, import_file
from lloco.workbench.services import command


def test_import_select_and_generate(tmp_path):
  body = dict(name="dance.bvh", data=base64.b64encode(b"MOTION").decode())
  first = import_file(tmp_path, body)
  second = import_file(tmp_path, body)
  assert first["path"] != second["path"]
  assert len(catalog(tmp_path)) == 2
  action, values = conversion(tmp_path, dict(source=first["path"]))
  assert action == "gmr-retarget"
  assert values["output"].startswith(str(LIBRARY / "generated/g1"))
  assert values["output"].endswith(".npz")
  assert (
    values["output"] != conversion(tmp_path, dict(source=first["path"]))[1]["output"]
  )
  assert "--output" in command(tmp_path, action, values)
  with pytest.raises(ValueError):
    conversion(tmp_path, dict(source=first["path"], robot="g1_23dof"))


def test_only_library_sources(tmp_path):
  (tmp_path / "external.bvh").write_text("MOTION")
  with pytest.raises(ValueError):
    conversion(tmp_path, dict(source="external.bvh"))
  with pytest.raises(ValueError):
    import_file(tmp_path, dict(name="../bad.npz", data="AA=="))


def test_csv_target(tmp_path):
  item = import_file(tmp_path, dict(name="dance.csv", data="MSwyLDM="))
  action, values = conversion(tmp_path, dict(source=item["path"], robot="g1_23dof"))
  assert action == "csv-convert"
  args = command(tmp_path, action, values)
  assert args[args.index("--robot") + 1] == "g1_23dof"
