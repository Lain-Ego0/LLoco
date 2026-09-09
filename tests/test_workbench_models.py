"""Native parsing and kinematics for the small LLoco model inspector."""

import base64
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
from lloco.workbench.models import Models

URDF = b"""<robot name="inspection">
<link name="base"><visual><geometry><box size=".3 .2 .1"/></geometry></visual></link>
<link name="arm"><inertial><mass value="1"/>
<inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/></inertial>
<visual><geometry><box size=".1 .1 .4"/></geometry></visual></link>
<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
<axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
</robot>"""


def uploaded(name, data):
  return dict(name=name, data=base64.b64encode(data).decode())


def test_urdf_import_and_joint_reset(tmp_path):
  models = Models()
  initial = models.load(
    tmp_path,
    dict(selected="robot/test.urdf", files=[uploaded("robot/test.urdf", URDF)]),
  )
  assert len(initial["joints"]) == 1
  assert len(initial["bodies"]) == 2
  assert len(initial["geoms"]) == 2
  joint = initial["joints"][0]["id"]
  updated = models.pose(dict(id=initial["id"], joint=joint, value=5))
  assert updated["axes"][0]["value"] == 1
  assert updated["bodies"][1]["rotation"] != initial["bodies"][1]["rotation"]
  reset = models.pose(dict(id=initial["id"], reset=True))
  assert reset["axes"][0]["value"] == 0
  with pytest.raises(ValueError):
    models.pose(dict(id=initial["id"], joint=joint, value=float("nan")))


def test_mjcf_include_and_inertia(tmp_path):
  main = b'<mujoco><include file="parts/body.xml"/></mujoco>'
  part = b'<mujoco><worldbody><body name="sphere"><geom type="sphere" size=".2" mass="2"/></body></worldbody></mujoco>'
  model = Models().load(
    tmp_path,
    dict(
      selected="robot/scene.xml",
      files=[uploaded("robot/scene.xml", main), uploaded("robot/parts/body.xml", part)],
    ),
  )
  assert model["bodies"][0]["mass"] == pytest.approx(2)
  assert model["bodies"][0]["radii"] == pytest.approx([0.2, 0.2, 0.2])


def test_builtin_go2_meshes():
  root = Path(__file__).resolve().parents[1]
  model = Models().load(
    root, dict(path="src/lloco/assets/robots/unitree_go2/xmls/scene_go2.xml")
  )
  assert len(model["joints"]) == 12
  assert model["meshes"]
  assert any(g["collision"] for g in model["geoms"])


def test_invalid_import(tmp_path):
  with pytest.raises(ValueError):
    Models().load(
      tmp_path,
      dict(selected="../robot.xml", files=[uploaded("../robot.xml", b"<mujoco/>")]),
    )
  with pytest.raises(ValueError, match="XML"):
    Models().load(
      tmp_path, dict(selected="bad.xml", files=[uploaded("bad.xml", b"<mujoco>")])
    )
