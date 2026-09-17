"""Convert the OpenDoge URDF into the checked-in MuJoCo XML.

Run from the repository root:

  python -m lloco.assets.robots.opendoge.convert_urdf

The URDF is still the source of truth for inertia, joint axes/limits and
collision primitives.  This script applies the mjlab-specific post-processing:

* move the URDF root link into a floating-base ``base_link`` body
* add visual mesh geoms and the ``imu`` site
* add LainLab foot sites and stable collision-geom names
* add the IMU/root-angmom sensors used by the velocity task observations
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

_HERE = Path(__file__).resolve().parent
_DEFAULT_URDF = _HERE / "urdf" / "opendoge.urdf"
_DEFAULT_OUTPUT = _HERE / "xmls" / "opendoge.xml"

_LINK_MESHES = {
  "base_link": "base_link.STL",
  "FL_hip": "FL_hip.STL",
  "FL_thigh": "FL_thigh.STL",
  "FL_calf": "FL_calf.STL",
  "FR_hip": "FR_hip.STL",
  "FR_thigh": "FR_thigh.STL",
  "FR_calf": "FR_calf.STL",
  "RL_hip": "RL_hip.STL",
  "RL_thigh": "RL_thigh.STL",
  "RL_calf": "RL_calf.STL",
  "RR_hip": "RR_hip.STL",
  "RR_thigh": "RR_thigh.STL",
  "RR_calf": "RR_calf.STL",
}


def _prefix(link_name: str) -> str:
  return link_name.split("_", 1)[0]


def _build_asset() -> ET.Element:
  asset = ET.Element("asset")
  for link_name, file_name in _LINK_MESHES.items():
    ET.SubElement(asset, "mesh", {"name": link_name, "file": file_name})
  return asset


def _build_defaults() -> ET.Element:
  default = ET.Element("default")
  root = ET.SubElement(default, "default", {"class": "opendoge"})
  visual = ET.SubElement(root, "default", {"class": "visual"})
  ET.SubElement(
    visual,
    "geom",
    {
      "type": "mesh",
      "contype": "0",
      "conaffinity": "0",
      "density": "0",
      "group": "2",
    },
  )
  collision = ET.SubElement(root, "default", {"class": "collision"})
  ET.SubElement(collision, "geom", {"priority": "1", "condim": "6", "group": "3"})
  ET.SubElement(root, "site", {"rgba": "1 0 0 1", "group": "5"})
  return default


def _name_collisions(body: ET.Element) -> None:
  body_name = body.get("name", "")
  collision_idx = 0
  for geom in [child for child in list(body) if child.tag == "geom"]:
    if geom.get("class") == "visual":
      continue
    geom_type = geom.get("type", "sphere")
    if body_name == "base_link":
      geom.set("name", "base_collision")
    elif body_name.endswith("_hip"):
      geom.set("name", f"{_prefix(body_name)}_hip_collision")
    elif body_name.endswith("_thigh"):
      suffix = "" if collision_idx == 0 else "2"
      geom.set("name", f"{_prefix(body_name)}_thigh_collision{suffix}")
    elif body_name.endswith("_calf"):
      if geom_type == "sphere":
        geom.set("name", f"{_prefix(body_name)}_foot_collision")
        geom.set("type", "sphere")
      else:
        geom.set("name", f"{_prefix(body_name)}_calf_collision")
    geom.set("class", "collision")
    collision_idx += 1
  for child in list(body):
    if child.tag == "body":
      _name_collisions(child)


def _insert_visuals_and_sites(body: ET.Element) -> None:
  body_name = body.get("name", "")
  if body_name in _LINK_MESHES:
    children = list(body)
    insert_at = len(children)
    for i, child in enumerate(children):
      if child.tag in ("joint", "freejoint"):
        insert_at = i + 1
        break
    visual = ET.Element(
      "geom",
      {
        "type": "mesh",
        "mesh": body_name,
        "class": "visual",
        "rgba": "0.75294 0.75294 0.75294 1",
      },
    )
    body.insert(insert_at, visual)
  if body_name.endswith("_calf"):
    for geom in list(body):
      if geom.tag == "geom" and geom.get("name", "").endswith("_foot_collision"):
        ET.SubElement(
          body,
          "site",
          {
            "name": _prefix(body_name),
            "type": "sphere",
            "size": geom.get("size", "0.015"),
            "pos": geom.get("pos", "0 0 -0.1"),
            "group": "5",
          },
        )
  for child in list(body):
    if child.tag == "body":
      _insert_visuals_and_sites(child)


def convert(urdf_path: Path, output_path: Path) -> None:
  raw_xml = mujoco.MjSpec.from_file(str(urdf_path)).to_xml()
  root = ET.fromstring(raw_xml)

  compiler = root.find("compiler")
  if compiler is None:
    raise RuntimeError("URDF import did not produce a compiler element")
  compiler.set("angle", "radian")
  compiler.set("meshdir", "assets")
  compiler.set("autolimits", "true")

  worldbody = root.find("worldbody")
  if worldbody is None:
    raise RuntimeError("URDF import did not produce a worldbody")
  base_geom = worldbody.find("geom")
  if base_geom is None:
    raise RuntimeError("Expected a root collision geom from URDF import")
  worldbody.remove(base_geom)

  leg_bodies = [child for child in list(worldbody) if child.tag == "body"]
  for child in leg_bodies:
    worldbody.remove(child)

  base_body = ET.Element("body", {"name": "base_link", "childclass": "opendoge"})
  ET.SubElement(
    base_body,
    "inertial",
    {
      "pos": "0.0002 0.0020 0.0005",
      "mass": "1.6346",
      "diaginertia": "0.0027 0.0070 0.0091",
    },
  )
  ET.SubElement(base_body, "freejoint", {"name": "floating_base_joint"})
  base_geom.set("name", "base_collision")
  base_geom.set("class", "collision")
  base_body.append(base_geom)
  ET.SubElement(base_body, "site", {"name": "imu", "pos": "0 0 0.03"})
  for leg_body in leg_bodies:
    base_body.append(leg_body)
  worldbody.append(base_body)

  _name_collisions(base_body)
  _insert_visuals_and_sites(base_body)

  sensor = ET.Element("sensor")
  ET.SubElement(sensor, "gyro", {"name": "imu_ang_vel", "site": "imu"})
  ET.SubElement(sensor, "velocimeter", {"name": "imu_lin_vel", "site": "imu"})
  ET.SubElement(sensor, "accelerometer", {"name": "imu_lin_acc", "site": "imu"})
  ET.SubElement(sensor, "subtreeangmom", {"name": "root_angmom", "body": "base_link"})

  for child in list(root):
    root.remove(child)
  root.append(compiler)
  root.append(_build_defaults())
  root.append(_build_asset())
  root.append(worldbody)
  root.append(sensor)
  root.set("model", "opendoge")

  ET.indent(root, space="  ")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(ET.tostring(root, encoding="unicode") + "\n")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--urdf", type=Path, default=_DEFAULT_URDF)
  parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
  args = parser.parse_args()
  convert(args.urdf, args.output)
  # Compile immediately so asset path/name regressions fail at conversion time.
  mujoco.MjModel.from_xml_path(str(args.output))
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main()
