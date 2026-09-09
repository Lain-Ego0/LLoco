"""Small inspection adapter using MuJoCo's native URDF/MJCF compiler.

This module is loaded on demand and does not import the training environment.
"""

import base64
import posixpath
import threading
import uuid
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path, PurePosixPath

from .services import inside


class Models:
  def __init__(self):
    self.lock = threading.RLock()
    self.models = OrderedDict()

  def load(self, root: Path, request: dict) -> dict:
    import mujoco as mj

    if request.get("path"):
      path = inside(root, request["path"])
      if path.suffix.lower() not in (".urdf", ".xml"):
        raise ValueError("请选择 URDF 或 MJCF XML")
      base = path.parent.parent if path.parent.name in ("xmls", "urdf") else path.parent
      selected = path.relative_to(base).as_posix()
      files = {
        p.relative_to(base).as_posix(): p.read_bytes()
        for p in base.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".xml", ".urdf", ".stl", ".obj", ".png"}
        and p.resolve().is_relative_to(root)
      }
    else:
      selected = request.get("selected", "")
      files = {}
      for entry in request.get("files", []):
        name = entry["name"]
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
          raise ValueError("导入资源路径不能包含上级目录")
        files[name] = base64.b64decode(entry["data"], validate=True)
    if selected not in files:
      raise ValueError("未找到选中的模型文件")
    # Preserve visual and fixed-link geometry when importing URDF.
    try:
      xml = ET.fromstring(files[selected])
    except ET.ParseError as error:
      raise ValueError(f"XML 格式无效：{error}") from error
    if xml.tag == "robot":
      extension = xml.find("mujoco")
      if extension is None:
        extension = ET.SubElement(xml, "mujoco")
      compiler = extension.find("compiler")
      if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
      compiler.set("discardvisual", "false")
      compiler.set("fusestatic", "false")
      files[selected] = ET.tostring(xml)
    elif xml.tag != "mujoco":
      raise ValueError("仅支持 URDF <robot> 或 MJCF <mujoco>")
    # Resolve virtual resource names relative to the chosen entry XML.
    assets = {
      posixpath.relpath(name, posixpath.dirname(selected) or "."): content
      for name, content in files.items()
    }
    model = mj.MjModel.from_xml_string(files[selected].decode(), assets=assets)
    data = mj.MjData(model)
    if model.nkey:
      mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_forward(model, data)
    identifier = uuid.uuid4().hex
    with self.lock:
      self.models[identifier] = (model, data, data.qpos.copy())
      while len(self.models) > 8:
        self.models.popitem(last=False)
      result = self._pose(identifier)
      result.update(name=Path(selected).name, meshes={}, joints=[])
      for index in range(model.nmesh):
        va, vn = model.mesh_vertadr[index], model.mesh_vertnum[index]
        fa, fn = model.mesh_faceadr[index], model.mesh_facenum[index]
        result["meshes"][str(index)] = dict(
          vertices=model.mesh_vert[va : va + vn].tolist(),
          faces=model.mesh_face[fa : fa + fn].tolist(),
        )
      for index in range(model.njnt):
        kind = int(model.jnt_type[index])
        if kind not in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
          continue
        limited = model.jnt_limited[index]
        result["joints"].append(
          dict(
            id=index,
            name=mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, index) or f"joint {index}",
            slide=kind == int(mj.mjtJoint.mjJNT_SLIDE),
            limits=model.jnt_range[index].tolist() if limited else [-3.14159, 3.14159],
            value=float(data.qpos[model.jnt_qposadr[index]]),
          )
        )
      return result

  def pose(self, request: dict) -> dict:
    import mujoco as mj
    import numpy as np

    with self.lock:
      identifier = request.get("id")
      if identifier not in self.models:
        raise ValueError("模型会话已过期，请重新加载")
      model, data, initial = self.models[identifier]
      if request.get("reset"):
        data.qpos[:] = initial
      elif "joint" in request:
        joint, value = int(request["joint"]), float(request["value"])
        if not 0 <= joint < model.njnt or not np.isfinite(value):
          raise ValueError("无效关节值")
        if model.jnt_type[joint] not in (
          mj.mjtJoint.mjJNT_HINGE,
          mj.mjtJoint.mjJNT_SLIDE,
        ):
          raise ValueError("仅支持转动和平移关节")
        if model.jnt_limited[joint]:
          value = float(np.clip(value, *model.jnt_range[joint]))
        data.qpos[model.jnt_qposadr[joint]] = value
      mj.mj_forward(model, data)
      self.models.move_to_end(identifier)
      return self._pose(identifier)

  def _pose(self, identifier):
    import mujoco as mj
    import numpy as np

    model, data, _ = self.models[identifier]
    geoms = []
    for i in range(model.ngeom):
      geoms.append(
        dict(
          id=i,
          type=int(model.geom_type[i]),
          mesh=int(model.geom_dataid[i]),
          size=model.geom_size[i].tolist(),
          color=model.geom_rgba[i].tolist(),
          collision=bool(model.geom_contype[i] or model.geom_conaffinity[i]),
          body=int(model.geom_bodyid[i]),
          position=data.geom_xpos[i].tolist(),
          rotation=data.geom_xmat[i].reshape(3, 3).tolist(),
        )
      )
    bodies = []
    for i in range(1, model.nbody):
      inertia, mass = model.body_inertia[i], float(model.body_mass[i])
      radii = (
        np.sqrt(np.maximum(0, 5 * (inertia.sum() - 2 * inertia) / (2 * mass)))
        if mass > 0
        else np.zeros(3)
      )
      bodies.append(
        dict(
          id=i,
          name=mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or f"body {i}",
          mass=mass,
          inertia=inertia.tolist(),
          radii=radii.tolist(),
          position=data.xpos[i].tolist(),
          rotation=data.xmat[i].reshape(3, 3).tolist(),
          com=data.xipos[i].tolist(),
          inertia_rotation=data.ximat[i].reshape(3, 3).tolist(),
        )
      )
    axes = [
      dict(
        id=i,
        anchor=data.xanchor[i].tolist(),
        axis=data.xaxis[i].tolist(),
        value=float(data.qpos[model.jnt_qposadr[i]]),
      )
      for i in range(model.njnt)
      if model.jnt_type[i] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE)
    ]
    return dict(id=identifier, geoms=geoms, bodies=bodies, axes=axes)
