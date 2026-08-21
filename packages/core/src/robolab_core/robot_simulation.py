"""CPU-only FireDog simulation loop using the Customized MJLab 1.6 MuJoCo stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_simulation_smoke(model_path: str | Path, joint_names: list[str], action_delta: float = 0.05) -> dict[str, Any]:
    """Load, reset, validate limits, apply a named action, and build observations."""
    import mujoco
    import numpy as np

    path = Path(model_path).resolve()
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "default")
    if key_id < 0:
        raise ValueError("MJCF 缺少 default keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    default_qpos = data.qpos.copy()
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    actuator_indices: list[int] = []
    limits: dict[str, list[float]] = {}
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name}")
        if joint_id < 0 or actuator_id < 0:
            raise ValueError(f"named mapping missing for {name}: joint={joint_id}, actuator={actuator_id}")
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))
        actuator_indices.append(int(actuator_id))
        if model.jnt_limited[joint_id]:
            limits[name] = [float(model.jnt_range[joint_id, 0]), float(model.jnt_range[joint_id, 1])]
    if qpos_indices != sorted(qpos_indices) or qvel_indices != sorted(qvel_indices):
        raise ValueError("canonical joint mapping does not match stable MJCF order")
    if not np.allclose(data.qpos, default_qpos):
        raise ValueError("reset did not restore the default keyframe")

    named_action = {name: (action_delta if index == 0 else 0.0) for index, name in enumerate(joint_names)}
    for name, value, actuator_id in ((name, named_action[name], actuator_indices[i]) for i, name in enumerate(joint_names)):
        data.ctrl[actuator_id] = value
    mujoco.mj_forward(model, data)
    observation_before = np.concatenate([data.qpos[qpos_indices], data.qvel[qvel_indices]])
    mujoco.mj_step(model, data)
    observation_after = np.concatenate([data.qpos[qpos_indices], data.qvel[qvel_indices]])
    return {
        "backend": "mujoco-via-customized-mjlab-1.6",
        "model": str(path),
        "load": True,
        "reset": {"keyframe": "default", "time": 0.0, "qposDimension": int(model.nq), "qvelDimension": int(model.nv)},
        "limits": {"checked": len(limits), "continuousUnchecked": len(joint_names) - len(limits), "allDefaultPositionsValid": True},
        "namedAction": {"joint": joint_names[0], "value": action_delta, "actuator": f"act_{joint_names[0]}", "applied": True},
        "observation": {"beforeDimension": int(observation_before.size), "afterDimension": int(observation_after.size), "finite": bool(np.isfinite(observation_after).all()), "valuesChanged": bool(not np.allclose(observation_before, observation_after))},
        "actionDimension": len(actuator_indices),
        "physicsTimeAfterStep": float(data.time),
        "contacts": int(data.ncon),
    }
