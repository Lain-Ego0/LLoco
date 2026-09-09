"""Convert General Motion Retargeting (GMR) output into LLoco motion data."""

import pickle
from pathlib import Path

import numpy as np

from lloco.motion_conversion import convert_csv_to_npz


def convert_gmr_to_npz(
  input_file: str, output_name: str, *, output_fps: float = 50.0, device: str = "cuda:0"
) -> Path:
  """Turn a Unitree G1 GMR pickle into the NPZ used by tracking tasks.

  GMR stores root quaternions as XYZW, exactly the CSV convention consumed by
  LLoco's established converter.  Its Unitree-G1 model has 29 actuated DoFs,
  so the bridge deliberately rejects other robot variants instead of silently
  producing a policy with mismatched joint ordering.
  """
  source = Path(input_file).expanduser().resolve()
  if source.suffix != ".pkl":
    raise ValueError("GMR input must be a .pkl file")
  with source.open("rb") as file:
    motion = pickle.load(file)
  required = {"fps", "root_pos", "root_rot", "dof_pos"}
  missing = required - set(motion)
  if missing:
    raise ValueError(f"Not a GMR motion; missing: {', '.join(sorted(missing))}")
  root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
  root_rot = np.asarray(motion["root_rot"], dtype=np.float32)
  dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
  if (
    root_pos.ndim != 2 or root_pos.shape[1] != 3 or root_rot.shape != (len(root_pos), 4)
  ):
    raise ValueError("GMR root_pos/root_rot must have shapes (frames, 3)/(frames, 4)")
  if dof_pos.shape != (len(root_pos), 29):
    raise ValueError(
      "Only GMR's Unitree G1 (29 DoF) output is supported initially; "
      f"received dof_pos shape {dof_pos.shape}."
    )
  fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
  if fps <= 0:
    raise ValueError("GMR motion fps must be positive")
  csv_path = source.with_suffix(".lloco-gmr.csv")
  np.savetxt(
    csv_path, np.concatenate((root_pos, root_rot, dof_pos), axis=1), delimiter=","
  )
  try:
    return convert_csv_to_npz(
      robot="g1",
      input_file=str(csv_path),
      output_name=output_name,
      input_fps=fps,
      output_fps=output_fps,
      device=device,
    )
  finally:
    csv_path.unlink(missing_ok=True)


def main(
  input_file: str, output_name: str, output_fps: float = 50.0, device: str = "cuda:0"
) -> None:
  """CLI entrypoint for a Unitree G1 GMR pickle."""
  output = convert_gmr_to_npz(
    input_file, output_name, output_fps=output_fps, device=device
  )
  print(f"[INFO] GMR motion converted: {output}")
