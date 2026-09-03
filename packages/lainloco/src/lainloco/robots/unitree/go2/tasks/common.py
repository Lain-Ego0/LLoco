"""Shared helpers for source-compatible Go2 task factories."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.utils.noise.noise_cfg import UniformNoiseCfg


def _go2_source_observation_clipping(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply the source environment's final [-100, 100] observation clip."""
  for group in cfg.observations.values():
    for term in group.terms.values():
      term.clip = (-100.0, 100.0)


def _go2_source_geom_friction(
  cfg: ManagerBasedRlEnvCfg,
  ranges: tuple[float, float],
) -> None:
  """Match Isaac Gym's shared friction sample across every Go2 shape.

  The source ``_process_rigid_shape_props`` callback assigns one scalar
  coefficient to every rigid shape in an environment.  MuJoCo's first geom
  friction component is the closest equivalent.  Do not retain the generic
  torsional/rolling randomizers here: those are useful for the public mjlab
  baseline, but are not sampled by the Go2 source tasks.
  """
  event = cfg.events.get("foot_friction_slide")
  if event is None:
    return
  asset_cfg = event.params["asset_cfg"]
  asset_cfg.geom_names = (r".*",)
  event.params.update(
    {
      "ranges": ranges,
      "axes": [0],
      "shared_random": True,
    }
  )
  cfg.events.pop("foot_friction_spin", None)
  cfg.events.pop("foot_friction_roll", None)


def _go2_source_noise_cfg(
  *, command_first: bool, dof_pos_noise: float = 0.01, ang_vel_noise: float = 0.2
) -> UniformNoiseCfg:
  """Return the source uniform observation-noise vector for a 45-D frame."""
  # Source noise is sampled in [-scale, scale].  The stand tasks place IMU,
  # gravity, then command; CTS/DreamWaQ/TS place command first.
  imu = [ang_vel_noise * 0.25] * 3
  gravity = [0.05] * 3
  command = [0.0] * 3
  q = [dof_pos_noise] * 12
  dq = [1.5 * 0.05] * 12
  actions = [0.0] * 12
  values = (
    command + imu + gravity + q + dq + actions
    if command_first
    else imu + gravity + command + q + dq + actions
  )
  return UniformNoiseCfg(
    n_min=tuple(-value for value in values),
    n_max=tuple(values),
  )


def _go2_source_47_noise_cfg() -> UniformNoiseCfg:
  """Return the source uniform noise vector for phase-based 47-D frames."""
  values = (
    [0.0] * 5
    + [0.2 * 0.25] * 3
    + [0.1] * 3
    + [0.01] * 12
    + [1.5 * 0.05] * 12
    + [0.0] * 12
  )
  return UniformNoiseCfg(n_min=tuple(-value for value in values), n_max=tuple(values))
