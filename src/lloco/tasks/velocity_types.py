"""Types and validation for shared velocity-task robot profiles."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from mjlab.entity import EntityCfg

RobotKind = Literal["quadruped", "humanoid"]
TerrainName = Literal["Flat", "Rough"]
ActionScale = float | dict[str, float]
CommandRange = tuple[float, float]
CommandRanges = tuple[CommandRange, CommandRange, CommandRange]

TASK_GROUP_UNITREE = "Unitree"
TASK_GROUP_LAINLAB = "LainLab"

_TASK_GROUP_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_TERRAINS = {"Flat", "Rough"}


@dataclass(frozen=True)
class VelocityScaling:
  """Explicit environment/action scaling for one robot profile.

  Keeping these values in a separate object avoids silently inheriting
  Go2-sized defaults when a new robot is added.
  """

  action_scale: ActionScale
  command_z_offset: float
  max_iterations: int
  viewer_distance: float
  foot_radius: float | None
  foot_sample_count: int | None
  clearance_target_height: float
  reset_height_range: tuple[float, float]
  command_ranges: CommandRanges
  play_command_ranges: CommandRanges
  command_velocity_stages: tuple[dict[str, object], ...] | None
  base_com_scale: float
  decimation: int


def quadruped_velocity_scaling() -> VelocityScaling:
  """Baseline scaling for the existing small/medium Unitree quadrupeds."""
  return VelocityScaling(
    action_scale=0.25,
    command_z_offset=0.5,
    max_iterations=10_000,
    viewer_distance=1.5,
    foot_radius=None,
    foot_sample_count=None,
    clearance_target_height=0.1,
    reset_height_range=(0.01, 0.05),
    command_ranges=((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)),
    play_command_ranges=((-1.0, 1.5), (-0.5, 0.5), (-0.7, 0.7)),
    command_velocity_stages=None,
    base_com_scale=1.0,
    decimation=4,
  )


def humanoid_velocity_scaling(
  action_scale: ActionScale,
  command_z_offset: float,
  max_iterations: int = 30_000,
) -> VelocityScaling:
  """Baseline scaling for the existing Unitree humanoids."""
  return VelocityScaling(
    action_scale=action_scale,
    command_z_offset=command_z_offset,
    max_iterations=max_iterations,
    viewer_distance=1.5,
    foot_radius=None,
    foot_sample_count=None,
    clearance_target_height=0.1,
    reset_height_range=(0.01, 0.05),
    command_ranges=((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)),
    play_command_ranges=((-1.0, 1.5), (-0.5, 0.5), (-0.7, 0.7)),
    command_velocity_stages=None,
    base_com_scale=1.0,
    decimation=4,
  )


@dataclass(frozen=True)
class VelocityRobotProfile:
  """Robot-specific names and tuning layered on mjlab's shared velocity task."""

  task_name: str
  robot_cfg: Callable[[], EntityCfg]
  kind: RobotKind
  root_body: str
  viewer_body: str
  foot_sites: tuple[str, ...]
  foot_geoms: tuple[str, ...]
  foot_contact_pattern: str | tuple[str, ...]
  scaling: VelocityScaling
  task_group: str = ""
  terrains: tuple[TerrainName, ...] = ("Flat", "Rough")

  def __post_init__(self) -> None:
    if not self.task_group:
      raise ValueError(f"Velocity profile {self.task_name!r} must declare a task_group")
    if not _TASK_GROUP_RE.fullmatch(self.task_group):
      raise ValueError(
        f"Invalid task_group {self.task_group!r}; expected letters, digits, "
        "underscores, or hyphens"
      )
    if not self.terrains:
      raise ValueError(f"Velocity profile {self.task_name!r} must declare terrains")
    if len(set(self.terrains)) != len(self.terrains):
      raise ValueError(f"Velocity profile {self.task_name!r} has duplicate terrains")
    invalid = set(self.terrains) - _TERRAINS
    if invalid:
      raise ValueError(
        f"Velocity profile {self.task_name!r} has unknown terrains: {sorted(invalid)}"
      )

  @property
  def action_scale(self) -> ActionScale:
    return self.scaling.action_scale

  @property
  def command_z_offset(self) -> float:
    return self.scaling.command_z_offset

  @property
  def max_iterations(self) -> int:
    return self.scaling.max_iterations

  @property
  def viewer_distance(self) -> float:
    return self.scaling.viewer_distance

  @property
  def foot_radius(self) -> float | None:
    return self.scaling.foot_radius

  @property
  def foot_sample_count(self) -> int | None:
    return self.scaling.foot_sample_count

  @property
  def clearance_target_height(self) -> float:
    return self.scaling.clearance_target_height

  @property
  def reset_height_range(self) -> tuple[float, float]:
    return self.scaling.reset_height_range

  @property
  def command_ranges(self) -> CommandRanges:
    return self.scaling.command_ranges

  @property
  def play_command_ranges(self) -> CommandRanges:
    return self.scaling.play_command_ranges

  @property
  def command_velocity_stages(self) -> tuple[dict[str, object], ...] | None:
    return self.scaling.command_velocity_stages

  @property
  def base_com_scale(self) -> float:
    return self.scaling.base_com_scale

  @property
  def decimation(self) -> int:
    return self.scaling.decimation
