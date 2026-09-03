"""Robot facts that remain stable across tasks and training profiles."""

from dataclasses import dataclass

from ._validation import require_text


@dataclass(frozen=True, slots=True)
class RobotSpec:
  """Robot facts that do not change between tasks or training profiles."""

  robot_id: str
  asset_factory: str
  joint_order: tuple[str, ...]
  base_body: str
  foot_sites: tuple[str, ...]
  collision_geoms: tuple[str, ...]
  default_pose: tuple[tuple[str, float], ...]
  action_scale: tuple[float, ...]
  physics_dt: float
  control_dt: float
  hardware_joint_mapping: tuple[int, ...] = ()

  def __post_init__(self) -> None:
    require_text(self.robot_id, "robot_id")
    if not self.joint_order or len(set(self.joint_order)) != len(self.joint_order):
      raise ValueError("joint_order must contain unique joint names")
    if len(self.action_scale) != len(self.joint_order):
      raise ValueError("action_scale must have one value per joint")
    if self.hardware_joint_mapping and sorted(self.hardware_joint_mapping) != list(
      range(len(self.joint_order))
    ):
      raise ValueError("hardware_joint_mapping must be a joint-order permutation")
    if self.physics_dt <= 0 or self.control_dt <= 0:
      raise ValueError("physics_dt and control_dt must be positive")
    ratio = self.control_dt / self.physics_dt
    if abs(ratio - round(ratio)) > 1e-9:
      raise ValueError("control_dt must be an integer multiple of physics_dt")

  @property
  def action_dim(self) -> int:
    return len(self.joint_order)
