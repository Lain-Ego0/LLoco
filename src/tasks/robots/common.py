"""Shared constants for built-in velocity robot profiles."""

_QUAD_FEET = ("FR", "FL", "RR", "RL")
_QUAD_GEOMS = tuple(f"{name}_foot_collision" for name in _QUAD_FEET)
_HUMANOID_SITES = ("left_foot", "right_foot")
_HUMANOID_GEOMS = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)
