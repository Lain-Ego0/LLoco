"""Go2 robot task composition package."""

from . import skills as skills
from . import velocity as velocity

__all__ = ["skills", "velocity"]


def register_tasks() -> None:
  """Register Go2 velocity tasks; skills register on import."""
  velocity.register_velocity_tasks()
