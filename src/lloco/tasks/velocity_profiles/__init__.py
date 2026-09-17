"""Built-in velocity robot profile registry."""

from .opendoge import OPENDOGE_PROFILES
from .unitree import UNITREE_PROFILES

PROFILES = (*UNITREE_PROFILES, *OPENDOGE_PROFILES)

__all__ = ["OPENDOGE_PROFILES", "PROFILES", "UNITREE_PROFILES"]
