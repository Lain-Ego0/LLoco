"""Shared Go2 contact-sensor configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

FEET_SENSOR = "feet_ground_contact"
PENALIZED_SENSOR = "thigh_calf_ground_contact"
BASE_SENSOR = "base_ground_contact"


def replace_sensors(cfg: ManagerBasedRlEnvCfg) -> None:
  terrain = ContactMatch(mode="body", pattern="terrain")
  feet = ContactSensorCfg(
    name=FEET_SENSOR,
    primary=ContactMatch(
      mode="geom",
      pattern=(
        "FL_foot_collision",
        "FR_foot_collision",
        "RL_foot_collision",
        "RR_foot_collision",
      ),
      entity="robot",
    ),
    secondary=terrain,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  penalized = ContactSensorCfg(
    name=PENALIZED_SENSOR,
    primary=ContactMatch(
      mode="geom", pattern=r".*_(thigh|calf)_collision", entity="robot"
    ),
    secondary=terrain,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  base = ContactSensorCfg(
    name=BASE_SENSOR,
    primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
    secondary=terrain,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (feet, penalized, base)


def replace_rear_stand_sensors(cfg: ManagerBasedRlEnvCfg) -> None:
  # Source asset.penalize_contacts_on contains only thigh and calf. Base contact
  # is handled separately as a termination and hip contact is not penalized.
  replace_sensors(cfg)
