"""Registration for the migrated Go2 rear-stand task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from ..rear_stand.config import make_rear_stand_env_cfg, make_rear_stand_runner_cfg
from ..rear_stand.profile import REAR_STAND

register_mjlab_task(
  task_id=REAR_STAND.task_id,
  env_cfg=make_rear_stand_env_cfg(),
  play_env_cfg=make_rear_stand_env_cfg(play=True),
  rl_cfg=make_rear_stand_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
