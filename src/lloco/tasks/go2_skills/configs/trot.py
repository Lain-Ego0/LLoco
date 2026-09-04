"""Registration for the migrated Go2 trot task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from ..common import make_trot_env_cfg, make_trot_runner_cfg
from ..profiles import TROT

register_mjlab_task(
  task_id=TROT.task_id,
  env_cfg=make_trot_env_cfg(),
  play_env_cfg=make_trot_env_cfg(play=True),
  rl_cfg=make_trot_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
