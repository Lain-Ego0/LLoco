"""Registration for the migrated Go2 jump task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from ..common import make_jump_env_cfg, make_jump_runner_cfg
from ..profiles import JUMP

register_mjlab_task(
  task_id=JUMP.task_id,
  env_cfg=make_jump_env_cfg(),
  play_env_cfg=make_jump_env_cfg(play=True),
  rl_cfg=make_jump_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
