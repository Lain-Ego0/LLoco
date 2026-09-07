"""Registration for the migrated Gym ``go2_spring_jump`` task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from ..spring_jump.config import make_spring_jump_env_cfg, make_spring_jump_runner_cfg
from ..spring_jump.profile import SPRING_JUMP

register_mjlab_task(task_id=SPRING_JUMP.task_id, env_cfg=make_spring_jump_env_cfg(), play_env_cfg=make_spring_jump_env_cfg(play=True), rl_cfg=make_spring_jump_runner_cfg(), runner_cls=VelocityOnPolicyRunner)
