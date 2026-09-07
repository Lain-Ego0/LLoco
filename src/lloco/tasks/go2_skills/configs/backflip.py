from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from ..backflip.config import make_backflip_env_cfg,make_backflip_runner_cfg
from ..backflip.profile import BACKFLIP
register_mjlab_task(task_id=BACKFLIP.task_id,env_cfg=make_backflip_env_cfg(),play_env_cfg=make_backflip_env_cfg(play=True),rl_cfg=make_backflip_runner_cfg(),runner_cls=VelocityOnPolicyRunner)
