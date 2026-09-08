from mjlab.tasks.registry import register_mjlab_task

from ..cts.config import make_cts_env_cfg, make_cts_runner_cfg
from ..cts.profile import CTS

register_mjlab_task(
  task_id=CTS.task_id,
  env_cfg=make_cts_env_cfg(),
  play_env_cfg=make_cts_env_cfg(play=True),
  rl_cfg=make_cts_runner_cfg(),
)
