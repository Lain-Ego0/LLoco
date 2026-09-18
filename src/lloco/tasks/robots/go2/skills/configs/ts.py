from mjlab.tasks.registry import register_mjlab_task

from ..ts.config import make_ts_env_cfg, make_ts_runner_cfg
from ..ts.profile import TS

register_mjlab_task(
  task_id=TS.task_id,
  env_cfg=make_ts_env_cfg(),
  play_env_cfg=make_ts_env_cfg(play=True),
  rl_cfg=make_ts_runner_cfg(),
)
