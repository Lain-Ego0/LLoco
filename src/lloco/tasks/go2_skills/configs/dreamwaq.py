"""Register the source DreamWaQ task without affecting Flat/Rough tasks."""

from mjlab.tasks.registry import register_mjlab_task

from ..dreamwaq.config import make_dreamwaq_env_cfg, make_dreamwaq_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-DreamWaQ-Rough",
  env_cfg=make_dreamwaq_env_cfg(),
  play_env_cfg=make_dreamwaq_env_cfg(play=True),
  rl_cfg=make_dreamwaq_runner_cfg(),
)
