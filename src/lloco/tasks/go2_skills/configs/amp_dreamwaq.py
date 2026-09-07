"""Registration for the local AMP-DreamWaQ task."""

from mjlab.tasks.registry import register_mjlab_task

from ..amp_dreamwaq.config import make_amp_dreamwaq_env_cfg, make_amp_dreamwaq_runner_cfg

register_mjlab_task(task_id="Unitree-Go2-AMP-DreamWaQ-Rough", env_cfg=make_amp_dreamwaq_env_cfg(), play_env_cfg=make_amp_dreamwaq_env_cfg(play=True), rl_cfg=make_amp_dreamwaq_runner_cfg())
