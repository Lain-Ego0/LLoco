from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_amp_cts_env_cfg,
  unitree_go2_amp_dreamwaq_env_cfg,
  unitree_go2_amp_ts_env_cfg,
  unitree_go2_amp_ts_student_env_cfg,
  unitree_go2_backflip_env_cfg,
  unitree_go2_cts_env_cfg,
  unitree_go2_dreamwaq_env_cfg,
  unitree_go2_flat_env_cfg,
  unitree_go2_handstand_env_cfg,
  unitree_go2_jump_env_cfg,
  unitree_go2_leggedstand_env_cfg,
  unitree_go2_rough_env_cfg,
  unitree_go2_spring_jump_env_cfg,
  unitree_go2_trot_env_cfg,
  unitree_go2_ts_env_cfg,
  unitree_go2_ts_student_env_cfg,
)
from .rl_cfg import unitree_go2_custom_runner_cfg, unitree_go2_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-Go2",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go2",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Trot-Flat-Unitree-Go2",
  env_cfg=unitree_go2_trot_env_cfg(),
  play_env_cfg=unitree_go2_trot_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

for _task_name, _task_cfg in (
  ("Jump", unitree_go2_jump_env_cfg),
  ("Spring-Jump", unitree_go2_spring_jump_env_cfg),
  ("Backflip", unitree_go2_backflip_env_cfg),
  ("Handstand", unitree_go2_handstand_env_cfg),
  ("Leggedstand", unitree_go2_leggedstand_env_cfg),
):
  register_mjlab_task(
    task_id=f"Mjlab-{_task_name}-Flat-Unitree-Go2",
    env_cfg=_task_cfg(),
    play_env_cfg=_task_cfg(play=True),
    rl_cfg=unitree_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )

for _task_name, _task_kind, _task_cfg in (
  ("DreamWaQ", "dreamwaq", unitree_go2_dreamwaq_env_cfg),
  ("AMP-DreamWaQ", "amp_dreamwaq", unitree_go2_amp_dreamwaq_env_cfg),
  ("CTS", "cts", unitree_go2_cts_env_cfg),
  ("AMP-CTS", "amp_cts", unitree_go2_amp_cts_env_cfg),
  ("AMP-TS", "amp_ts", unitree_go2_amp_ts_env_cfg),
  ("AMP-TS-Student", "amp_ts_student", unitree_go2_amp_ts_student_env_cfg),
  ("TS", "ts", unitree_go2_ts_env_cfg),
  ("TS-Student", "ts_student", unitree_go2_ts_student_env_cfg),
):
  register_mjlab_task(
    task_id=f"Mjlab-{_task_name}-Rough-Unitree-Go2",
    env_cfg=_task_cfg(),
    play_env_cfg=_task_cfg(play=True),
    rl_cfg=unitree_go2_custom_runner_cfg(_task_kind),
    runner_cls=VelocityOnPolicyRunner,
  )
