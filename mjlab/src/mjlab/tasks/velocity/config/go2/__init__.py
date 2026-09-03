from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityDistillationRunner, VelocityOnPolicyRunner

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
from .rl_cfg import (
  unitree_go2_custom_runner_cfg,
  unitree_go2_ppo_runner_cfg,
  unitree_go2_source_ppo_runner_cfg,
)

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
  rl_cfg=unitree_go2_source_ppo_runner_cfg(
    learning_rate=1.0e-5,
    symmetry_func=(
      "mjlab.tasks.velocity.rl.go2_algorithms.symmetry.trot_jump_symmetry"
    ),
  ),
  runner_cls=VelocityOnPolicyRunner,
)

for _task_name, _task_cfg, _learning_rate, _max_iterations, _symmetry_func in (
  (
    "Jump", unitree_go2_jump_env_cfg, 1.0e-4, 15_000,
    "mjlab.tasks.velocity.rl.go2_algorithms.symmetry.trot_jump_symmetry",
  ),
  (
    "Spring-Jump", unitree_go2_spring_jump_env_cfg, 1.0e-5, 50_000,
    "mjlab.tasks.velocity.rl.go2_algorithms.symmetry.spring_jump_symmetry",
  ),
  ("Backflip", unitree_go2_backflip_env_cfg, 1.0e-5, 50_000, None),
  (
    "Handstand", unitree_go2_handstand_env_cfg, 1.0e-3, 15_000,
    "mjlab.tasks.velocity.rl.go2_algorithms.symmetry.handstand_symmetry",
  ),
  ("Leggedstand", unitree_go2_leggedstand_env_cfg, 1.0e-3, 15_000, None),
):
  register_mjlab_task(
    task_id=f"Mjlab-{_task_name}-Flat-Unitree-Go2",
    env_cfg=_task_cfg(),
    play_env_cfg=_task_cfg(play=True),
    rl_cfg=unitree_go2_source_ppo_runner_cfg(
      learning_rate=_learning_rate,
      max_iterations=_max_iterations,
      symmetry_func=_symmetry_func,
    ),
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
    runner_cls=(
      VelocityDistillationRunner
      if _task_kind in ("amp_ts_student", "ts_student")
      else VelocityOnPolicyRunner
    ),
  )
