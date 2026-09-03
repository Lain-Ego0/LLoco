"""Go2 skill catalog without algorithm or optimizer parameters."""

from lainloco.core import Catalog, TaskSpec


def _task(
  task_id: str,
  family: str,
  terrain: str,
  observation: str,
  episode_length_s: float = 20.0,
) -> TaskSpec:
  return TaskSpec(
    task_id=task_id,
    robot_id="unitree/go2",
    family=family,
    terrain_profile=terrain,
    command_profile=f"go2/{family}",
    observation_profile=observation,
    reward_profile=f"go2/{family}",
    termination_profile=f"go2/{family}",
    randomization_profile="go2/source-compatible",
    episode_length_s=episode_length_s,
  )


GO2_TASKS = Catalog(
  (
    _task("go2/velocity-flat", "locomotion", "flat", "velocity"),
    _task("go2/velocity-rough", "locomotion", "rough", "velocity"),
    _task("go2/trot", "locomotion", "flat", "trot-history"),
    _task("go2/jump", "aerial", "stairs", "jump-history", 24.0),
    _task("go2/spring-jump", "aerial", "flat", "spring-jump-history", 5.0),
    _task("go2/backflip", "aerial", "flat", "backflip-history", 4.0),
    _task("go2/handstand", "balance", "flat", "balance"),
    _task("go2/leggedstand", "balance", "flat", "balance"),
  ),
  id_of=lambda task: task.task_id,
)
