"""G1 skill catalog independent of training configuration."""

from lainloco.core import Catalog, TaskSpec


def _velocity_task(task_id: str, terrain: str) -> TaskSpec:
  return TaskSpec(
    task_id=task_id,
    robot_id="unitree/g1",
    family="locomotion",
    terrain_profile=terrain,
    command_profile="g1/base-velocity",
    observation_profile=f"g1/velocity-{terrain}",
    reward_profile="g1/velocity",
    termination_profile="g1/locomotion",
    randomization_profile="g1/mjlab-1.6",
    episode_length_s=20.0,
  )


G1_TASKS = Catalog(
  (
    _velocity_task("g1/velocity-flat", "flat"),
    _velocity_task("g1/velocity-rough", "rough"),
  ),
  id_of=lambda task: task.task_id,
)
