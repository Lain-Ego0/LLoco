"""Robot skill semantics independent of learning configuration."""

from dataclasses import dataclass

from ._validation import require_text


@dataclass(frozen=True, slots=True)
class TaskSpec:
  """A robot skill and its environment semantics, independent of learning."""

  task_id: str
  robot_id: str
  family: str
  terrain_profile: str
  command_profile: str
  observation_profile: str
  reward_profile: str
  termination_profile: str
  randomization_profile: str
  episode_length_s: float

  def __post_init__(self) -> None:
    require_text(self.task_id, "task_id")
    require_text(self.robot_id, "robot_id")
    if self.episode_length_s <= 0:
      raise ValueError("episode_length_s must be positive")
