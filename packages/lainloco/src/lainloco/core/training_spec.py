"""Learning strategy selection independent of a concrete task."""

from dataclasses import dataclass

from ._validation import require_text


@dataclass(frozen=True, slots=True)
class TrainingSpec:
  """Algorithm and runner selection, independent of a concrete robot task."""

  profile_id: str
  algorithm: str
  actor_model: str
  critic_model: str
  storage: str
  runner: str
  optimizer: str
  auxiliary_losses: tuple[str, ...]
  required_observation_groups: tuple[str, ...]
  exporter: str

  def __post_init__(self) -> None:
    require_text(self.profile_id, "profile_id")
    require_text(self.algorithm, "algorithm")
    if len(set(self.required_observation_groups)) != len(
      self.required_observation_groups
    ):
      raise ValueError("required_observation_groups must be unique")
